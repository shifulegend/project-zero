/**
 * gguf_quant.c — Phase 34.2b
 *
 * Dequantization for GGUF k-quant formats used by standard float models
 * (DeepSeek, Llama, Mistral, etc.).
 *
 * Per CPU_LLM_TERNARY_ENGINE.md §"Category 3: Standard Models":
 *   Standard FP16/BF16 models must NOT be ternary-quantized (causes intelligence loss).
 *   Q4_K and Q8_0 are the correct PTQ formats — they mathematically scale weights
 *   to 4/8-bit while preserving model quality.
 *
 * Block layout references: llama.cpp ggml-quants.h (public domain / MIT).
 */
#include "core/gguf_quant.h"
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── FP16 helper ──────────────────────────────────────────────────────────── */
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t u;
    if (exp == 0x1fu) {                          /* inf / NaN */
        u = sign | 0x7f800000u | (mant << 13);
    } else if (exp == 0u) {
        if (mant == 0u) {                        /* signed zero */
            u = sign;
        } else {                                 /* subnormal → normalise */
            exp = 1u;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            u = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else {                                     /* normal */
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

/* ── Q8_0 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (34 bytes per 32 elements):
 *   [d: fp16 (2 bytes)] [qs: int8 × 32 (32 bytes)]
 * Decode: out[i] = qs[i] * fp16_to_f32(d)
 */
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BYTES_PER_BLOCK 34   /* 2 (scale) + 32 (int8) */

void gguf_dequant_q8_0(float *out, const void *data, size_t n_elems) {
    const uint8_t *p    = (const uint8_t *)data;
    size_t n_blocks      = n_elems / Q8_0_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q8_0_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float scale = fp16_to_f32(d_bits);
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *dst = out + b * Q8_0_BLOCK_SIZE;
        for (int i = 0; i < Q8_0_BLOCK_SIZE; i++)
            dst[i] = qs[i] * scale;
    }
    /* Handle partial last block (rare; n_elems usually a multiple of 32) */
    size_t rem = n_elems - n_blocks * Q8_0_BLOCK_SIZE;
    if (rem > 0) {
        const uint8_t *blk = p + n_blocks * Q8_0_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float scale = fp16_to_f32(d_bits);
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *dst = out + n_blocks * Q8_0_BLOCK_SIZE;
        for (size_t i = 0; i < rem; i++) dst[i] = qs[i] * scale;
    }
}

/* ── Q4_K ─────────────────────────────────────────────────────────────────── */
/*
 * K-quant super-block layout (144 bytes per 256 elements):
 *   [d:    fp16 (2 bytes)] — super-block scale
 *   [dmin: fp16 (2 bytes)] — super-block min
 *   [scales: 12 bytes]     — 8 sub-blocks × (6-bit scale + 6-bit min) packed
 *   [qs:   128 bytes]      — 256 × 4-bit values, two nibbles per byte
 *
 * Scale/min extraction from the 12-byte packed field (ggml convention):
 *   Each sub-block j has:
 *     sc  = scales[j/2]       & (j%2 == 0 ? 0x3F : 0x0F) | ((scales[j/2+4] >> (j%2==0 ? 0 : 4)) << 4) & 0x30
 *     m   = scales[4 + j/2]   ... (complex bit packing, see decode below)
 *
 * Simpler unpacking: ggml stores scales as two 6-bit fields per byte:
 *   For sub-block j (0..7):
 *     byte_idx   = j * 3 / 4   — but the actual layout is:
 *       bytes 0..5:  lower 4 bits = scale_low[j], upper 4 bits = min_low[j]
 *       bytes 8..11: upper 2 bits of scale and min packed
 *
 * Using the exact llama.cpp layout (from ggml-quants.h):
 *   uint8_t scales[12]:
 *     bits[ 0.. 5] of scales[0] = sc[0] low 6 bits
 *     bits[ 6..11] of scales[0..1] cross-byte = sc[1] low 6 bits
 *     ...
 * This is complex; use the proven decode from ggml reference:
 */

#define Q4_K_SUPER  256
#define Q4_K_NSUB   8
#define Q4_K_SUB    32   /* elements per sub-block */
#define Q4_K_BYTES  144  /* bytes per 256-element super-block */

/* Decode 8 scale+min pairs from the 12-byte packed field.
 * Based on ggml-quants.h make_qkx2_quants / dequantize_row_q4_K. */
static void q4k_decode_scales(const uint8_t *sc12,
                               float d, float dmin,
                               float *scales, float *mins) {
    for (int j = 0; j < Q4_K_NSUB; j++) {
        uint8_t sc, m;
        if (j < 4) {
            sc = sc12[j]     & 63;
            m  = sc12[j + 4] & 63;
        } else {
            /* Exact ggml formula from get_scale_min_k4():
             *   d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
             *   m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4)
             * where k=j-4, so q[j-4]=sc12[k], q[j]=sc12[k+4], q[j+4]=sc12[k+8] */
            int k = j - 4;
            sc = (sc12[k + 8] & 0x0F) | ((sc12[k]     >> 6) << 4);
            m  = (sc12[k + 8] >>    4) | ((sc12[k + 4] >> 6) << 4);
        }
        scales[j] = sc * d;
        mins[j]   = m  * dmin;
    }
}

void gguf_dequant_q4_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p  = (const uint8_t *)data;
    size_t n_super     = n_elems / Q4_K_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * Q4_K_BYTES;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d    = fp16_to_f32(d_bits);
        float dmin = fp16_to_f32(dmin_bits);

        const uint8_t *sc12 = blk + 4;        /* 12 bytes */
        const uint8_t *qs   = blk + 4 + 12;   /* 128 bytes (256 × 4-bit) */

        float scales[Q4_K_NSUB], mins[Q4_K_NSUB];
        q4k_decode_scales(sc12, d, dmin, scales, mins);

        float *dst = out + b * Q4_K_SUPER;

        /* Exact replica of llama.cpp dequantize_row_q4_K():
         * Four groups of 64 elements each. Each group reads 32 bytes (=64 nibbles):
         *   first 32 output elems = low  nibbles of qs[g*32 .. g*32+31]  (scale pair 2g)
         *   next  32 output elems = high nibbles of qs[g*32 .. g*32+31]  (scale pair 2g+1)
         * This is NOT the same as 8 independent 16-byte sub-blocks. */
        for (int g = 0; g < 4; g++) {
            int    is   = g * 2;
            float  d1   = scales[is],     m1 = mins[is];
            float  d2   = scales[is + 1], m2 = mins[is + 1];
            const uint8_t *q = qs + g * 32;   /* 32 bytes per 64-element group */
            float *yd   = dst + g * 64;
            for (int l = 0; l < 32; l++) yd[l]      = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) yd[32 + l] = d2 * (float)(q[l] >>  4) - m2;
        }
    }
    /* Partial last super-block (very rare) */
    size_t done = n_super * Q4_K_SUPER;
    if (done < n_elems) {
        /* Fill remaining with 0 — correct since partial blocks are unusual in practice */
        size_t rem = n_elems - done;
        for (size_t i = 0; i < rem; i++) out[done + i] = 0.0f;
    }
}

/* ── Q4_0 (legacy) ────────────────────────────────────────────────────────── */
/*
 * Block layout (18 bytes per 32 elements):
 *   [d: fp16 (2 bytes)] [qs: 4-bit×32 = 16 bytes]
 * Decode: out[i] = (nibble[i] - 8) * fp16_to_f32(d)   (zero-point = 8)
 */
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

void gguf_dequant_q4_0(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q4_0_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q4_0_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float scale = fp16_to_f32(d_bits);
        const uint8_t *qs = blk + 2;
        float *dst = out + b * Q4_0_BLOCK_SIZE;
        for (int i = 0; i < Q4_0_BLOCK_SIZE / 2; i++) {
            dst[i]                        = ((int)(qs[i] & 0xF) - 8) * scale;
            dst[i + Q4_0_BLOCK_SIZE / 2] = ((int)(qs[i] >> 4)  - 8) * scale;
        }
    }
}

/* ── Q5_0 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (22 bytes per 32 elements):
 *   [d: fp16 (2)] [qh: u32 (4)] [qs: u8×16 (16)]
 * Decode: out[i] = (low4[i] | ((qh >> i) & 1) << 4 - 16) * d
 * Values are signed (-16..+15 shifted by 16 for unsigned storage).
 */
#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BYTES_PER_BLOCK 22

void gguf_dequant_q5_0(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q5_0_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q5_0_BYTES_PER_BLOCK;
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t *qs = blk + 6;  /* 16 bytes */
        float *dst = out + b * Q5_0_BLOCK_SIZE;
        for (int i = 0; i < 16; i++) {
            int q0 = (int)((qs[i] & 0xF) | (((qh >>  i)      & 1) << 4)) - 16;
            int q1 = (int)((qs[i] >>  4) | (((qh >> (i + 16)) & 1) << 4)) - 16;
            dst[i]      = q0 * d;
            dst[i + 16] = q1 * d;
        }
    }
    size_t done = n_blocks * Q5_0_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q2_0 ─────────────────────────────────────────────────────────────────── */
/*
 * 2-bit block quant used by PrismML's Qwen3.6-based "Bonsai" ternary GGUF
 * releases — specifically their "group-128" packing (34 bytes per 128
 * elements = 2.125 bits/weight), which predates/differs from mainline
 * ggml's own canonical block_q2_0 (group-64, 2.25 bits/weight, shipped
 * separately as "*_g64.gguf"). Both are tagged GGUF type 42; group size
 * isn't recoverable from the type ID and was confirmed empirically against
 * a real downloaded Ternary-Bonsai-27B-Q2_0.gguf (bytes-per-tensor /
 * elements-per-tensor from adjacent tensor offsets = exactly 2.125 bits).
 * Block layout (34 bytes per 128 elements):
 *   [d: fp16 (2)] [qs: u8×32 (32)]  — 2 bits/value, 4 values/byte
 * Decode (same 2-bit code mapping as ggml's dequantize_row_q2_0, just a
 * larger group):
 *   code = (qs[byte_index] >> bit_offset) & 0x3, byte_index=j/4, bit_offset=(j%4)*2
 *   out[j] = (code - 1) * d     // 00=-1, 01=0, 10=+1, 11=+2 (all scaled by d)
 */
#define Q2_0_BLOCK_SIZE 128
#define Q2_0_BYTES_PER_BLOCK 34

void gguf_dequant_q2_0(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q2_0_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q2_0_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        const uint8_t *qs = blk + 2;  /* 16 bytes */
        float *dst = out + b * Q2_0_BLOCK_SIZE;
        for (int j = 0; j < Q2_0_BLOCK_SIZE; j++) {
            int byte_index = j / 4;
            int bit_offset = (j % 4) * 2;
            int code = (qs[byte_index] >> bit_offset) & 0x3;
            dst[j] = (float)(code - 1) * d;
        }
    }
    size_t done = n_blocks * Q2_0_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q5_1 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (24 bytes per 32 elements):
 *   [d: fp16 (2)] [m: fp16 (2)] [qh: u32 (4)] [qs: u8×16 (16)]
 * Decode: out[i] = (low4[i] | ((qh >> i) & 1) << 4) * d + m
 * Low nibble of qs[i] = element i (0..15); high nibble = element i+16.
 * qh bit i = 5th bit for element i; bit i+16 = 5th bit for element i+16.
 */
#define Q5_1_BLOCK_SIZE 32
#define Q5_1_BYTES_PER_BLOCK 24

void gguf_dequant_q5_1(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q5_1_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q5_1_BYTES_PER_BLOCK;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        float d = fp16_to_f32(d_bits);
        float m = fp16_to_f32(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t *qs = blk + 8;  /* 16 bytes */
        float *dst = out + b * Q5_1_BLOCK_SIZE;
        for (int i = 0; i < 16; i++) {
            int q0 = (qs[i] & 0xF) | (((qh >>  i)      & 1) << 4);
            int q1 = (qs[i] >>  4) | (((qh >> (i + 16)) & 1) << 4);
            dst[i]      = q0 * d + m;
            dst[i + 16] = q1 * d + m;
        }
    }
    size_t done = n_blocks * Q5_1_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q4_1 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (20 bytes per 32 elements):
 *   [d: fp16 (2)] [m: fp16 (2)] [qs: u8×16 (16)]
 * Decode: out[i] = nibble[i] * d + m (additive min, no zero-point offset).
 * Low nibble of qs[i] = element i (0..15); high nibble = element i+16.
 * Verified against llama.cpp's block_q4_1/dequantize_row_q4_1 (ggml-common.h/ggml-quants.c).
 */
#define Q4_1_BLOCK_SIZE 32
#define Q4_1_BYTES_PER_BLOCK 20

void gguf_dequant_q4_1(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q4_1_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q4_1_BYTES_PER_BLOCK;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        float d = fp16_to_f32(d_bits);
        float m = fp16_to_f32(m_bits);
        const uint8_t *qs = blk + 4;  /* 16 bytes */
        float *dst = out + b * Q4_1_BLOCK_SIZE;
        for (int i = 0; i < 16; i++) {
            int q0 = qs[i] & 0xF;
            int q1 = qs[i] >>  4;
            dst[i]      = q0 * d + m;
            dst[i + 16] = q1 * d + m;
        }
    }
    size_t done = n_blocks * Q4_1_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q5_K ─────────────────────────────────────────────────────────────────── */
/*
 * Super-block layout (176 bytes per 256 elements):
 *   [d: fp16 (2)] [dmin: fp16 (2)] [scales: 12 bytes] [qh: 32 bytes] [qs: 128 bytes]
 * Same 8-subblock scale/min encoding as Q4_K, plus 1 high bit per element from qh.
 *
 * qh memory layout (matches ggml block_q5_K):
 *   qh[l] (l=0..31) stores the 5th bits for element l across ALL 4 outer groups:
 *     bit 0 of qh[l] = 5th bit of element l        (group 0 low  nibble)
 *     bit 1 of qh[l] = 5th bit of element l+32     (group 0 high nibble)
 *     bit 2 of qh[l] = 5th bit of element l+64     (group 1 low  nibble)
 *     bit 3 of qh[l] = 5th bit of element l+96     (group 1 high nibble)
 *     ...
 *   So for element G: 5th_bit = (qh[G % 32] >> (G / 32)) & 1
 *
 * qs memory layout: 4 groups × 32 bytes, group g uses qs[g*32..(g+1)*32-1]:
 *   low  nibble of qs[g*32+l] → element g*64 + l
 *   high nibble of qs[g*32+l] → element g*64 + l + 32
 *
 * This implementation mirrors the exact loop structure of llama.cpp's
 * dequantize_row_q5_K() in ggml-quants.c for correctness.
 */
#define Q5_K_SUPER  256
#define Q5_K_NSUB   8
#define Q5_K_BYTES  176

void gguf_dequant_q5_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / Q5_K_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * Q5_K_BYTES;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d    = fp16_to_f32(d_bits);
        float dmin = fp16_to_f32(dmin_bits);

        const uint8_t *sc12 = blk + 4;   /* 12 bytes — same format as Q4_K */
        const uint8_t *qh   = blk + 16;  /* 32 bytes — interleaved high bits */
        const uint8_t *ql   = blk + 48;  /* 128 bytes — 4-bit low values     */

        float scales[Q5_K_NSUB], mins[Q5_K_NSUB];
        q4k_decode_scales(sc12, d, dmin, scales, mins);

        float *dst = out + b * Q5_K_SUPER;

        /* Mirror llama.cpp dequantize_row_q5_K exactly:
         * Outer loop: 4 groups of 64 elements (j = 0, 64, 128, 192).
         * u1 mask selects the 5th bit for the low-nibble  32-element half.
         * u2 mask selects the 5th bit for the high-nibble 32-element half.
         * Both masks use qh[l] (l = 0..31), shifting left by 2 each iteration
         * so that successive groups use consecutive bit pairs within each qh byte. */
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < Q5_K_SUPER; j += 64) {
            float d1 = scales[is],     m1 = mins[is];
            float d2 = scales[is + 1], m2 = mins[is + 1];
            for (int l = 0; l < 32; l++) {
                dst[j + l]      = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                dst[j + l + 32] = d2 * ((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
    size_t done = n_super * Q5_K_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ4_NL ───────────────────────────────────────────────────────────────── */
/*
 * Non-linear 4-bit quantization (32 elements, 18 bytes per block):
 *   [d: fp16 (2)] [qs: 4-bit×32 = 16 bytes]
 * Decode: out[i] = d * kvalues_iq4nl[nibble[i]]
 * Uses a fixed 16-entry lookup table of int8 values (same as llama.cpp).
 * Nibble packing: low nibble of qs[i] → element 2i, high nibble → element 2i+1.
 */
#define IQ4_NL_BLOCK_SIZE 32
#define IQ4_NL_BYTES_PER_BLOCK 18

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

void gguf_dequant_iq4_nl(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / IQ4_NL_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * IQ4_NL_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard: bad scale → zero block */
        const uint8_t *qs = blk + 2;
        float *dst = out + b * IQ4_NL_BLOCK_SIZE;
        for (int i = 0; i < IQ4_NL_BLOCK_SIZE / 2; i++) {
            dst[i * 2]     = d * (float)kvalues_iq4nl[qs[i] & 0xF];
            dst[i * 2 + 1] = d * (float)kvalues_iq4nl[qs[i] >> 4];
        }
    }
    size_t done = n_blocks * IQ4_NL_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ4_XS ───────────────────────────────────────────────────────────────── */
/*
 * Non-linear 4-bit k-quant (256 elements, 136 bytes per super-block):
 *   [d: fp16 (2)] [scales_h: u16 (2)] [scales_l: u8×4 (4)] [qs: 4-bit×256 = 128 bytes]
 * Reuses the same kvalues_iq4nl[16] codebook as IQ4_NL, plus a Q4_K-style 6-bit
 * per-32-element scale (4 bits from scales_l, 2 bits from scales_h) biased by -32.
 * Decode: out[i] = d * (scale_ib - 32) * kvalues_iq4nl[nibble[i]], for 8 sub-blocks
 * of 32 elements each. Verified against llama.cpp's dequantize_row_iq4_xs.
 */
#define IQ4_XS_SUPER 256
#define IQ4_XS_BYTES 136

void gguf_dequant_iq4_xs(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ4_XS_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ4_XS_BYTES;
        uint16_t d_bits;      memcpy(&d_bits,      blk,     2);
        uint16_t scales_h;    memcpy(&scales_h,    blk + 2, 2);
        const uint8_t *scales_l = blk + 4;      /* 4 bytes */
        const uint8_t *qs_base  = blk + 8;       /* 128 bytes */
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */

        float *dst = out + b * IQ4_XS_SUPER;
        const uint8_t *qs = qs_base;
        for (int ib = 0; ib < IQ4_XS_SUPER / 32; ib++) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; j++) {
                dst[j +  0] = dl * (float)kvalues_iq4nl[qs[j] & 0xF];
                dst[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >>  4];
            }
            dst += 32;
            qs  += 16;
        }
    }
    size_t done = n_super * IQ4_XS_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q3_K ─────────────────────────────────────────────────────────────────── */
/*
 * Super-block layout (110 bytes per 256 elements):
 *   [hmask: 32 bytes] — 1 high bit per element (bit j of byte b = element b*8+j's high bit)
 *   [qs:    64 bytes] — 2-bit low values; 4 per byte via shifts 0,2,4,6
 *   [scales: 12 bytes] — 16 sub-block 6-bit scales packed 2 per byte
 *   [d: fp16 (2 bytes)] — super-block scale
 *
 * Decode mirrors llama.cpp dequantize_row_q3_K (ggml-quants.c):
 *   3-bit value = (qs_low2 | (hmask_bit << 2)); actual = 3bit - 4 → [-4, 3]
 *   out[e] = d * scale[sub] * actual
 *   Inner loop re-uses the same 32 qs bytes across all 4 shifts; q advances by
 *   32 once per outer 128-element group (not inside the shift loop).
 *   hmask bit for element e: byte = hm[e%32], bit-mask = m (1<<j, cycling 1-128
 *   across the 8 j iterations spanning both n-groups).
 */
#define Q3_K_SUPER  256
#define Q3_K_BYTES  110

void gguf_dequant_q3_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / Q3_K_SUPER;

    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk   = p + b * Q3_K_BYTES;
        const uint8_t *hm    = blk;          /* hmask[32] */
        const uint8_t *qs    = blk + 32;     /* qs[64]    */
        const uint8_t *sc    = blk + 96;     /* scales[12]*/
        uint16_t d_bits;
        memcpy(&d_bits, blk + 108, 2);
        float d = fp16_to_f32(d_bits);

        /* Unpack 16 sub-block scales (6-bit each) from 12 packed bytes.
         * Exact bit manipulation from llama.cpp dequantize_row_q3_K(). */
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;  /* scales[0..15], bias -32 */

        float *dst = out + b * Q3_K_SUPER;
        const uint8_t *q = qs;
        uint8_t m = 1;
        int is  = 0;

        for (int n = 0; n < Q3_K_SUPER; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *dst++ = dl * ((int8_t)((q[l +  0] >> shift) & 3) - ((hm[l +  0] & m) ? 0 : 4));

                dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *dst++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));

                shift += 2;
                m     = (uint8_t)(m << 1);
            }
            q += 32;   /* advance past the 32 qs bytes used by this 128-element group */
        }
    }
    size_t done = n_super * Q3_K_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q2_K ─────────────────────────────────────────────────────────────────── */
/*
 * Super-block layout (84 bytes per 256 elements):
 *   [scales: 16 bytes]  — 16 sub-blocks, each byte: low-nibble=scale, high-nibble=min
 *   [qs:     64 bytes]  — 2-bit quants, 4 per byte (bits [1:0] shifted by 0,2,4,6)
 *   [d:    fp16 2 bytes] — super-block scale (multiplies per-sub-block scale)
 *   [dmin: fp16 2 bytes] — super-block min   (multiplies per-sub-block min)
 *
 * Decode mirrors llama.cpp dequantize_row_q2_K():
 *   256 elements = 2 groups of 128.
 *   Each group: 4 iterations (shift=0,2,4,6) × 2 sub-groups of 16 → 128 elems.
 *   Sub-group 0 uses q[0..15], sub-group 1 uses q[16..31] with same shift bits.
 *   out[elem] = d * (scale_nibble) * ((q_byte >> shift) & 3) - dmin * (min_nibble)
 */
#define Q2_K_SUPER  256
#define Q2_K_BYTES  84

void gguf_dequant_q2_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / Q2_K_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk  = p + b * Q2_K_BYTES;
        const uint8_t *sc   = blk;           /* scales[16] */
        const uint8_t *qs_b = blk + 16;      /* qs[64]     */
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk + 80, 2);
        memcpy(&dmin_bits, blk + 82, 2);
        float d    = fp16_to_f32(d_bits);
        float dmin = fp16_to_f32(dmin_bits);
        if (!(d    == d))    d    = 0.0f;  /* NaN guard */
        if (!(dmin == dmin)) dmin = 0.0f;  /* NaN guard */

        float *dst = out + b * Q2_K_SUPER;
        int is = 0;

        for (int n = 0; n < Q2_K_SUPER; n += 128) {
            const uint8_t *q = qs_b + n / 4;  /* 32 bytes per 128-element group */
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d    * (float)(sc[is]   & 0xF);
                float ml = dmin * (float)(sc[is++] >>  4);
                for (int l = 0; l < 16; l++)
                    dst[n + j * 32 + l]      = dl * (float)((q[l]      >> shift) & 3) - ml;

                dl = d    * (float)(sc[is]   & 0xF);
                ml = dmin * (float)(sc[is++] >>  4);
                for (int l = 0; l < 16; l++)
                    dst[n + j * 32 + 16 + l] = dl * (float)((q[l + 16] >> shift) & 3) - ml;

                shift += 2;
            }
        }
    }
    size_t done = n_super * Q2_K_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q6_K ─────────────────────────────────────────────────────────────────── */
/*
 * Super-block layout (210 bytes per 256 elements):
 *   [ql: 128 bytes — low 4 bits]  [qh: 64 bytes — high 2 bits]
 *   [scales: 16 bytes i8]         [d: fp16 (2 bytes)]
 *
 * 6-bit values are signed (bias -32): q6 = raw6 - 32  →  range [-32, 31].
 * Sub-blocks: 16 sub-blocks of 16 elements each.
 * ggml processes in two 128-element halves; within each half:
 *   for l = 0..31:
 *     q1 = ((ql[l+ 0] & 0xF) | ((qh[l]>>0 & 3)<<4)) - 32   element l+0
 *     q2 = ((ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)) - 32   element l+32
 *     q3 = ((ql[l+ 0] >> 4)  | ((qh[l]>>4 & 3)<<4)) - 32   element l+64
 *     q4 = ((ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4)) - 32   element l+96
 */
#define Q6_K_SUPER  256
#define Q6_K_BYTES  210

void gguf_dequant_q6_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / Q6_K_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t   *blk    = p + b * Q6_K_BYTES;
        const uint8_t   *ql_base = blk;           /* 128 bytes */
        const uint8_t   *qh_base = blk + 128;     /* 64 bytes  */
        const int8_t    *sc_base = (const int8_t *)(blk + 192); /* 16 bytes */
        uint16_t d_bits; memcpy(&d_bits, blk + 208, 2);
        float d = fp16_to_f32(d_bits);

        float *dst = out + b * Q6_K_SUPER;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = ql_base + half * 64;
            const uint8_t *qh = qh_base + half * 32;
            const int8_t  *sc = sc_base + half * 8;
            float         *yy = dst     + half * 128;

            for (int l = 0; l < 32; l++) {
                int is  = l / 16;
                int8_t q1 = (int8_t)(((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                int8_t q2 = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                int8_t q3 = (int8_t)(((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                int8_t q4 = (int8_t)(((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                yy[l +  0] = d * sc[is + 0] * q1;
                yy[l + 32] = d * sc[is + 2] * q2;
                yy[l + 64] = d * sc[is + 4] * q3;
                yy[l + 96] = d * sc[is + 6] * q4;
            }
        }
    }
    size_t done = n_super * Q6_K_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q8_1 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (36 bytes per 32 elements):
 *   [d: fp16 (2)] [s: fp16 (2)] [qs: int8 × 32 (32)]
 * Decode: out[i] = qs[i] * fp16_to_f32(d). `s` (= d * sum(qs)) is a llama.cpp
 * GEMM dot-product shortcut, unused for plain dequant — same as Q8_0 but with
 * an extra unused fp16 field before qs[]. Verified against llama.cpp's real
 * block_q8_1 (ggml-common.h): d and s are BOTH fp16, not fp32 as an earlier
 * draft of the plan doc assumed — gguf_block_size(Q8_1) in gguf_reader.c had
 * the same fp32 assumption baked in (40 bytes) and has been corrected to 36.
 */
#define Q8_1_BLOCK_SIZE 32
#define Q8_1_BYTES_PER_BLOCK 36   /* 2 (d) + 2 (s) + 32 (int8) */

void gguf_dequant_q8_1(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks   = n_elems / Q8_1_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *blk = p + b * Q8_1_BYTES_PER_BLOCK;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        const int8_t *qs = (const int8_t *)(blk + 4);  /* skip d(2)+s(2) */
        float *dst = out + b * Q8_1_BLOCK_SIZE;
        for (int i = 0; i < Q8_1_BLOCK_SIZE; i++)
            dst[i] = qs[i] * d;
    }
    size_t done = n_blocks * Q8_1_BLOCK_SIZE;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── Q8_K ─────────────────────────────────────────────────────────────────── */
/*
 * Super-block layout (292 bytes per 256 elements):
 *   [d: fp32 (4)] [qs: int8 × 256 (256)] [bsums: int16 × 16 (32)]
 * Decode: out[i] = qs[i] * d. `bsums` (per-16 sub-block sums) is a GEMM
 * dot-product shortcut, unused for plain dequant. Unlike every other format
 * in this file, `d` here is a genuine fp32 (matches llama.cpp block_q8_K) —
 * no fp16_to_f32 conversion needed.
 */
#define Q8_K_SUPER 256
#define Q8_K_BYTES 292

void gguf_dequant_q8_k(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / Q8_K_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * Q8_K_BYTES;
        float d; memcpy(&d, blk, 4);
        const int8_t *qs = (const int8_t *)(blk + 4);
        float *dst = out + b * Q8_K_SUPER;
        for (int i = 0; i < Q8_K_SUPER; i++)
            dst[i] = qs[i] * d;
    }
    size_t done = n_super * Q8_K_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ2_XXS codebook tables ─────────────────────────────────────────────────
 * Verbatim from llama.cpp's ggml-common.h (fetched from upstream 2026-09-17,
 * ggerganov/llama.cpp master) -- copied programmatically (not hand-transcribed)
 * to avoid transcription errors in ~2000+ hex entries across the IQ-family
 * codebooks used by this and the following IQ2/IQ3/IQ1 dequant functions.
 */
static const uint8_t kmask_iq2xs[8] = {
1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};


/* ── IQ2_XXS ──────────────────────────────────────────────────────────────── */
/*
 * "True" 2-bit quantization (256 elements, 66 bytes per super-block):
 *   [d: fp16 (2)] [qs: u16×32 (64 bytes), reinterpreted as 8 groups of 2×u32]
 * Per 32-element sub-block (8 of them): read 2 uint32 words from qs; the top
 * 4 bits of the second word are an extra scale nibble, the low 28 bits are
 * 4×7-bit sign patterns (one per 8-element grid vector); the low byte of
 * each of the first word's 4 bytes indexes the 256-entry codebook grid.
 * Verified against llama.cpp's dequantize_row_iq2_xxs.
 */
#define IQ2_XXS_SUPER 256
#define IQ2_XXS_BYTES 66

void gguf_dequant_iq2_xxs(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ2_XXS_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ2_XXS_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint8_t *qs = blk + 2;  /* 64 bytes = 16 groups of 4 bytes = 8 * 2 u32 */

        float *dst = out + b * IQ2_XXS_SUPER;
        for (int ib32 = 0; ib32 < IQ2_XXS_SUPER / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 8 * ib32, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    dst[j] = db * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                dst += 8;
            }
        }
    }
    size_t done = n_super * IQ2_XXS_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ2_XS / IQ2_S codebook tables ──────────────────────────────────────────
 * Verbatim from llama.cpp's ggml-common.h, fetched from upstream 2026-09-17.
 * ksigns_iq2xs and kmask_iq2xs (shared with IQ2_XXS above) are reused for
 * IQ2_XS; IQ2_S encodes its sign bits directly in its qs[] field (no ksigns
 * lookup) but still uses kmask_iq2xs for the per-bit mask.
 */
static const uint64_t iq2xs_grid[512] = {
0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

static const uint64_t iq2s_grid[1024] = {
0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};


/* ── IQ2_XS ───────────────────────────────────────────────────────────────── */
/*
 * 2.3125 bpw (256 elements, 74 bytes per super-block):
 *   [d: fp16 (2)] [qs: u16×32 (64 bytes)] [scales: u8×8 (8 bytes)]
 * Per 32-element sub-block (8 of them, indexed by ib32): scales[ib32]'s two
 * nibbles give two 4-bit sub-scale values (for l<2 and l>=2 respectively).
 * Each of the 4 groups of 8 elements (l=0..3) reads one uint16_t: low 9 bits
 * index the 512-entry grid, high 7 bits index ksigns_iq2xs (shared table with
 * IQ2_XXS above). Verified against llama.cpp's dequantize_row_iq2_xs.
 */
#define IQ2_XS_SUPER 256
#define IQ2_XS_BYTES 74

void gguf_dequant_iq2_xs(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ2_XS_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ2_XS_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint16_t *qs = (const uint16_t *)(blk + 2);      /* 32 u16 = 64 bytes */
        const uint8_t  *scales = blk + 2 + 64;                  /* 8 bytes */

        float *dst = out + b * IQ2_XS_SUPER;
        for (int ib32 = 0; ib32 < IQ2_XS_SUPER / 32; ib32++) {
            float db[2];
            db[0] = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            db[1] = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                uint16_t word = qs[4 * ib32 + l];
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (word & 511));
                uint8_t signs = ksigns_iq2xs[word >> 9];
                float dl = db[l / 2];
                for (int j = 0; j < 8; j++)
                    dst[j] = dl * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                dst += 8;
            }
        }
    }
    size_t done = n_super * IQ2_XS_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ2_S ────────────────────────────────────────────────────────────────── */
/*
 * 2.5625 bpw (256 elements, 82 bytes per super-block):
 *   [d: fp16 (2)] [qs: u8×64 (64 bytes)] [qh: u8×8 (8)] [scales: u8×8 (8)]
 * The 64-byte qs field is usage-split: first 32 bytes are one grid-index byte
 * per (ib32,l), the second 32 bytes are one raw sign-mask byte per (ib32,l)
 * (not looked up via ksigns_iq2xs -- IQ2_S stores signs directly). qh[ib32]
 * supplies the missing 2 high bits (bits 8-9) of each 10-bit grid index into
 * the 1024-entry iq2s_grid. Verified against llama.cpp's dequantize_row_iq2_s.
 */
#define IQ2_S_SUPER 256
#define IQ2_S_BYTES 82

void gguf_dequant_iq2_s(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ2_S_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ2_S_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint8_t *qs_base = blk + 2;        /* 64 bytes: idx[32] + signs[32] */
        const uint8_t *qh      = blk + 2 + 64;    /* 8 bytes */
        const uint8_t *scales  = blk + 2 + 64 + 8; /* 8 bytes */

        float *dst = out + b * IQ2_S_SUPER;
        const uint8_t *qs    = qs_base;
        const uint8_t *signs = qs_base + IQ2_S_SUPER / 8;  /* offset 32 */
        for (int ib32 = 0; ib32 < IQ2_S_SUPER / 32; ib32++) {
            float db[2];
            db[0] = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            db[1] = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                float dl = db[l / 2];
                unsigned idx = (unsigned)qs[l] | (((unsigned)qh[ib32] << (8 - 2 * l)) & 0x300u);
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + idx);
                for (int j = 0; j < 8; j++)
                    dst[j] = dl * (float)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                dst += 8;
            }
            qs    += 4;
            signs += 4;
        }
    }
    size_t done = n_super * IQ2_S_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ3_XXS / IQ3_S codebook tables ─────────────────────────────────────────
 * Verbatim from llama.cpp's ggml-common.h, fetched from upstream 2026-09-17.
 * Entries are uint32_t (4 packed int8 values each), unlike the uint64_t IQ2
 * grids above.
 */
static const uint32_t iq3xxs_grid[256] = {
0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const uint32_t iq3s_grid[512] = {
0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};


/* ── IQ3_XXS ──────────────────────────────────────────────────────────────── */
/*
 * "True" 3-bit quantization (256 elements, 98 bytes per super-block):
 *   [d: fp16 (2)] [qs: u8×96 (96 bytes)]
 * The 96-byte qs field is usage-split: first 64 bytes are grid-index bytes
 * (2 per l, 4 l's per ib32, 8 ib32's = 64), last 32 bytes ("scales_and_signs")
 * pack a uint32 per ib32: top 4 bits = extra scale nibble, low 28 bits = four
 * 7-bit ksigns_iq2xs indices. Each grid entry is 4 bytes (uint32_t), so two
 * grid lookups (grid1, grid2) combine to produce 8 output elements per l.
 * Verified against llama.cpp's dequantize_row_iq3_xxs.
 */
#define IQ3_XXS_SUPER 256
#define IQ3_XXS_BYTES 98

void gguf_dequant_iq3_xxs(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ3_XXS_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ3_XXS_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint8_t *qs_base = blk + 2;                    /* 96 bytes */
        const uint8_t *scales_and_signs = qs_base + IQ3_XXS_SUPER / 4; /* offset 64 */

        float *dst = out + b * IQ3_XXS_SUPER;
        const uint8_t *qs = qs_base;
        for (int ib32 = 0; ib32 < IQ3_XXS_SUPER / 32; ib32++) {
            uint32_t aux32; memcpy(&aux32, scales_and_signs + 4 * ib32, 4);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    dst[j + 0] = db * (float)grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    dst[j + 4] = db * (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                dst += 8;
            }
            qs += 8;
        }
    }
    size_t done = n_super * IQ3_XXS_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ3_S ────────────────────────────────────────────────────────────────── */
/*
 * 3.4375 bpw (256 elements, 110 bytes per super-block):
 *   [d: fp16 (2)] [qs: u8×64 (64)] [qh: u8×8 (8)] [signs: u8×32 (32)]
 *   [scales: u8×4 (4)]
 * Processes 2 ib32 sub-blocks per outer iteration (scales[ib32/2]'s two
 * nibbles give db1/db2 = d*(1+2*nibble)). Each grid entry is 4 bytes;
 * qh supplies the missing high bit (bit 8) of each 9-bit grid index into the
 * 512-entry iq3s_grid. Verified against llama.cpp's dequantize_row_iq3_s.
 */
#define IQ3_S_SUPER 256
#define IQ3_S_BYTES 110

void gguf_dequant_iq3_s(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ3_S_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ3_S_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint8_t *qs_base = blk + 2;              /* 64 bytes */
        const uint8_t *qh_base = blk + 2 + 64;          /* 8 bytes */
        const uint8_t *signs_base = blk + 2 + 64 + 8;    /* 32 bytes */
        const uint8_t *scales = blk + 2 + 64 + 8 + 32;   /* 4 bytes */

        float *dst = out + b * IQ3_S_SUPER;
        const uint8_t *qs = qs_base, *qh = qh_base, *signs = signs_base;
        for (int ib32 = 0; ib32 < IQ3_S_SUPER / 32; ib32 += 2) {
            float db1 = d * (float)(1 + 2 * (scales[ib32 / 2] & 0xF));
            float db2 = d * (float)(1 + 2 * (scales[ib32 / 2] >>  4));

            for (int l = 0; l < 4; l++) {
                unsigned idx1 = (unsigned)qs[2 * l + 0] | (((unsigned)qh[0] << (8 - 2 * l)) & 256u);
                unsigned idx2 = (unsigned)qs[2 * l + 1] | (((unsigned)qh[0] << (7 - 2 * l)) & 256u);
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + idx1);
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + idx2);
                for (int j = 0; j < 4; j++) {
                    dst[j + 0] = db1 * (float)grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    dst[j + 4] = db1 * (float)grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                dst += 8;
            }
            qs += 8; signs += 4;

            for (int l = 0; l < 4; l++) {
                unsigned idx1 = (unsigned)qs[2 * l + 0] | (((unsigned)qh[1] << (8 - 2 * l)) & 256u);
                unsigned idx2 = (unsigned)qs[2 * l + 1] | (((unsigned)qh[1] << (7 - 2 * l)) & 256u);
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + idx1);
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + idx2);
                for (int j = 0; j < 4; j++) {
                    dst[j + 0] = db2 * (float)grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    dst[j + 4] = db2 * (float)grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                dst += 8;
            }
            qh += 2; qs += 8; signs += 4;
        }
    }
    size_t done = n_super * IQ3_S_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}
