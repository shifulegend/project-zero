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
 * Nibble packing is "split halves", NOT interleaved: for j in [0,16), the low
 * nibble of qs[j] is element j and the high nibble is element j+16 (matches
 * ggml's dequantize_row_iq4_nl and this file's own IQ4_XS decode, which
 * already uses this split-half layout for the same kvalues_iq4nl codebook).
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
            dst[i]      = d * (float)kvalues_iq4nl[qs[i] & 0xF];
            dst[i + 16] = d * (float)kvalues_iq4nl[qs[i] >> 4];
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

/* ── IQ1_S / IQ1_M codebook table ────────────────────────────────────────────
 * Verbatim from llama.cpp's ggml-common.h (the CPU/`int8_t`-viewed variant,
 * not the separate `iq1s_grid_gpu` table), fetched from upstream 2026-09-17.
 * Shared by both IQ1_S and IQ1_M. Entries are viewed as int8_t (values in
 * roughly {-1,0,1}), unlike the IQ2/IQ3 grids above which are unsigned.
 */
static const uint64_t iq1s_grid[2048] = {
0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};


/* ── IQ1_S ────────────────────────────────────────────────────────────────── */
/*
 * 1.5625 bpw (256 elements, 50 bytes per super-block):
 *   [d: fp16 (2)] [qs: u8×32 (32)] [qh: u16×8 (16)]
 * Per 32-element sub-block (ib, 8 of them): qh[ib] packs a 3-bit scale
 * (bits 12-14) and a delta-sign bit (bit 15) shared by the whole sub-block,
 * plus 3 extra high bits per l (bits 3l..3l+2) that combine with qs[l] to
 * form an 11-bit index into the 2048-entry iq1s_grid. Each grid entry (8
 * signed bytes, values roughly in {-1,0,1}) gets a per-element `+ delta`
 * offset before scaling -- IQ1S_DELTA=0.125 (llama.cpp ggml-common.h).
 * Verified against llama.cpp's dequantize_row_iq1_s.
 */
#define IQ1S_DELTA 0.125f
#define IQ1_S_SUPER 256
#define IQ1_S_BYTES 50

void gguf_dequant_iq1_s(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ1_S_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ1_S_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = fp16_to_f32(d_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */
        const uint8_t  *qs = blk + 2;        /* 32 bytes */
        const uint8_t  *qh_bytes = blk + 2 + 32; /* 16 bytes -> 8 x u16 */

        float *dst = out + b * IQ1_S_SUPER;
        for (int ib = 0; ib < IQ1_S_SUPER / 32; ib++) {
            uint16_t qh_val; memcpy(&qh_val, qh_bytes + 2 * ib, 2);
            float dl = d * (float)(2 * ((qh_val >> 12) & 7) + 1);
            float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; l++) {
                unsigned idx = (unsigned)qs[l] | (((unsigned)(qh_val >> (3 * l)) & 7u) << 8);
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx);
                for (int j = 0; j < 8; j++)
                    dst[j] = dl * ((float)grid[j] + delta);
                dst += 8;
            }
            qs += 4;
        }
    }
    size_t done = n_super * IQ1_S_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── IQ1_M ────────────────────────────────────────────────────────────────── */
/*
 * 1.75 bpw (256 elements, 56 bytes per super-block, no top-level `d` field):
 *   [qs: u8×32 (32)] [qh: u8×16 (16)] [scales: u8×8 (8), viewed as 4x u16]
 * The super-block scale `d` is itself packed across all 4 scale-u16's top
 * nibbles (a llama.cpp space-saving trick -- see iq1m_scale_t): sc[0]>>12 |
 * (sc[1]>>8 & 0xf0) | (sc[2]>>4 & 0xf00) | (sc[3] & 0xf000), reinterpreted
 * as an fp16. Two sub-blocks (ib, ib+1) share one scales[] u16, each with
 * its own 3-bit dl and IQ1S_DELTA-sized shared grid offset. Shares the same
 * iq1s_grid table as IQ1_S. Verified against llama.cpp's dequantize_row_iq1_m.
 */
#define IQ1_M_SUPER 256
#define IQ1_M_BYTES 56

void gguf_dequant_iq1_m(float *out, const void *data, size_t n_elems) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_super    = n_elems / IQ1_M_SUPER;

    for (size_t b = 0; b < n_super; b++) {
        const uint8_t *blk = p + b * IQ1_M_BYTES;
        const uint8_t *qs = blk;              /* 32 bytes */
        const uint8_t *qh = blk + 32;          /* 16 bytes */
        const uint8_t *scales_bytes = blk + 32 + 16; /* 8 bytes -> 4 x u16 */
        uint16_t sc[4];
        memcpy(sc, scales_bytes, 8);

        uint16_t scale_bits = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0u) |
                                          ((sc[2] >>  4) & 0x0F00u) | (sc[3] & 0xF000u));
        float d = fp16_to_f32(scale_bits);
        if (!(d == d)) d = 0.0f;  /* NaN guard */

        float *dst = out + b * IQ1_M_SUPER;
        /* Flat 8-iteration loop, one per 32-element sub-block -- matches
         * llama.cpp's dequantize_row_iq1_m exactly (each ib reads its own
         * 4 qs bytes + 2 qh bytes; sc[ib/2] and ib%2 select which half of
         * the shared scale u16 pair this ib uses). */
        for (int ib = 0; ib < IQ1_M_SUPER / 32; ib++) {
            float dl1 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            float dl2 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            uint16_t idx[4];
            idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
            idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
            idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
            idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; l++) {
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; j++) dst[j] = dl1 * ((float)grid[j] + delta[l]);
                dst += 8;
            }
            for (int l = 2; l < 4; l++) {
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; j++) dst[j] = dl2 * ((float)grid[j] + delta[l]);
                dst += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
    size_t done = n_super * IQ1_M_SUPER;
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

/* ── BF16 ─────────────────────────────────────────────────────────────────── */
/*
 * Not a block format -- one bf16 (top 16 bits of an IEEE-754 float32) per
 * element, no scale/min. Extracted from gguf_loader.c's tensor_to_f32() for
 * consistency/testability with the rest of this file (Phase 37.16 -- pure
 * refactor, no behavior change).
 */
void gguf_dequant_bf16(float *out, const void *data, size_t n_elems) {
    const uint16_t *s = (const uint16_t *)data;
    for (size_t i = 0; i < n_elems; i++) {
        uint32_t bits = (uint32_t)s[i] << 16;
        memcpy(&out[i], &bits, 4);
    }
}
