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
