/**
 * gguf_quant.c — GGUF dequantization, legacy block formats (Q8_0, Q4_0, Q5_0, Q2_0, Q5_1, Q4_1,
 * Q8_1) and BF16.
 *
 * Split out of the former single gguf_quant.c (Phase 34.2b); the public entry points are declared
 * in include/core/gguf_quant.h. Block layout references: llama.cpp ggml-quants.h (public domain /
 * MIT).
 */
#include "core/gguf_quant.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "core/fp16.h"
#include "core/gguf_dequant_blocks.h"

/* Every legacy format is a run of fixed-size blocks; a format supplies how one block decodes and
 * gguf_dequant_blocks (include/core/gguf_dequant_blocks.h) walks the whole blocks. */
typedef GgufBlockDecode BlockDecode;

size_t gguf_dequant_blocks(float *out, const void *data, size_t n_elems, size_t block_elems,
                           size_t block_bytes, GgufBlockDecode decode) {
    const uint8_t *p = (const uint8_t *)data;
    size_t n_blocks  = n_elems / block_elems;
    for (size_t b = 0; b < n_blocks; b++) decode(out + b * block_elems, p + b * block_bytes);
    return n_blocks * block_elems;
}

void gguf_dequant_zero_tail(float *out, size_t done, size_t n_elems) {
    for (size_t i = done; i < n_elems; i++) out[i] = 0.0f;
}

#define dequant_blocks gguf_dequant_blocks
#define zero_tail gguf_dequant_zero_tail

/* ── Q8_0 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (34 bytes per 32 elements):
 *   [d: fp16 (2 bytes)] [qs: int8 × 32 (32 bytes)]
 * Decode: out[i] = qs[i] * tn_fp16_to_f32(d)
 */
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BYTES_PER_BLOCK 34 /* 2 (scale) + 32 (int8) */

static void q8_0_block(float *dst, const uint8_t *blk) {
    float scale      = tn_fp16_load(blk);
    const int8_t *qs = (const int8_t *)(blk + 2);
    for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) dst[i] = qs[i] * scale;
}

void gguf_dequant_q8_0(float *out, const void *data, size_t n_elems) {
    size_t done =
        dequant_blocks(out, data, n_elems, Q8_0_BLOCK_SIZE, Q8_0_BYTES_PER_BLOCK, q8_0_block);
    /* Handle partial last block (rare; n_elems usually a multiple of 32) */
    size_t rem = n_elems - done;
    if (rem > 0) {
        const uint8_t *blk =
            (const uint8_t *)data + (done / Q8_0_BLOCK_SIZE) * Q8_0_BYTES_PER_BLOCK;
        float scale      = tn_fp16_load(blk);
        const int8_t *qs = (const int8_t *)(blk + 2);
        for (size_t i = 0; i < rem; i++) out[done + i] = qs[i] * scale;
    }
}

/* ── Q4_0 (legacy) ────────────────────────────────────────────────────────── */
/*
 * Block layout (18 bytes per 32 elements):
 *   [d: fp16 (2 bytes)] [qs: 4-bit×32 = 16 bytes]
 * Decode: out[i] = (nibble[i] - 8) * tn_fp16_to_f32(d)   (zero-point = 8)
 */
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

static void q4_0_block(float *dst, const uint8_t *blk) {
    float scale       = tn_fp16_load(blk);
    const uint8_t *qs = blk + 2;
    for (int i = 0; i < Q4_0_BLOCK_SIZE / 2; i++) {
        dst[i]                       = ((int)(qs[i] & 0xF) - 8) * scale;
        dst[i + Q4_0_BLOCK_SIZE / 2] = ((int)(qs[i] >> 4) - 8) * scale;
    }
}

void gguf_dequant_q4_0(float *out, const void *data, size_t n_elems) {
    dequant_blocks(out, data, n_elems, Q4_0_BLOCK_SIZE, Q4_0_BYTES_PER_BLOCK, q4_0_block);
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

static void q5_0_block(float *dst, const uint8_t *blk) {
    float d = tn_fp16_load(blk);
    uint32_t qh;
    memcpy(&qh, blk + 2, 4);
    const uint8_t *qs = blk + 6; /* 16 bytes */
    for (int i = 0; i < 16; i++) {
        int q0      = (int)((qs[i] & 0xF) | (((qh >> i) & 1) << 4)) - 16;
        int q1      = (int)((qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4)) - 16;
        dst[i]      = q0 * d;
        dst[i + 16] = q1 * d;
    }
}

void gguf_dequant_q5_0(float *out, const void *data, size_t n_elems) {
    zero_tail(out,
              dequant_blocks(out, data, n_elems, Q5_0_BLOCK_SIZE, Q5_0_BYTES_PER_BLOCK, q5_0_block),
              n_elems);
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

static void q2_0_block(float *dst, const uint8_t *blk) {
    float d           = tn_fp16_load(blk);
    const uint8_t *qs = blk + 2; /* 32 bytes */
    for (int j = 0; j < Q2_0_BLOCK_SIZE; j++) {
        int byte_index = j / 4;
        int bit_offset = (j % 4) * 2;
        int code       = (qs[byte_index] >> bit_offset) & 0x3;
        dst[j]         = (float)(code - 1) * d;
    }
}

void gguf_dequant_q2_0(float *out, const void *data, size_t n_elems) {
    zero_tail(out,
              dequant_blocks(out, data, n_elems, Q2_0_BLOCK_SIZE, Q2_0_BYTES_PER_BLOCK, q2_0_block),
              n_elems);
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

static void q5_1_block(float *dst, const uint8_t *blk) {
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, blk, 2);
    memcpy(&m_bits, blk + 2, 2);
    float d = tn_fp16_to_f32(d_bits);
    float m = tn_fp16_to_f32(m_bits);
    uint32_t qh;
    memcpy(&qh, blk + 4, 4);
    const uint8_t *qs = blk + 8; /* 16 bytes */
    for (int i = 0; i < 16; i++) {
        int q0      = (qs[i] & 0xF) | (((qh >> i) & 1) << 4);
        int q1      = (qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
        dst[i]      = q0 * d + m;
        dst[i + 16] = q1 * d + m;
    }
}

void gguf_dequant_q5_1(float *out, const void *data, size_t n_elems) {
    zero_tail(out,
              dequant_blocks(out, data, n_elems, Q5_1_BLOCK_SIZE, Q5_1_BYTES_PER_BLOCK, q5_1_block),
              n_elems);
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

static void q4_1_block(float *dst, const uint8_t *blk) {
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, blk, 2);
    memcpy(&m_bits, blk + 2, 2);
    float d           = tn_fp16_to_f32(d_bits);
    float m           = tn_fp16_to_f32(m_bits);
    const uint8_t *qs = blk + 4; /* 16 bytes */
    for (int i = 0; i < 16; i++) {
        int q0      = qs[i] & 0xF;
        int q1      = qs[i] >> 4;
        dst[i]      = q0 * d + m;
        dst[i + 16] = q1 * d + m;
    }
}

void gguf_dequant_q4_1(float *out, const void *data, size_t n_elems) {
    zero_tail(out,
              dequant_blocks(out, data, n_elems, Q4_1_BLOCK_SIZE, Q4_1_BYTES_PER_BLOCK, q4_1_block),
              n_elems);
}

/* ── Q8_1 ─────────────────────────────────────────────────────────────────── */
/*
 * Block layout (36 bytes per 32 elements):
 *   [d: fp16 (2)] [s: fp16 (2)] [qs: int8 × 32 (32)]
 * Decode: out[i] = qs[i] * tn_fp16_to_f32(d). `s` (= d * sum(qs)) is a llama.cpp
 * GEMM dot-product shortcut, unused for plain dequant — same as Q8_0 but with
 * an extra unused fp16 field before qs[]. Verified against llama.cpp's real
 * block_q8_1 (ggml-common.h): d and s are BOTH fp16, not fp32 as an earlier
 * draft of the plan doc assumed — gguf_block_size(Q8_1) in gguf_reader.c had
 * the same fp32 assumption baked in (40 bytes) and has been corrected to 36.
 */
#define Q8_1_BLOCK_SIZE 32
#define Q8_1_BYTES_PER_BLOCK 36 /* 2 (d) + 2 (s) + 32 (int8) */

static void q8_1_block(float *dst, const uint8_t *blk) {
    float d          = tn_fp16_load(blk);
    const int8_t *qs = (const int8_t *)(blk + 4); /* skip d(2)+s(2) */
    for (int i = 0; i < Q8_1_BLOCK_SIZE; i++) dst[i] = qs[i] * d;
}

void gguf_dequant_q8_1(float *out, const void *data, size_t n_elems) {
    zero_tail(out,
              dequant_blocks(out, data, n_elems, Q8_1_BLOCK_SIZE, Q8_1_BYTES_PER_BLOCK, q8_1_block),
              n_elems);
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
