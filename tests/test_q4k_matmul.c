/*
 * test_q4k_matmul.c — Unit test for the Q4_K x Q8K fused dot product
 * (dot_q4k_row_q8k in src/math/matmul_q4k.c), covering the AVX2 path (the
 * only SIMD implementation of this kernel — an AVX-512 VNNI variant was
 * attempted 2026-07-31 while bringing Q4_K throughput on Qwen3-8B-Q4_K_M
 * closer to llama.cpp's, measured 33-40% SLOWER, and reverted; see
 * docs/ai/mistakes.md for the root cause).
 *
 * There was previously zero test coverage for this kernel at all, despite
 * it being the sole matmul for the generic dense attention/FFN path on any
 * Q4_K-quantized GGUF model (a very common real-world format).
 *
 * Reference: a self-contained scalar re-implementation of the documented
 * formula (see matmul_q4k.c's header comment), independent of which SIMD
 * path parallel_matmul_q4k() actually dispatches to on this build — unlike
 * Q2_0 (where AVX2-vs-VNNI compute genuinely different things, see
 * test_q2_0_matmul.c), both Q4_K paths consume the *same* pre-quantized
 * Q8K activation blocks, so one reference formula is correct for either.
 */

#include "math/matmul_q4k.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q4K_SUPER 256
#define Q4K_NSUB  8
#define Q4K_BYTES 144

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* Pack 8 6-bit scales + 8 6-bit mins into the 12-byte sc12 field, inverse
 * of gguf_quant.c's q4k_decode_scales() / get_scale_min_k4(). */
static void pack_scales(uint8_t *sc12, const uint8_t *sc, const uint8_t *mn) {
    for (int i = 0; i < 4; i++) {
        sc12[i]     = (uint8_t)((sc[i] & 0x3F) | (((sc[i + 4] >> 4) & 0x3) << 6));
        sc12[i + 4] = (uint8_t)((mn[i] & 0x3F) | (((mn[i + 4] >> 4) & 0x3) << 6));
    }
    for (int k = 0; k < 4; k++) {
        sc12[k + 8] = (uint8_t)((sc[k + 4] & 0xF) | ((mn[k + 4] & 0xF) << 4));
    }
}

/* Build one deterministic-but-varied 144-byte Q4_K super-block. */
static void make_q4k_block(uint8_t *blk, int seed) {
    float d    = 0.01f + (float)(seed % 11) * 0.011f;
    float dmin = 0.002f + (float)(seed % 7) * 0.003f;
    uint16_t d_h    = f32_to_fp16(d);
    uint16_t dmin_h = f32_to_fp16(dmin);
    memcpy(blk,     &d_h,    2);
    memcpy(blk + 2, &dmin_h, 2);

    uint8_t sc[Q4K_NSUB], mn[Q4K_NSUB];
    for (int j = 0; j < Q4K_NSUB; j++) {
        sc[j] = (uint8_t)((seed * 7 + j * 5) % 64);
        mn[j] = (uint8_t)((seed * 3 + j * 11) % 64);
    }
    pack_scales(blk + 4, sc, mn);

    uint8_t *qs = blk + 16; /* 128 bytes = 256 nibbles */
    for (int i = 0; i < 128; i++) {
        int lo = (i * 5 + seed) & 0xF;
        int hi = (i * 7 + seed + 3) & 0xF;
        qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

/* Reference dot product: independent scalar re-implementation of the
 * documented Q4_K x Q8K formula (matmul_q4k.c's header comment). */
static float ref_dot_q4k_row(const uint8_t *row, const TnQ8KActBlock *acts, int n_blocks) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    float dot = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + (size_t)b * Q4K_BYTES;
        const TnQ8KActBlock *act = acts + b;
        float d_a = act->d;
        if (d_a == 0.0f) continue;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);
        /* fp16 -> f32 (same conversion matmul_q4k.c uses) */
        uint32_t sign = (uint32_t)(d_bits >> 15) << 31;
        uint32_t exp  = (d_bits >> 10) & 0x1f;
        uint32_t bits = sign | ((exp + 112u) << 23) | ((uint32_t)(d_bits & 0x3ff) << 13);
        float d_w; memcpy(&d_w, &bits, 4);
        sign = (uint32_t)(dmin_bits >> 15) << 31;
        exp  = (dmin_bits >> 10) & 0x1f;
        bits = sign | ((exp + 112u) << 23) | ((uint32_t)(dmin_bits & 0x3ff) << 13);
        float dmin_w; memcpy(&dmin_w, &bits, 4);

        uint32_t utmp[4];
        memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8_t *sc8 = (const uint8_t *)utmp;
        uint8_t raw_sc[8], raw_mn[8];
        for (int j = 0; j < 8; j++) { raw_sc[j] = sc8[j]; raw_mn[j] = sc8[j + 8]; }

        const uint8_t *qs = blk + 16;
        int32_t int_dot = 0, int_min = 0;
        for (int g = 0; g < 4; g++) {
            const uint8_t *q = qs + g * 32;
            const int8_t *al = act->qs + g * 64;
            const int8_t *ah = act->qs + g * 64 + 32;
            int32_t sum_lo = 0, sum_hi = 0;
            for (int l = 0; l < 32; l++) {
                sum_lo += (int32_t)(q[l] & 0x0F) * (int32_t)al[l];
                sum_hi += (int32_t)(q[l] >>    4) * (int32_t)ah[l];
            }
            int_dot += (int32_t)raw_sc[2 * g]     * sum_lo;
            int_dot += (int32_t)raw_sc[2 * g + 1] * sum_hi;
        }
        for (int j = 0; j < 8; j++) {
            int32_t bsum_sb = (int32_t)act->bsums[2 * j] + (int32_t)act->bsums[2 * j + 1];
            int_min += (int32_t)raw_mn[j] * bsum_sb;
        }
        dot += d_a * (d_w * (float)int_dot - dmin_w * (float)int_min);
    }
    return dot;
}

static void make_q8k_block(TnQ8KActBlock *blk, int seed) {
    blk->d = 0.005f + (float)(seed % 9) * 0.0017f;
    for (int i = 0; i < 256; i++)
        blk->qs[i] = (int8_t)(((i * 13 + seed * 37) % 255) - 127);
    for (int j = 0; j < 16; j++) {
        int16_t s = 0;
        for (int k = 0; k < 16; k++) s += blk->qs[j * 16 + k];
        blk->bsums[j] = s;
    }
}

static void run_case(int n_blocks, int d, const char *label) {
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;

    uint8_t *w = (uint8_t *)malloc(row_bytes * (size_t)d);
    TnQ8KActBlock *acts = (TnQ8KActBlock *)malloc((size_t)n_blocks * sizeof(TnQ8KActBlock));
    float *out = (float *)malloc((size_t)d * sizeof(float));
    float *ref = (float *)malloc((size_t)d * sizeof(float));

    for (int b = 0; b < n_blocks; b++)
        make_q8k_block(&acts[b], b + 1);

    for (int r = 0; r < d; r++)
        for (int b = 0; b < n_blocks; b++)
            make_q4k_block(w + (size_t)r * row_bytes + (size_t)b * Q4K_BYTES, r * 31 + b + 1);

    for (int r = 0; r < d; r++)
        ref[r] = ref_dot_q4k_row(w + (size_t)r * row_bytes, acts, n_blocks);

    parallel_matmul_q4k_preq(out, acts, w, n_blocks * Q4K_SUPER, d, NULL);

    for (int r = 0; r < d; r++) {
        float scale = fabsf(ref[r]) > 1.0f ? fabsf(ref[r]) : 1.0f;
        float rel_err = fabsf(out[r] - ref[r]) / scale;
        char msg[160];
        snprintf(msg, sizeof(msg), "%s row %d: kernel=%.6f ref=%.6f rel_err=%.6f",
                 label, r, out[r], ref[r], rel_err);
        TEST_ASSERT(rel_err < 1e-3f, msg);
    }

    free(w); free(acts); free(out); free(ref);
}

static void test_single_block(void)      { run_case(1,  8, "n=256 (1 block)"); }
static void test_few_blocks(void)        { run_case(4, 16, "n=1024 (4 blocks)"); }
static void test_real_attn_dim(void)     { run_case(16, 4, "n=4096 (16 blocks, real attn dim)"); }
static void test_real_ffn_hidden_dim(void) { run_case(48, 4, "n=12288 (48 blocks, real ffn hidden_dim)"); }

int main(void) {
    RUN_TEST(test_single_block);
    RUN_TEST(test_few_blocks);
    RUN_TEST(test_real_attn_dim);
    RUN_TEST(test_real_ffn_hidden_dim);
    TEST_SUMMARY();
}
