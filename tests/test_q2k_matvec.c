/*
 * test_q2k_matvec.c — Unit test for parallel_matvec_q2k.
 *
 * Builds a synthetic Q2K weight matrix, computes:
 *   (a) reference: gguf_dequant_q2_k → matmul_float32
 *   (b) fused:     parallel_matvec_q2k
 * and checks that both outputs match within floating-point tolerance.
 */

#include "math/matmul_q2k.h"
#include "core/gguf_quant.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Q2K constants (must match matmul_q2k.c) ───────────────────────────────── */
#define Q2K_SUPER 256
#define Q2K_BYTES  84

/* ── fp16 helpers ─────────────────────────────────────────────────────────── */
static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF); /* clamp to max fp16 */
    if (exp <= 102) return (uint16_t) sign;            /* underflow → zero   */
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* ── Reference: dequant + dot product ──────────────────────────────────────── */
static float ref_dot_q2k_row(const uint8_t *row, const float *inp, int in_dim) {
    /* Dequant entire row to float, then dot with inp */
    float *tmp = (float *)malloc((size_t)in_dim * sizeof(float));
    gguf_dequant_q2_k(tmp, row, (size_t)in_dim);
    float s = 0.0f;
    for (int i = 0; i < in_dim; i++) s += tmp[i] * inp[i];
    free(tmp);
    return s;
}

/* ── Build one Q2K super-block deterministically ─────────────────────────── */
static void make_q2k_block(uint8_t *blk, int seed) {
    /* d, dmin: small positive fp16 values */
    float d_f    = 0.02f + (seed & 0xF) * 0.003f;
    float dmin_f = 0.01f + (seed & 0x7) * 0.002f;
    uint16_t d_h    = f32_to_fp16(d_f);
    uint16_t dmin_h = f32_to_fp16(dmin_f);
    memcpy(blk + 80, &d_h,    2);
    memcpy(blk + 82, &dmin_h, 2);

    /* scales[16]: deterministic mix of low/high nibbles */
    for (int i = 0; i < 16; i++)
        blk[i] = (uint8_t)(((i * 3 + seed) & 0xF) | (((i + seed + 1) & 0xF) << 4));

    /* qs[64]: fill with pseudo-random 2-bit values */
    for (int i = 0; i < 64; i++)
        blk[16 + i] = (uint8_t)((i * 7 + seed * 13) & 0xFF);
}

/* ── Tests ─────────────────────────────────────────────────────────────────── */

/* Test 1: single block (in_dim=256, out_dim=1) */
static void test_q2k_single_block(void) {
    uint8_t w[Q2K_BYTES];
    make_q2k_block(w, 42);

    float inp[256];
    for (int i = 0; i < 256; i++) inp[i] = (float)(i % 7) * 0.1f - 0.3f;

    float ref = ref_dot_q2k_row(w, inp, 256);
    float got = 0.0f;
    parallel_matvec_q2k(&got, inp, w, 256, 1, NULL);

    TEST_ASSERT_FLOAT_EQ(got, ref, 1e-4f, "single block: fused matches ref");
}

/* Test 2: 8 blocks per row (in_dim=2048, out_dim=1) — matches actual model dims */
static void test_q2k_one_row_2048(void) {
    uint8_t w[8 * Q2K_BYTES];
    for (int b = 0; b < 8; b++) make_q2k_block(w + b * Q2K_BYTES, b + 1);

    float inp[2048];
    for (int i = 0; i < 2048; i++) inp[i] = sinf((float)i * 0.05f);

    float ref = ref_dot_q2k_row(w, inp, 2048);
    float got = 0.0f;
    parallel_matvec_q2k(&got, inp, w, 2048, 1, NULL);

    TEST_ASSERT_FLOAT_EQ(got, ref, 1e-3f, "2048-wide row: fused matches ref");
}

/* Test 3: 4 output rows (matrix, in_dim=256, out_dim=4) */
static void test_q2k_four_rows(void) {
    const int OUT = 4, IN = 256;
    uint8_t w[4 * Q2K_BYTES];
    for (int r = 0; r < OUT; r++) make_q2k_block(w + r * Q2K_BYTES, r * 5 + 7);

    float inp[256];
    for (int i = 0; i < IN; i++) inp[i] = cosf((float)i * 0.1f) * 0.5f;

    float ref[4], got[4];
    for (int r = 0; r < OUT; r++)
        ref[r] = ref_dot_q2k_row(w + r * Q2K_BYTES, inp, IN);

    parallel_matvec_q2k(got, inp, w, IN, OUT, NULL);

    for (int r = 0; r < OUT; r++) {
        char msg[64]; snprintf(msg, sizeof(msg), "row %d matches ref", r);
        TEST_ASSERT_FLOAT_EQ(got[r], ref[r], 1e-4f, msg);
    }
}

/* Test 4: all-zero input → output must be zero (or -min accumulation check) */
static void test_q2k_zero_input(void) {
    uint8_t w[Q2K_BYTES];
    make_q2k_block(w, 99);

    float inp[256]; memset(inp, 0, sizeof(inp));
    float got = 99.0f;
    parallel_matvec_q2k(&got, inp, w, 256, 1, NULL);
    /* With zero input, sum += dl*0 - ml*0 = 0 for all elements */
    TEST_ASSERT_FLOAT_EQ(got, 0.0f, 1e-7f, "zero input → zero output");
}

/* Test 5: all weights = 0 (zero qs bytes, non-zero scales) */
static void test_q2k_zero_weights(void) {
    uint8_t w[Q2K_BYTES]; memset(w, 0, sizeof(w));
    /* d=1.0, dmin=0.5 but all qs=0 → all q values = 0 → out = 0 - 0.5*inp */
    uint16_t d_h = f32_to_fp16(1.0f), dm_h = f32_to_fp16(0.0f);
    memcpy(w + 80, &d_h,  2);
    memcpy(w + 82, &dm_h, 2);

    float inp[256];
    for (int i = 0; i < 256; i++) inp[i] = (float)(i % 5) * 0.1f;

    float ref = ref_dot_q2k_row(w, inp, 256);
    float got = 0.0f;
    parallel_matvec_q2k(&got, inp, w, 256, 1, NULL);
    TEST_ASSERT_FLOAT_EQ(got, ref, 1e-5f, "zero qs: fused matches ref");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Q2K Matvec Unit Tests ===\n");
    test_q2k_single_block();
    test_q2k_one_row_2048();
    test_q2k_four_rows();
    test_q2k_zero_input();
    test_q2k_zero_weights();
    printf("\nResults: %d/%d passed", tn_tests_passed, tn_tests_run);
    if (tn_tests_failed) printf(", %d FAILED", tn_tests_failed);
    printf("\n");
    return tn_tests_failed ? 1 : 0;
}
