/*
 * test_batched_matmul.c — Stage 1 of Phase 18 (speculative decoding):
 * correctness of the new batched-matmul kernels (include/math/batched_matmul.h)
 * against this project's existing single-vector reference kernels.
 *
 * Idiom mirrors tests/test_threading.c's test_parallel_matmul_matches_scalar():
 * compute with the reference path (single-vector, called once per token),
 * compute with the path under test (batched, one call for all tokens),
 * assert element-wise closeness for every (token, output-row) pair.
 */
#include "test_harness.h"
#include "math/batched_matmul.h"
#include "math/parallel_matmul.h"
#include "math/matmul_f16.h"
#include "math/ternary_matmul_packed.h"
#include "core/unpack.h"
#include "core/weights.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOL 1e-4f

static void fill_ternary(tn_i8 *w, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        int r = (seed >> 16) % 3;
        w[i] = (tn_i8)(r - 1);
    }
}

static void fill_floats(float *x, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        x[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
    }
}

/* Random RAW u16 bit patterns can land on F16/BF16's NaN/Inf encodings
 * (all-ones exponent field) -- comparing NaN via `diff > tol` is always
 * false in C, which would silently let a real kernel mismatch pass. These
 * fillers constrain the exponent field away from 0 and all-ones so every
 * generated value is a finite, normal float, with sign/mantissa still
 * randomized. */
static void fill_f16_safe(tn_u16 *w, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned r = seed >> 16;
        unsigned sign = (r >> 15) & 1u;
        unsigned exp  = 10u + (r % 10u);   /* [10,19]: away from 0 and 31 */
        unsigned mant = (r >> 5) & 0x3FFu; /* 10 bits */
        w[i] = (tn_u16)((sign << 15) | (exp << 10) | mant);
    }
}

static void fill_bf16_safe(tn_u16 *w, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        unsigned r = seed >> 16;
        unsigned sign = (r >> 15) & 1u;
        unsigned exp  = 120u + (r % 16u);  /* [120,135]: away from 0 and 255 */
        unsigned mant = (r >> 7) & 0x7Fu;  /* 7 bits */
        w[i] = (tn_u16)((sign << 15) | (exp << 7) | mant);
    }
}

/* ── Ternary-packed batched vs. N single-vector reference calls ─────────── */
static void test_ternary_batch_matches_single(void) {
    int n = 96, d = 20, n_tokens = 5;
    float scale = 1.7f;

    tn_i8 weights[96 * 20];
    fill_ternary(weights, n * d, 111);
    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc((size_t)d * row_bytes, 1);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < n; j++)
            pack_ternary(&packed[(size_t)i * row_bytes], j, weights[i * n + j]);

    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    fill_floats(x, n_tokens * n, 222);

    float *out_batch = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    tn_ternary_matmul_packed_batch(out_batch, x, packed, n, d, scale, n_tokens, NULL);

    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        ternary_matmul_packed(out_ref, x + (size_t)k * n, packed, n, d, &scale, 0);
        for (int i = 0; i < d; i++) {
            if (fabsf(out_ref[i] - out_batch[(size_t)k * d + i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "ternary batched matmul matches N single-vector calls");

    free(packed); free(x); free(out_batch); free(out_ref);
}

/* ── F16 batched vs. N single-vector reference calls ─────────────────────── */
static void test_f16_batch_matches_single(void) {
    int n = 64, d = 12, n_tokens = 4;

    tn_u16 *w = (tn_u16 *)malloc((size_t)n * d * sizeof(tn_u16));
    fill_f16_safe(w, n * d, 333);

    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    fill_floats(x, n_tokens * n, 444);

    float *out_batch = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    tn_matmul_f16_batch(out_batch, x, w, n, d, n_tokens, NULL);

    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        parallel_matmul_f16(out_ref, x + (size_t)k * n, w, n, d, NULL);
        for (int i = 0; i < d; i++) {
            float diff = fabsf(out_ref[i] - out_batch[(size_t)k * d + i]);
            float mag = fabsf(out_ref[i]) + 1.0f;
            if (diff > TOL * mag) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "F16 batched matmul matches N single-vector calls");

    free(w); free(x); free(out_batch); free(out_ref);
}

/* ── BF16 batched vs. N single-vector reference calls ─────────────────────── */
static void test_bf16_batch_matches_single(void) {
    int n = 48, d = 10, n_tokens = 3;

    tn_u16 *w = (tn_u16 *)malloc((size_t)n * d * sizeof(tn_u16));
    fill_bf16_safe(w, n * d, 555);

    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    fill_floats(x, n_tokens * n, 666);

    float *out_batch = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    tn_matmul_bf16_batch(out_batch, x, w, n, d, n_tokens, NULL);

    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        parallel_matmul_bf16(out_ref, x + (size_t)k * n, w, n, d, NULL);
        for (int i = 0; i < d; i++) {
            float diff = fabsf(out_ref[i] - out_batch[(size_t)k * d + i]);
            float mag = fabsf(out_ref[i]) + 1.0f;
            if (diff > TOL * mag) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "BF16 batched matmul matches N single-vector calls");

    free(w); free(x); free(out_batch); free(out_ref);
}

/* ── INT8 (block-quantized) batched vs. N single-vector reference calls ──── */
static void test_i8_batch_matches_single(void) {
    int n = 80, d = 6, n_tokens = 4; /* n spans >1 TN_CLS_QUANT_BLOCK (32) */
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;

    tn_u8 *w = (tn_u8 *)malloc((size_t)n * d);
    for (int i = 0; i < n * d; i++) w[i] = (tn_u8)((i * 37 + 5) & 0xFF);

    float *scales = (float *)malloc((size_t)d * n_blocks * sizeof(float));
    for (int i = 0; i < d * n_blocks; i++) scales[i] = 0.01f + 0.001f * (float)i;

    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    fill_floats(x, n_tokens * n, 777);

    float *out_batch = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    tn_matmul_i8_batch(out_batch, x, w, scales, n, d, n_tokens, NULL);

    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        parallel_matmul_i8(out_ref, x + (size_t)k * n, w, scales, n, d, NULL);
        for (int i = 0; i < d; i++) {
            if (fabsf(out_ref[i] - out_batch[(size_t)k * d + i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "INT8 batched matmul matches N single-vector calls");

    free(w); free(scales); free(x); free(out_batch); free(out_ref);
}

/* ── INT4 (packed, block-quantized) batched vs. N single-vector calls ────── */
static void test_i4_batch_matches_single(void) {
    int n = 80, d = 6, n_tokens = 4;
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    size_t row_bytes = ((size_t)n + 1) / 2;

    tn_u8 *w = (tn_u8 *)malloc((size_t)d * row_bytes);
    for (size_t i = 0; i < (size_t)d * row_bytes; i++) w[i] = (tn_u8)((i * 53 + 3) & 0xFF);

    float *scales = (float *)malloc((size_t)d * n_blocks * sizeof(float));
    for (int i = 0; i < d * n_blocks; i++) scales[i] = 0.02f + 0.002f * (float)i;

    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    fill_floats(x, n_tokens * n, 888);

    float *out_batch = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    tn_matmul_i4_batch(out_batch, x, w, scales, n, d, n_tokens, NULL);

    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        parallel_matmul_i4(out_ref, x + (size_t)k * n, w, scales, n, d, NULL);
        for (int i = 0; i < d; i++) {
            if (fabsf(out_ref[i] - out_batch[(size_t)k * d + i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "INT4 batched matmul matches N single-vector calls");

    free(w); free(scales); free(x); free(out_batch); free(out_ref);
}

/* ── n_tokens == 1 degenerates cleanly (sanity edge case) ─────────────────── */
static void test_batch_single_token_edge_case(void) {
    int n = 32, d = 5;
    float scale = 1.0f;

    tn_i8 weights[32 * 5];
    fill_ternary(weights, n * d, 999);
    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc(d * row_bytes, 1);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < n; j++)
            pack_ternary(&packed[i * row_bytes], j, weights[i * n + j]);

    float x[32];
    fill_floats(x, n, 1010);

    float out_batch[5], out_ref[5];
    tn_ternary_matmul_packed_batch(out_batch, x, packed, n, d, scale, 1, NULL);
    ternary_matmul_packed(out_ref, x, packed, n, d, &scale, 0);

    int all_match = 1;
    for (int i = 0; i < d; i++)
        if (fabsf(out_ref[i] - out_batch[i]) > TOL) all_match = 0;
    TEST_ASSERT(all_match, "n_tokens=1 batched call matches the single-vector call");

    free(packed);
}

int main(void) {
    RUN_TEST(test_ternary_batch_matches_single);
    RUN_TEST(test_f16_batch_matches_single);
    RUN_TEST(test_bf16_batch_matches_single);
    RUN_TEST(test_i8_batch_matches_single);
    RUN_TEST(test_i4_batch_matches_single);
    RUN_TEST(test_batch_single_token_edge_case);
    TEST_SUMMARY();
}
