/*
 * test_q2_0_matmul.c — Unit test for parallel_matmul_q2_0 (and, on an
 * AVX-512 VNNI host, its internal matmul_q2_0_vnni.c fast path).
 *
 * Builds synthetic Q2_0 rows (group-128 packing: 2-byte fp16 scale + 32
 * packed-code bytes per 128 elements — see matmul_q2_0.c's header comment)
 * and checks parallel_matmul_q2_0's output against gguf_dequant_q2_0 (an
 * independent decode path in src/core/gguf_quant.c) + a plain float dot
 * product, across sizes that exercise both the VNNI kernel (n a multiple
 * of 128) and the portable scalar/AVX2 fallback (n NOT a multiple of 128,
 * which matmul_q2_0_vnni.c declines and hands back to matmul_q2_0.c).
 */

#include "math/matmul_q2_0.h"
#include "core/gguf_quant.h"
#include "core/platform.h"
#include "math/quantize_i8.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q2_0_BLOCK 128
#define Q2_0_BYTES 34

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* Fill one 34-byte Q2_0 block deterministically from `seed`: a small
 * positive fp16 scale, and codes cycling through all 3 real ternary values
 * (0,1,2 -> -1,0,+1) plus the unused code 3, so the test exercises every
 * decodable value, not just the ones a real ternary-only model would emit. */
static void make_q2_0_block(uint8_t *blk, int seed) {
    float scale = 0.01f + (float)(seed % 13) * 0.007f;
    uint16_t d_h = f32_to_fp16(scale);
    memcpy(blk, &d_h, 2);
    uint8_t *qs = blk + 2;
    for (int i = 0; i < 32; i++) {
        int c0 = (i * 3 + seed) & 0x3;
        int c1 = (i * 5 + seed + 1) & 0x3;
        int c2 = (i * 7 + seed + 2) & 0x3;
        int c3 = (i * 11 + seed + 3) & 0x3;
        qs[i] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
}

static void make_q2_0_row(uint8_t *row, int n, int row_seed) {
    int n_blocks = n / Q2_0_BLOCK;
    for (int b = 0; b < n_blocks; b++)
        make_q2_0_block(row + (size_t)b * Q2_0_BYTES, row_seed * 31 + b);
}

/*
 * Reference dot product for one row. Which comparison is "correct" depends
 * on which kernel parallel_matmul_q2_0() will actually dispatch to on this
 * build, and the two paths compute genuinely different things:
 *
 *   - AVX-512 VNNI host (matmul_q2_0_vnni.c): the kernel itself quantizes x
 *     to int8 (hardware requirement for dpbusds), so quantize_row_to_i8's
 *     rounding is an intentional, expected approximation of the true float
 *     dot product, not a correctness bug — comparing against the same
 *     *quantized* x isolates whether the VNNI bias-trick/bit-unpack are
 *     themselves correct, independent of that expected int8 rounding.
 *   - Portable/AVX2 host (matmul_q2_0.c, TN_HAS_AVX512VNNI == 0): the kernel
 *     never quantizes x at all — dot_q2_0_row() FMAs the raw float32 x
 *     directly. Comparing that against an int8-quantized reference compares
 *     two different operations and fails on whichever rows this test's
 *     synthetic data happens to round unfavorably for, independent of
 *     whether the kernel is actually correct (root cause of the 2026-07-17
 *     regression: this function was made VNNI-aware but parallel_matmul_q2_0
 *     dispatches to the portable path on every CI runner, none of which have
 *     AVX-512 VNNI, so every run compared the exact-float portable kernel
 *     against a quantized reference it never computes with).
 *
 * So: match the reference to whichever path this build will actually take,
 * same #if this file's own matmul_q2_0.c dispatch uses.
 */
static float ref_dot_q2_0_row(const uint8_t *row, const float *x, int n) {
    float *decoded = (float *)malloc((size_t)n * sizeof(float));
    gguf_dequant_q2_0(decoded, row, (size_t)n);

    float s = 0.0f;

#if TN_HAS_AVX512VNNI
    int8_t *q_x = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    float act_scale = quantize_row_to_i8(x, q_x, n);
    if (act_scale > 0.0f) {
        for (int i = 0; i < n; i++)
            s += decoded[i] * ((float)q_x[i] * act_scale);
    }
    free(q_x);
#else
    for (int i = 0; i < n; i++)
        s += decoded[i] * x[i];
#endif

    free(decoded);
    return s;
}

/* n must be a multiple of Q2_0_BLOCK (128) — every real call site in this
 * model satisfies that; the non-multiple case is tested separately below,
 * exercising matmul_q2_0_vnni.c's decline-and-fall-back path instead. */
static void run_case(int n, int d, const char *label) {
    size_t row_bytes = (size_t)(n / Q2_0_BLOCK) * Q2_0_BYTES;

    uint8_t *w = (uint8_t *)malloc(row_bytes * (size_t)d);
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)d * sizeof(float));
    float *ref = (float *)malloc((size_t)d * sizeof(float));

    for (int i = 0; i < n; i++)
        x[i] = ((float)((i * 17 + 3) % 23) - 11.0f) * 0.37f;

    for (int r = 0; r < d; r++)
        make_q2_0_row(w + (size_t)r * row_bytes, n, r + 1);

    parallel_matmul_q2_0(out, x, w, n, d, NULL);
    for (int r = 0; r < d; r++)
        ref[r] = ref_dot_q2_0_row(w + (size_t)r * row_bytes, x, n);

    for (int r = 0; r < d; r++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: row %d matches reference dequant+dot", label, r);
        /* Both paths sum ~n terms of magnitude ~O(1) in different orders
         * (VNNI's int32 accumulation vs. the reference's serial float sum) —
         * tolerance scales with n to absorb that reordering, not to mask a
         * real mismatch. */
        float tol = 1e-3f * (float)n * 0.02f + 1e-3f;
        TEST_ASSERT_FLOAT_EQ(out[r], ref[r], tol, msg);
    }

    free(w); free(x); free(out); free(ref);
}

static void test_single_block_single_row(void) {
    run_case(128, 1, "single block, single row");
}

static void test_single_block_multi_row(void) {
    run_case(128, 5, "single block, multi row");
}

static void test_multi_block(void) {
    run_case(1280, 3, "10 blocks (n=1280)");
}

static void test_real_model_dims(void) {
    /* Actual dims from the model this kernel was written for
     * (Ternary-Bonsai-27B): dim=5120, ffn hidden=17408. */
    run_case(5120, 2, "real dim=5120");
    run_case(17408, 1, "real ffn hidden=17408");
}

static void test_non_multiple_of_128_falls_back(void) {
    /* n not a multiple of 128 — matmul_q2_0_vnni.c declines (returns 0) and
     * matmul_q2_0.c's portable path handles it instead. Confirms the
     * fallback wiring itself works, not just the VNNI path. run_case()
     * truncates n down to the nearest full block internally to build a
     * valid row, so this and test_single_block_single_row would be
     * identical test bodies — the real target here is exercising
     * parallel_matmul_q2_0 with a non-block-aligned n end-to-end. */
    int n = 150; /* not a multiple of 128 */
    int d = 2;
    int n_blocks = n / Q2_0_BLOCK; /* 1 full block; matmul_q2_0.c's row_bytes
                                      formula (n/128)*34 ignores the rest */
    size_t row_bytes = (size_t)n_blocks * Q2_0_BYTES;

    uint8_t *w = (uint8_t *)malloc(row_bytes * (size_t)d);
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)d * sizeof(float));
    float *ref = (float *)malloc((size_t)d * sizeof(float));

    for (int i = 0; i < n; i++) x[i] = ((float)((i * 13 + 1) % 19) - 9.0f) * 0.29f;
    for (int r = 0; r < d; r++) make_q2_0_row(w + (size_t)r * row_bytes, n_blocks * Q2_0_BLOCK, r + 5);

    parallel_matmul_q2_0(out, x, w, n, d, NULL);
    for (int r = 0; r < d; r++)
        ref[r] = ref_dot_q2_0_row(w + (size_t)r * row_bytes, x, n_blocks * Q2_0_BLOCK);

    for (int r = 0; r < d; r++)
        TEST_ASSERT_FLOAT_EQ(out[r], ref[r], 0.5f, "non-128-multiple n still matches reference");

    free(w); free(x); free(out); free(ref);
}

static void test_all_zero_activation(void) {
    int n = 128, d = 3;
    uint8_t w[Q2_0_BYTES * 3];
    for (int r = 0; r < d; r++) make_q2_0_block(w + (size_t)r * Q2_0_BYTES, r + 9);
    float x[128] = {0};
    float out[3] = {1.0f, 1.0f, 1.0f}; /* poison to confirm it gets overwritten to 0 */

    parallel_matmul_q2_0(out, x, w, n, d, NULL);
    for (int r = 0; r < d; r++)
        TEST_ASSERT_FLOAT_EQ(out[r], 0.0f, 1e-6f, "all-zero activation yields all-zero output");
}

/*
 * parallel_matmul_q2_0_batch (and, on this VNNI host, its internal
 * matmul_q2_0_vnni.c fast path parallel_matmul_q2_0_batch_vnni) has no real
 * call site yet — no Q2_0-MoE model has been wired into the engine, so
 * moe_ffn.c never invokes it. Verified here instead by checking that batching
 * k independent experts through the batch entry point produces bit-for-bit-
 * equivalent results (within FP tolerance) to calling the already-verified
 * single-matrix parallel_matmul_q2_0 once per expert — this isolates "is the
 * batch dispatch/indexing correct" from "is the underlying dot product
 * correct" (the latter is already covered by the tests above).
 */
static void test_batch_matches_single_matrix_path(void) {
    int n = 256, d = 4, k = 3;
    size_t row_bytes = (size_t)(n / Q2_0_BLOCK) * Q2_0_BYTES;

    uint8_t *ws_storage = (uint8_t *)malloc(row_bytes * (size_t)d * (size_t)k);
    float *xs_storage = (float *)malloc((size_t)n * (size_t)k * sizeof(float));
    const uint8_t *ws[3];
    float *xs[3];
    float *outs_batch[3];
    float *outs_single[3];

    for (int e = 0; e < k; e++) {
        uint8_t *w = ws_storage + (size_t)e * row_bytes * (size_t)d;
        ws[e] = w;
        for (int r = 0; r < d; r++)
            make_q2_0_row(w + (size_t)r * row_bytes, n, e * 101 + r + 1);

        float *x = xs_storage + (size_t)e * n;
        xs[e] = x;
        for (int i = 0; i < n; i++)
            x[i] = ((float)((i * (7 + e) + e) % 23) - 11.0f) * 0.31f;

        outs_batch[e] = (float *)malloc((size_t)d * sizeof(float));
        outs_single[e] = (float *)malloc((size_t)d * sizeof(float));
    }

    parallel_matmul_q2_0_batch(outs_batch, xs, ws, n, d, k, NULL);
    for (int e = 0; e < k; e++)
        parallel_matmul_q2_0(outs_single[e], xs[e], ws[e], n, d, NULL);

    for (int e = 0; e < k; e++) {
        for (int r = 0; r < d; r++) {
            char msg[128];
            snprintf(msg, sizeof(msg), "batch expert %d row %d matches single-matrix path", e, r);
            TEST_ASSERT_FLOAT_EQ(outs_batch[e][r], outs_single[e][r], 1e-3f, msg);
        }
        free(outs_batch[e]);
        free(outs_single[e]);
    }
    free(ws_storage);
    free(xs_storage);
}

/* One expert's activation is all-zero, the others are not — exercises the
 * batch VNNI path's per-expert act_scale==0 short-circuit
 * (matmul_q2_0_vnni.c's matmul_q2_0_batch_vnni_task) independent of the
 * other experts' normal rows. */
static void test_batch_mixed_zero_activation(void) {
    int n = 128, d = 2, k = 3;
    size_t row_bytes = (size_t)(n / Q2_0_BLOCK) * Q2_0_BYTES;

    uint8_t *ws_storage = (uint8_t *)malloc(row_bytes * (size_t)d * (size_t)k);
    float *xs_storage = (float *)calloc((size_t)n * (size_t)k, sizeof(float));
    const uint8_t *ws[3];
    float *xs[3];
    float *outs[3];

    for (int e = 0; e < k; e++) {
        uint8_t *w = ws_storage + (size_t)e * row_bytes * (size_t)d;
        ws[e] = w;
        for (int r = 0; r < d; r++)
            make_q2_0_row(w + (size_t)r * row_bytes, n, e * 53 + r + 1);

        float *x = xs_storage + (size_t)e * n;
        xs[e] = x;
        if (e != 1) { /* expert 1 stays all-zero */
            for (int i = 0; i < n; i++)
                x[i] = ((float)((i * 3 + e) % 17) - 8.0f) * 0.41f;
        }

        outs[e] = (float *)malloc((size_t)d * sizeof(float));
        for (int r = 0; r < d; r++) outs[e][r] = 1.0f; /* poison */
    }

    parallel_matmul_q2_0_batch(outs, xs, ws, n, d, k, NULL);

    for (int r = 0; r < d; r++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "batch expert 1 (all-zero activation) row %d is zero", r);
        TEST_ASSERT_FLOAT_EQ(outs[1][r], 0.0f, 1e-6f, msg);
    }
    float ref0[2], ref2[2];
    parallel_matmul_q2_0(ref0, xs[0], ws[0], n, d, NULL);
    parallel_matmul_q2_0(ref2, xs[2], ws[2], n, d, NULL);
    for (int r = 0; r < d; r++) {
        TEST_ASSERT_FLOAT_EQ(outs[0][r], ref0[r], 1e-3f, "batch expert 0 unaffected by expert 1's zero activation");
        TEST_ASSERT_FLOAT_EQ(outs[2][r], ref2[r], 1e-3f, "batch expert 2 unaffected by expert 1's zero activation");
    }

    for (int e = 0; e < k; e++) free(outs[e]);
    free(ws_storage);
    free(xs_storage);
}

int main(void) {
    RUN_TEST(test_single_block_single_row);
    RUN_TEST(test_single_block_multi_row);
    RUN_TEST(test_multi_block);
    RUN_TEST(test_real_model_dims);
    RUN_TEST(test_non_multiple_of_128_falls_back);
    RUN_TEST(test_all_zero_activation);
    RUN_TEST(test_batch_matches_single_matrix_path);
    RUN_TEST(test_batch_mixed_zero_activation);
    TEST_SUMMARY();
}
