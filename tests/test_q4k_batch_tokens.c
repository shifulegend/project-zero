/*
 * tests/test_q4k_batch_tokens.c — Phase 18-B: batched Q4_K, Q4_K-x8 and F32
 * matmuls over n_tokens activations are bit-identical to per-token calls.
 */
#include "test_harness.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q4k_x8.h"
#include "math/batched_matmul.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "threading/thread_pool.h"
#include "sampling/rng.h"
#include <stdint.h>

#define Q4K_BYTES 144

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000, exp = (u >> 23) & 0xFF, mant = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* Valid Q4_K block: small positive d/dmin, arbitrary 6-bit scales and nibbles. */
static void make_block(uint8_t *blk, unsigned long long *r) {
    uint16_t d = f32_to_fp16(0.01f + rng_float(r) * 0.05f);
    uint16_t dm = f32_to_fp16(0.002f + rng_float(r) * 0.01f);
    memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
    for (int i = 4; i < Q4K_BYTES; i++) blk[i] = (uint8_t)(rng_float(r) * 256.0f);
}

static void check(int n_blocks, int d, int n_tokens, ThreadPool *tp) {
    unsigned long long r; rng_seed(&r, (unsigned long long)(n_blocks * 977 + d * 31 + n_tokens));
    int n = n_blocks * 256;
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;
    uint8_t *w = (uint8_t *)malloc((size_t)d * row_bytes);
    for (int i = 0; i < d * n_blocks; i++) make_block(w + (size_t)i * Q4K_BYTES, &r);
    float *x = (float *)malloc((size_t)n_tokens * n * sizeof(float));
    for (int i = 0; i < n_tokens * n; i++) x[i] = rng_float(&r) * 2.0f - 1.0f;
    float *seq = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    float *bat = (float *)malloc((size_t)n_tokens * d * sizeof(float));

    for (int t = 0; t < n_tokens; t++)
        parallel_matmul_q4k(seq + (size_t)t * d, x + (size_t)t * n, w, n, d, tp);
    parallel_matmul_q4k_tokens(bat, x, w, n, d, n_tokens, tp);
    TEST_ASSERT(memcmp(seq, bat, (size_t)n_tokens * d * sizeof(float)) == 0,
                "Q4_K batched == per-token (bit-identical)");

    if (d % 8 == 0) {
        TnQ4KX8Block *rp = (TnQ4KX8Block *)malloc((size_t)(d / 8) * n_blocks * sizeof(TnQ4KX8Block));
        for (int g = 0; g < d / 8; g++) {
            const uint8_t *rows[8];
            for (int k = 0; k < 8; k++) rows[k] = w + (size_t)(g * 8 + k) * row_bytes;
            tn_q4k_repack_x8(rp + (size_t)g * n_blocks, rows, n_blocks);
        }
        TnQ8KActBlock *acts = (TnQ8KActBlock *)malloc((size_t)n_tokens * n_blocks * sizeof(TnQ8KActBlock));
        for (int t = 0; t < n_tokens; t++) {
            tn_quantize_q8k(acts + (size_t)t * n_blocks, x + (size_t)t * n, n_blocks);
            parallel_matmul_q4k_x8_preq(seq + (size_t)t * d, acts + (size_t)t * n_blocks, rp, n_blocks, d, tp);
        }
        parallel_matmul_q4k_x8_tokens_preq(bat, acts, rp, n_blocks, d, n_tokens, tp);
        TEST_ASSERT(memcmp(seq, bat, (size_t)n_tokens * d * sizeof(float)) == 0,
                    "Q4_K x8 batched == per-token (bit-identical)");
        free(rp); free(acts);
    }
    free(w); free(x); free(seq); free(bat);
}

static void test_f32_batch(void) {
    ThreadPool *tp = threadpool_create(3);
    unsigned long long r; rng_seed(&r, 4242);
    int n = 300, d = 37, nt = 5;
    float *w = (float *)malloc((size_t)n * d * sizeof(float));
    float *x = (float *)malloc((size_t)nt * n * sizeof(float));
    float *seq = (float *)malloc((size_t)nt * d * sizeof(float));
    float *bat = (float *)malloc((size_t)nt * d * sizeof(float));
    for (int i = 0; i < n * d; i++) w[i] = rng_float(&r) - 0.5f;
    for (int i = 0; i < nt * n; i++) x[i] = rng_float(&r) - 0.5f;
    for (int t = 0; t < nt; t++) parallel_matmul_float32(seq + (size_t)t * d, x + (size_t)t * n, w, n, d, tp);
    tn_matmul_f32_batch(bat, x, w, n, d, nt, tp);
    TEST_ASSERT(memcmp(seq, bat, (size_t)nt * d * sizeof(float)) == 0, "F32 batched == per-token (bit-identical)");
    free(w); free(x); free(seq); free(bat);
    threadpool_destroy(tp);
}

static void test_shapes(void) {
    ThreadPool *tp = threadpool_create(3);
    check(1, 8, 1, NULL);
    check(2, 16, 5, NULL);
    check(3, 24, 4, tp);
    check(4, 13, 6, tp);   /* d not a multiple of 8: plain Q4_K only */
    threadpool_destroy(tp);
}

int main(void) {
    tn_simd_init();
    printf("=== Q4_K batched-over-tokens (Phase 18-B) Tests ===\n");
    RUN_TEST(test_shapes);
    RUN_TEST(test_f32_batch);
    TEST_SUMMARY();
}
