/*
 * tests/test_q6k_matmul.c — fused Q6_K kernel: AVX2 dot == scalar dot
 * (exact integer path), both match dequantise-then-dot through the existing
 * gguf_dequant_q6_k(), and batched == per-token.
 */
#include "test_harness.h"
#include "math/matmul_q6k.h"
#include "core/gguf_quant.h"
#include "math/simd_dispatch.h"
#include "threading/thread_pool.h"
#include "sampling/rng.h"
#include <stdint.h>

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000, exp = (u >> 23) & 0xFF, mant = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

static void make_block(uint8_t *blk, unsigned long long *r) {
    for (int i = 0; i < 192; i++) blk[i] = (uint8_t)(rng_float(r) * 256.0f);  /* ql + qh */
    for (int i = 0; i < 16; i++) blk[192 + i] = (uint8_t)(int8_t)((int)(rng_float(r) * 255.0f) - 127);
    uint16_t d = f32_to_fp16(0.001f + rng_float(r) * 0.01f);
    memcpy(blk + 208, &d, 2);
}

static void test_row_dot(void) {
    unsigned long long r; rng_seed(&r, 606);
    for (int nb = 1; nb <= 6; nb++) {
        int n = nb * 256;
        uint8_t *row = (uint8_t *)malloc((size_t)nb * TN_Q6K_BYTES);
        for (int b = 0; b < nb; b++) make_block(row + (size_t)b * TN_Q6K_BYTES, &r);
        float *x = (float *)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = rng_float(&r) * 4.0f - 2.0f;
        TnQ8KActBlock *acts = (TnQ8KActBlock *)malloc((size_t)nb * sizeof(TnQ8KActBlock));
        tn_quantize_q8k(acts, x, nb);

        float simd = tn_dot_q6k_row_q8k(row, acts, nb);
        float scal = tn_dot_q6k_row_q8k_scalar(row, acts, nb);
        TEST_ASSERT(simd == scal, "SIMD dot == scalar dot (same integer math)");

        /* Independent reference: dequantise weights and activations to F32. */
        float *wf = (float *)malloc((size_t)n * sizeof(float));
        gguf_dequant_q6_k(wf, row, (size_t)n);
        double ref = 0;
        for (int b = 0; b < nb; b++)
            for (int i = 0; i < 256; i++)
                ref += (double)wf[b * 256 + i] * (double)acts[b].qs[i] * acts[b].d;
        TEST_ASSERT(fabs(scal - ref) <= 1e-3 * (1.0 + fabs(ref)), "matches dequantise-then-dot");
        free(row); free(x); free(acts); free(wf);
    }
}

static void test_batched_tokens(void) {
    unsigned long long r; rng_seed(&r, 707);
    int nb = 3, n = nb * 256, d = 11, nt = 4;
    uint8_t *w = (uint8_t *)malloc((size_t)d * nb * TN_Q6K_BYTES);
    for (int i = 0; i < d * nb; i++) make_block(w + (size_t)i * TN_Q6K_BYTES, &r);
    float *x = (float *)malloc((size_t)nt * n * sizeof(float));
    for (int i = 0; i < nt * n; i++) x[i] = rng_float(&r) - 0.5f;
    float *seq = (float *)malloc((size_t)nt * d * sizeof(float));
    float *bat = (float *)malloc((size_t)nt * d * sizeof(float));
    ThreadPool *tp = threadpool_create(3);
    for (int t = 0; t < nt; t++) parallel_matmul_q6k(seq + (size_t)t * d, x + (size_t)t * n, w, n, d, tp);
    parallel_matmul_q6k_tokens(bat, x, w, n, d, nt, tp);
    TEST_ASSERT(memcmp(seq, bat, (size_t)nt * d * sizeof(float)) == 0, "batched == per-token");
    threadpool_destroy(tp);
    free(w); free(x); free(seq); free(bat);
}

int main(void) {
    tn_simd_init();
    printf("=== Fused Q6_K matmul Tests ===\n");
    RUN_TEST(test_row_dot);
    RUN_TEST(test_batched_tokens);
    TEST_SUMMARY();
}
