/*
 * test_ternary_lut_avx512bw.c — Phase K-6 LUT ternary kernel correctness
 *
 * Covers lut_mm_avx512bw()/lut_pack_weights() against a plain scalar
 * reference matmul, across shapes that are and aren't multiples of the
 * kernel's 32-column vector width — the tail-column scalar fallback had a
 * digit-decomposition bug that only showed up when N % 32 != 0.
 */

#include "test_harness.h"
#include "core/platform.h"
#include <stdint.h>
#include <stdlib.h>

#if TN_HAS_AVX512

void lut_mm_avx512bw(const int8_t * restrict A, const int8_t * restrict P,
                     int M, int K, int N, int32_t * restrict C);
void lut_pack_weights(const int8_t *W, int K, int N, int8_t *P);

static void scalar_ref_matmul(const int8_t *A, const int8_t *W,
                               int M, int K, int N, int32_t *C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[m * K + k] * (int32_t)W[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

/* Fills A with int8 activations in [-127,127] and W with ternary {-1,0,1}. */
static void fill_case(int8_t *A, int8_t *W, int M, int K, int N,
                      unsigned int seed) {
    for (int i = 0; i < M * K; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (int8_t)(((seed >> 16) % 255) - 127);
    }
    for (int i = 0; i < K * N; i++) {
        seed = seed * 1103515245u + 12345u;
        W[i] = (int8_t)(((seed >> 16) % 3) - 1);
    }
}

static int run_case(int M, int K, int N, unsigned int seed) {
    int8_t *A = malloc((size_t)M * K);
    int8_t *W = malloc((size_t)K * N);
    int8_t *P = malloc((size_t)((K + 4) / 5) * N);
    int32_t *C_lut = malloc((size_t)M * N * sizeof(int32_t));
    int32_t *C_ref = malloc((size_t)M * N * sizeof(int32_t));

    fill_case(A, W, M, K, N, seed);
    lut_pack_weights(W, K, N, P);
    lut_mm_avx512bw(A, P, M, K, N, C_lut);
    scalar_ref_matmul(A, W, M, K, N, C_ref);

    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (C_lut[i] != C_ref[i]) mismatches++;
    }

    free(A); free(W); free(P); free(C_lut); free(C_ref);
    return mismatches;
}

static void test_lut_matches_scalar_full_blocks(void) {
    /* N a multiple of 32: no tail column, exercises only the vector path. */
    TEST_ASSERT_EQ(run_case(4, 40, 64, 1), 0,
                   "lut_mm_avx512bw matches scalar reference (N=64, K=40)");
    TEST_ASSERT_EQ(run_case(2, 500, 128, 2), 0,
                   "lut_mm_avx512bw matches scalar reference (N=128, K=500)");
}

static void test_lut_matches_scalar_tail_columns(void) {
    /* N not a multiple of 32: exercises the scalar tail fallback, which had
     * a digit-decomposition bug (missing +121 base-3 shift before hv/lv
     * divmod) that silently corrupted every tail column. */
    TEST_ASSERT_EQ(run_case(1, 20, 33, 7), 0,
                   "lut_mm_avx512bw matches scalar reference (N=33, tail=1)");
    TEST_ASSERT_EQ(run_case(3, 45, 100, 6), 0,
                   "lut_mm_avx512bw matches scalar reference (N=100, tail=4)");
    TEST_ASSERT_EQ(run_case(1, 5, 32 + 17, 9), 0,
                   "lut_mm_avx512bw matches scalar reference (N=49, tail=17)");
}

static void test_lut_extreme_activation_values(void) {
    /* av can reach 121 (all five weight digits saturated); a[i] can reach
     * the int8 extremes — check both without and with a tail column. */
    int8_t A[5]  = {127, -127, 127, -127, 127};
    int8_t W[5 * 33];
    for (int i = 0; i < 5 * 33; i++) W[i] = 0;
    int8_t w_col[5] = {1, 1, 1, 1, 1};
    for (int k = 0; k < 5; k++) W[k * 33 + 32] = w_col[k];

    int32_t expect = 0;
    for (int k = 0; k < 5; k++) expect += (int32_t)A[k] * (int32_t)w_col[k];

    int8_t P[1 * 33];
    lut_pack_weights(W, 5, 33, P);
    int32_t C[33];
    lut_mm_avx512bw(A, P, 1, 5, 33, C);

    TEST_ASSERT_EQ(C[32], expect,
                   "lut_mm_avx512bw tail column, saturated weights/activations");
}

int main(void) {
    printf("--- Phase K-6 LUT ternary kernel (AVX-512BW) ---\n");
    RUN_TEST(test_lut_matches_scalar_full_blocks);
    RUN_TEST(test_lut_matches_scalar_tail_columns);
    RUN_TEST(test_lut_extreme_activation_values);
    TEST_SUMMARY();
}

#else /* !TN_HAS_AVX512 */

int main(void) {
    printf("(AVX-512BW not available at compile time — test skipped)\n");
    return 0;
}

#endif
