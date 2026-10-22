/*
 * tests/test_ternary_branchless.c — Phase 3.2: branchless scalar ternary
 * matvec agrees with the reference across shapes (incl. non-multiple-of-4
 * widths, all-zero rows and scale).
 */
#include "test_harness.h"
#include "math/ternary_matmul.h"
#include "sampling/rng.h"

static void check_shape(int n, int d, float scale, unsigned long long seed) {
    unsigned long long rng;
    rng_seed(&rng, seed);
    tn_i8 *w = (tn_i8 *)malloc((size_t)n * d);
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *a = (float *)malloc((size_t)d * sizeof(float));
    float *b = (float *)malloc((size_t)d * sizeof(float));
    for (size_t i = 0; i < (size_t)n * d; i++) w[i] = (tn_i8)((int)(rng_float(&rng) * 3.0f) - 1);
    for (int r = 0; r < d && r < 2; r++) memset(w + (size_t)r * n, 0, (size_t)n); /* zero rows */
    for (int i = 0; i < n; i++) x[i] = rng_float(&rng) * 2.0f - 1.0f;
    ternary_matmul(a, x, w, n, d, scale);
    ternary_matmul_branchless(b, x, w, n, d, scale);
    float maxdiff = 0.0f;
    for (int i = 0; i < d; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > maxdiff) maxdiff = e;
    }
    char msg[96];
    snprintf(msg, sizeof(msg), "n=%d d=%d maxdiff=%g", n, d, maxdiff);
    TEST_ASSERT(maxdiff < 1e-3f * (float)n / 64.0f + 1e-4f, msg);
    TEST_ASSERT(b[0] == 0.0f, "zero row -> 0");
    free(w); free(x); free(a); free(b);
}

static void test_shapes(void) {
    check_shape(1, 3, 1.0f, 1);
    check_shape(7, 5, 0.5f, 2);
    check_shape(64, 64, 1.0f, 3);
    check_shape(2051, 17, 0.031f, 4);
}

int main(void) {
    printf("=== Branchless Ternary MatMul (Phase 3.2) Tests ===\n");
    RUN_TEST(test_shapes);
    TEST_SUMMARY();
}
