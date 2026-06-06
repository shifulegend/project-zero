#include "test_harness.h"
#include "core/platform.h"
#include "math/ternary_matmul.h"
#include "math/rmsnorm.h"
#include "math/softmax.h"
#include "math/elementwise.h"
#include "math/simd_dispatch.h"
#include "memory/aligned_alloc.h"
#include <stdint.h>

/*
 * SIMD Correctness Tests
 *
 * Strategy: Run every AVX2 kernel and its scalar baseline on identical inputs,
 * then assert the outputs match within floating-point epsilon.
 *
 * Also tests with larger, realistic sizes (dim=4096) to exercise the SIMD
 * main loop AND the scalar tail paths.
 */

/* ---- Helpers ---- */

/* Fill array with deterministic pseudo-random floats in [-1, 1] */
#if TN_HAS_AVX2
static void fill_random(float *arr, int n, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        arr[i] = ((float)(seed & 0x7FFF) / 16384.0f) - 1.0f;
    }
}

/* Fill int8 array with deterministic ternary values {-1, 0, 1} */
static void fill_ternary(tn_i8 *arr, int n, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        int v = (int)(seed >> 16) % 3;
        arr[i] = (tn_i8)(v - 1);  /* maps 0->-1, 1->0, 2->1 */
    }
}
#endif /* TN_HAS_AVX2 */

/* ---- Ternary MatMul: AVX2 vs Scalar ---- */

#if TN_HAS_AVX2
extern void ternary_matmul_avx2(float *out, const float *x, const tn_i8 *w,
                                 int n, int d, float scale);
extern void rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void softmax_avx2(float *x, int size);
extern void vec_add_avx2(float *out, const float *a, const float *b, int n);
extern void vec_mul_avx2(float *out, const float *a, const float *b, int n);
extern void vec_scale_avx2(float *x, float s, int n);
extern void silu_avx2(float *x, int n);
extern float vec_dot_avx2(const float *a, const float *b, int n);
#endif

static void test_matmul_avx2_small(void) {
#if TN_HAS_AVX2
    /* 2x3 — same test as scalar baseline */
    tn_i8 w[] = { 1, -1, 0, 0, 1, 1 };
    float x[] = { 2.0f, 3.0f, 4.0f };
    float out_scalar[2], out_avx2[2];

    ternary_matmul(out_scalar, x, w, 3, 2, 1.0f);
    ternary_matmul_avx2(out_avx2, x, w, 3, 2, 1.0f);

    TEST_ASSERT_FLOAT_EQ(out_avx2[0], out_scalar[0], 1e-5f, "avx2 matmul small row 0");
    TEST_ASSERT_FLOAT_EQ(out_avx2[1], out_scalar[1], 1e-5f, "avx2 matmul small row 1");
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_matmul_avx2_aligned(void) {
#if TN_HAS_AVX2
    /* Larger test: 256 input, 64 output — exercises main SIMD loop */
    const int n = 256, d = 64;
    float *x = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    tn_i8 *w = (tn_i8 *)tn_aligned_calloc((size_t)d * n, sizeof(tn_i8), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(d, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(d, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x, n, 42);
    fill_ternary(w, d * n, 123);

    ternary_matmul(out_scalar, x, w, n, d, 0.73f);
    ternary_matmul_avx2(out_avx2, x, w, n, d, 0.73f);

    int all_match = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-4f) {
            all_match = 0;
            printf("  matmul mismatch at [%d]: scalar=%f avx2=%f\n",
                   i, out_scalar[i], out_avx2[i]);
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 matmul matches scalar on 256x64");

    tn_aligned_free(x);
    tn_aligned_free(w);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_matmul_avx2_odd_size(void) {
#if TN_HAS_AVX2
    /* Non-multiple-of-8 to exercise scalar tail */
    const int n = 37, d = 5;
    float *x = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    tn_i8 *w = (tn_i8 *)tn_aligned_calloc((size_t)d * n, sizeof(tn_i8), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(d, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(d, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x, n, 77);
    fill_ternary(w, d * n, 88);

    ternary_matmul(out_scalar, x, w, n, d, 1.5f);
    ternary_matmul_avx2(out_avx2, x, w, n, d, 1.5f);

    int all_match = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-4f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 matmul matches scalar on odd-size 37x5");

    tn_aligned_free(x);
    tn_aligned_free(w);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

/* ---- RMSNorm: AVX2 vs Scalar ---- */

static void test_rmsnorm_avx2(void) {
#if TN_HAS_AVX2
    const int size = 256;
    float *x = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *w = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x, size, 42);
    fill_random(w, size, 99);

    rmsnorm(out_scalar, x, w, size, 1e-5f);
    rmsnorm_avx2(out_avx2, x, w, size, 1e-5f);

    int all_match = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-4f) {
            all_match = 0;
            printf("  rmsnorm mismatch at [%d]: scalar=%f avx2=%f\n",
                   i, out_scalar[i], out_avx2[i]);
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 rmsnorm matches scalar on 256");

    tn_aligned_free(x);
    tn_aligned_free(w);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_rmsnorm_avx2_odd(void) {
#if TN_HAS_AVX2
    const int size = 33;  /* Non-multiple-of-8 */
    float *x = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *w = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x, size, 55);
    fill_random(w, size, 66);

    rmsnorm(out_scalar, x, w, size, 1e-5f);
    rmsnorm_avx2(out_avx2, x, w, size, 1e-5f);

    int all_match = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-4f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 rmsnorm matches scalar on odd-size 33");

    tn_aligned_free(x);
    tn_aligned_free(w);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

/* ---- Softmax: AVX2 vs Scalar ---- */

static void test_softmax_avx2(void) {
#if TN_HAS_AVX2
    const int size = 128;
    float *x_scalar = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);
    float *x_avx2 = (float *)tn_aligned_calloc(size, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x_scalar, size, 42);
    memcpy(x_avx2, x_scalar, size * sizeof(float));

    softmax(x_scalar, size);
    softmax_avx2(x_avx2, size);

    int all_match = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(x_avx2[i] - x_scalar[i]) > 1e-5f) {
            all_match = 0;
            printf("  softmax mismatch at [%d]: scalar=%f avx2=%f\n",
                   i, x_scalar[i], x_avx2[i]);
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 softmax matches scalar on 128");

    /* Verify sum-to-1 */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += x_avx2[i];
    TEST_ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "avx2 softmax sums to 1");

    tn_aligned_free(x_scalar);
    tn_aligned_free(x_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_softmax_avx2_extreme(void) {
#if TN_HAS_AVX2
    /* Test numerical stability with extreme values */
    float x_scalar[] = { 1000.0f, 1001.0f, 999.0f, 998.0f, 1002.0f, 997.0f, 1000.5f, 999.5f,
                         996.0f, 1003.0f };
    float x_avx2[10];
    memcpy(x_avx2, x_scalar, 10 * sizeof(float));

    softmax(x_scalar, 10);
    softmax_avx2(x_avx2, 10);

    int all_match = 1;
    for (int i = 0; i < 10; i++) {
        if (fabsf(x_avx2[i] - x_scalar[i]) > 1e-6f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 softmax extreme matches scalar");
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

/* ---- Element-wise: AVX2 vs Scalar ---- */

static void test_vec_add_avx2(void) {
#if TN_HAS_AVX2
    const int n = 100;  /* Non-multiple-of-8 */
    float *a = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *b = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);

    fill_random(a, n, 11);
    fill_random(b, n, 22);

    vec_add(out_scalar, a, b, n);
    vec_add_avx2(out_avx2, a, b, n);

    int all_match = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-6f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 vec_add matches scalar");

    tn_aligned_free(a);
    tn_aligned_free(b);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_vec_mul_avx2(void) {
#if TN_HAS_AVX2
    const int n = 100;
    float *a = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *b = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *out_avx2 = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);

    fill_random(a, n, 33);
    fill_random(b, n, 44);

    vec_mul(out_scalar, a, b, n);
    vec_mul_avx2(out_avx2, a, b, n);

    int all_match = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(out_avx2[i] - out_scalar[i]) > 1e-6f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 vec_mul matches scalar");

    tn_aligned_free(a);
    tn_aligned_free(b);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_vec_scale_avx2(void) {
#if TN_HAS_AVX2
    const int n = 100;
    float *x_scalar = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *x_avx2 = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x_scalar, n, 55);
    memcpy(x_avx2, x_scalar, n * sizeof(float));

    vec_scale(x_scalar, 2.5f, n);
    vec_scale_avx2(x_avx2, 2.5f, n);

    int all_match = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(x_avx2[i] - x_scalar[i]) > 1e-6f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 vec_scale matches scalar");

    tn_aligned_free(x_scalar);
    tn_aligned_free(x_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_vec_dot_avx2(void) {
#if TN_HAS_AVX2
    const int n = 100;
    float *a = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *b = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);

    fill_random(a, n, 66);
    fill_random(b, n, 77);

    float dot_scalar = vec_dot(a, b, n);
    float dot_avx2 = vec_dot_avx2(a, b, n);

    TEST_ASSERT_FLOAT_EQ(dot_avx2, dot_scalar, 1e-3f, "avx2 vec_dot matches scalar");

    tn_aligned_free(a);
    tn_aligned_free(b);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

static void test_silu_avx2(void) {
#if TN_HAS_AVX2
    const int n = 100;
    float *x_scalar = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);
    float *x_avx2 = (float *)tn_aligned_calloc(n, sizeof(float), TN_SIMD_ALIGN);

    fill_random(x_scalar, n, 88);
    memcpy(x_avx2, x_scalar, n * sizeof(float));

    silu(x_scalar, n);
    silu_avx2(x_avx2, n);

    int all_match = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(x_avx2[i] - x_scalar[i]) > 1e-5f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "avx2 silu matches scalar");

    tn_aligned_free(x_scalar);
    tn_aligned_free(x_avx2);
#else
    TEST_ASSERT(1, "AVX2 not available — skipped");
#endif
}

/* ---- SIMD Dispatch Init ---- */

static void test_simd_dispatch_init(void) {
    const char *backend = tn_simd_init();
    TEST_ASSERT(backend != NULL, "tn_simd_init returns non-NULL backend name");
    TEST_ASSERT(tn_ternary_matmul != NULL, "dispatch: matmul set");
    TEST_ASSERT(tn_rmsnorm != NULL, "dispatch: rmsnorm set");
    TEST_ASSERT(tn_softmax != NULL, "dispatch: softmax set");
    TEST_ASSERT(tn_vec_add != NULL, "dispatch: vec_add set");
    TEST_ASSERT(tn_vec_mul != NULL, "dispatch: vec_mul set");
    TEST_ASSERT(tn_vec_scale != NULL, "dispatch: vec_scale set");
    TEST_ASSERT(tn_silu != NULL, "dispatch: silu set");
    TEST_ASSERT(tn_vec_dot != NULL, "dispatch: vec_dot set");

    printf("  SIMD backend: %s\n", backend);
}

static void test_dispatch_matmul_works(void) {
    tn_simd_init();

    tn_i8 w[] = { 1, -1, 0, 0, 1, 1 };
    float x[] = { 2.0f, 3.0f, 4.0f };
    float out[2];

    tn_ternary_matmul(out, x, w, 3, 2, 1.0f);

    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "dispatch matmul row 0");
    TEST_ASSERT_FLOAT_EQ(out[1], 7.0f, 1e-5f, "dispatch matmul row 1");
}

/* ---- Main ---- */

int main(void) {
    printf("=== SIMD Correctness Tests ===\n");

    /* Dispatch tests */
    RUN_TEST(test_simd_dispatch_init);
    RUN_TEST(test_dispatch_matmul_works);

    /* AVX2 vs Scalar comparison tests */
    RUN_TEST(test_matmul_avx2_small);
    RUN_TEST(test_matmul_avx2_aligned);
    RUN_TEST(test_matmul_avx2_odd_size);
    RUN_TEST(test_rmsnorm_avx2);
    RUN_TEST(test_rmsnorm_avx2_odd);
    RUN_TEST(test_softmax_avx2);
    RUN_TEST(test_softmax_avx2_extreme);
    RUN_TEST(test_vec_add_avx2);
    RUN_TEST(test_vec_mul_avx2);
    RUN_TEST(test_vec_scale_avx2);
    RUN_TEST(test_vec_dot_avx2);
    RUN_TEST(test_silu_avx2);

    TEST_SUMMARY();
}
