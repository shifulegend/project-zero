#include "test_harness.h"
#include "math/ternary_matmul.h"
#include "math/rmsnorm.h"
#include "math/softmax.h"
#include "math/elementwise.h"
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include "core/platform.h"
#include <stdint.h>

/* ---- Ternary MatMul ---- */

static void test_ternary_matmul_basic(void) {
    /* 2x3 weight matrix, input of size 3, output of size 2 */
    /* w = [[1, -1, 0], [0, 1, 1]] */
    tn_i8 w[] = { 1, -1, 0, 0, 1, 1 };
    float x[] = { 2.0f, 3.0f, 4.0f };
    float out[2];
    float scale = 1.0f;

    ternary_matmul(out, x, w, 3, 2, scale);

    /* row 0: 1*2 + (-1)*3 + 0*4 = 2 - 3 = -1 */
    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "matmul row 0");
    /* row 1: 0*2 + 1*3 + 1*4 = 3 + 4 = 7 */
    TEST_ASSERT_FLOAT_EQ(out[1], 7.0f, 1e-5f, "matmul row 1");
}

static void test_ternary_matmul_scale(void) {
    tn_i8 w[] = { 1, 1, 1, 1 };
    float x[] = { 1.0f, 2.0f };
    float out[2];

    ternary_matmul(out, x, w, 2, 2, 0.5f);

    /* row 0: (1+2) * 0.5 = 1.5 */
    TEST_ASSERT_FLOAT_EQ(out[0], 1.5f, 1e-5f, "scaled matmul row 0");
    TEST_ASSERT_FLOAT_EQ(out[1], 1.5f, 1e-5f, "scaled matmul row 1");
}

static void test_ternary_matmul_all_zero(void) {
    tn_i8 w[] = { 0, 0, 0, 0 };
    float x[] = { 999.0f, 888.0f };
    float out[2];

    ternary_matmul(out, x, w, 2, 2, 1.0f);

    TEST_ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-5f, "all-zero weights row 0");
    TEST_ASSERT_FLOAT_EQ(out[1], 0.0f, 1e-5f, "all-zero weights row 1");
}

/* ---- RMSNorm ---- */

static void test_rmsnorm_basic(void) {
    float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float w[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float out[4];

    rmsnorm(out, x, w, 4, 1e-5f);

    /* RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    /* out[i] = x[i] / RMS (approximately) */
    float rms = sqrtf(7.5f + 1e-5f);
    TEST_ASSERT_FLOAT_EQ(out[0], 1.0f / rms, 1e-4f, "rmsnorm element 0");
    TEST_ASSERT_FLOAT_EQ(out[2], 3.0f / rms, 1e-4f, "rmsnorm element 2");
}

/* ---- Softmax ---- */

static void test_softmax_sums_to_one(void) {
    float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    softmax(x, 4);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += x[i];
    TEST_ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "softmax sums to 1");
    TEST_ASSERT(x[3] > x[2], "softmax preserves order");
    TEST_ASSERT(x[2] > x[1], "softmax preserves order 2");
}

static void test_softmax_extreme(void) {
    float x[] = { 1000.0f, 1001.0f, 999.0f };
    softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    TEST_ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "softmax extreme sums to 1");
    TEST_ASSERT(x[1] > x[0], "softmax extreme order");
}

/* ---- Element-wise ---- */

static void test_vec_add(void) {
    float a[] = { 1.0f, 2.0f, 3.0f };
    float b[] = { 4.0f, 5.0f, 6.0f };
    float out[3];

    vec_add(out, a, b, 3);
    TEST_ASSERT_FLOAT_EQ(out[0], 5.0f, 1e-5f, "vec_add 0");
    TEST_ASSERT_FLOAT_EQ(out[2], 9.0f, 1e-5f, "vec_add 2");
}

static void test_silu(void) {
    float x[] = { 0.0f, 1.0f, -1.0f };
    silu(x, 3);

    /* silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689 */
    TEST_ASSERT_FLOAT_EQ(x[0], 0.0f, 1e-5f, "silu(0)");
    TEST_ASSERT_FLOAT_EQ(x[1], 1.0f / (1.0f + expf(-1.0f)), 1e-4f, "silu(1)");
    TEST_ASSERT(x[2] < 0.0f, "silu(-1) is negative");
}

static void test_vec_dot(void) {
    float a[] = { 1.0f, 2.0f, 3.0f };
    float b[] = { 4.0f, 5.0f, 6.0f };

    float d = vec_dot(a, b, 3);
    TEST_ASSERT_FLOAT_EQ(d, 32.0f, 1e-5f, "dot product");
}

/* ---- Aligned Alloc ---- */

static void test_aligned_alloc(void) {
    void *ptr = tn_aligned_alloc(1024, TN_SIMD_ALIGN);
    TEST_ASSERT(ptr != NULL, "aligned alloc returned non-NULL");
    TEST_ASSERT(((uintptr_t)ptr % TN_SIMD_ALIGN) == 0, "pointer is 64-byte aligned");
    tn_aligned_free(ptr);

    /* Odd sizes */
    ptr = tn_aligned_alloc(17, TN_SIMD_ALIGN);
    TEST_ASSERT(ptr != NULL, "aligned alloc odd size");
    TEST_ASSERT(((uintptr_t)ptr % TN_SIMD_ALIGN) == 0, "odd size still aligned");
    tn_aligned_free(ptr);

    /* Calloc zero-fills */
    float *fptr = (float *)tn_aligned_calloc(256, sizeof(float), TN_SIMD_ALIGN);
    TEST_ASSERT(fptr != NULL, "aligned calloc returned non-NULL");
    int all_zero = 1;
    for (int i = 0; i < 256; i++) {
        if (fptr[i] != 0.0f) { all_zero = 0; break; }
    }
    TEST_ASSERT(all_zero, "aligned calloc is zero-filled");
    tn_aligned_free(fptr);

    /* NULL is safe to free */
    tn_aligned_free(NULL);
}

/* ---- RoPE ---- */

static void test_rope_position_zero(void) {
    /* At position 0, cos(0)=1, sin(0)=0, so RoPE should be identity */
    float q[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float k[] = { 5.0f, 6.0f, 7.0f, 8.0f };
    float q_orig[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float k_orig[] = { 5.0f, 6.0f, 7.0f, 8.0f };

    float freq[2];
    rope_precompute_freqs(freq, 4, 10000.0f);
    float no_yarn_corr[2] = {0.0f, 0.0f};
    apply_rope(q, k, freq, 4, 0, 1, 1, 1.0f, 0.0f, 1.0f, no_yarn_corr);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_EQ(q[i], q_orig[i], 1e-5f, "rope q identity at pos 0");
        TEST_ASSERT_FLOAT_EQ(k[i], k_orig[i], 1e-5f, "rope k identity at pos 0");
    }
}

/* ---- Main ---- */

int main(void) {
    RUN_TEST(test_ternary_matmul_basic);
    RUN_TEST(test_ternary_matmul_scale);
    RUN_TEST(test_ternary_matmul_all_zero);
    RUN_TEST(test_rmsnorm_basic);
    RUN_TEST(test_softmax_sums_to_one);
    RUN_TEST(test_softmax_extreme);
    RUN_TEST(test_vec_add);
    RUN_TEST(test_silu);
    RUN_TEST(test_vec_dot);
    RUN_TEST(test_aligned_alloc);
    RUN_TEST(test_rope_position_zero);

    TEST_SUMMARY();
}
