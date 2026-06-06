/**
 * test_packed_weights.c — Phase 10 test suite for 2-bit weight packing.
 *
 * Tests:
 *  1. Pack/unpack round-trip correctness (scalar)
 *  2. AVX2 unpack matches scalar unpack
 *  3. Packed matmul matches unpacked matmul (per-matrix scale)
 *  4. Packed matmul matches unpacked matmul (per-group scale)
 *  5. AVX2 packed matmul matches scalar packed matmul
 *  6. Edge cases: odd dimensions, single element, large matrices
 *  7. SIMD dispatch integration
 */

#include "core/unpack.h"
#include "math/ternary_matmul.h"
#include "math/ternary_matmul_packed.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include "test_harness.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: create deterministic ternary weights {-1, 0, 1} */
static void fill_ternary(tn_i8 *w, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        int r = (seed >> 16) % 3;
        w[i] = (tn_i8)(r - 1);  /* -1, 0, or 1 */
    }
}

/* Helper: pack ternary weights into 2-bit packed format */
static void pack_weights(tn_u8 *packed, const tn_i8 *weights, int count) {
    memset(packed, 0, packed_bytes(count));
    for (int i = 0; i < count; i++) {
        pack_ternary(packed, i, weights[i]);
    }
}

/* Helper: fill float input vector */
static void fill_input(float *x, int n, unsigned seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        x[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
    }
}

/* ================================================================
 * 1. Pack/Unpack Round-Trip (Scalar)
 * ================================================================ */

static void test_pack_unpack_roundtrip(void) {
    tn_i8 original[256];
    fill_ternary(original, 256, 42);

    tn_u8 packed[64]; /* 256 / 4 = 64 bytes */
    pack_weights(packed, original, 256);

    tn_i8 unpacked[256];
    unpack_ternary_block(unpacked, packed, 256);

    for (int i = 0; i < 256; i++) {
        TEST_ASSERT(unpacked[i] == original[i],
                    "round-trip: unpacked matches original");
    }
}

static void test_pack_unpack_single(void) {
    tn_i8 vals[3] = {-1, 0, 1};
    for (int i = 0; i < 3; i++) {
        tn_u8 packed[1] = {0};
        pack_ternary(packed, 0, vals[i]);
        tn_i8 result = unpack_ternary(packed, 0);
        TEST_ASSERT(result == vals[i], "single pack/unpack correct");
    }
}

static void test_pack_unpack_odd_count(void) {
    /* 7 weights -> 2 packed bytes (with padding) */
    tn_i8 original[7] = {1, -1, 0, 1, 0, -1, 1};
    tn_u8 packed[2] = {0, 0};
    pack_weights(packed, original, 7);

    tn_i8 unpacked[7];
    unpack_ternary_block(unpacked, packed, 7);

    for (int i = 0; i < 7; i++) {
        TEST_ASSERT(unpacked[i] == original[i],
                    "odd count: unpacked matches original");
    }
}

static void test_packed_bytes_calc(void) {
    TEST_ASSERT(packed_bytes(0) == 0, "packed_bytes(0) == 0");
    TEST_ASSERT(packed_bytes(1) == 1, "packed_bytes(1) == 1");
    TEST_ASSERT(packed_bytes(4) == 1, "packed_bytes(4) == 1");
    TEST_ASSERT(packed_bytes(5) == 2, "packed_bytes(5) == 2");
    TEST_ASSERT(packed_bytes(8) == 2, "packed_bytes(8) == 2");
    TEST_ASSERT(packed_bytes(256) == 64, "packed_bytes(256) == 64");
}

/* ================================================================
 * 2. AVX2 Unpack Matches Scalar
 * ================================================================ */

static void test_avx2_unpack_matches_scalar(void) {
    tn_simd_init();

    int count = 1024;
    tn_i8 original[1024];
    fill_ternary(original, count, 7777);

    tn_u8 packed[256]; /* 1024 / 4 */
    pack_weights(packed, original, count);

    /* Scalar unpack */
    tn_i8 scalar_out[1024];
    unpack_ternary_block(scalar_out, packed, count);

    /* SIMD unpack (via dispatch) */
    tn_i8 simd_out[1024];
    tn_unpack_block(simd_out, packed, count);

    for (int i = 0; i < count; i++) {
        TEST_ASSERT(simd_out[i] == scalar_out[i],
                    "AVX2 unpack matches scalar");
    }
}

static void test_avx2_unpack_odd_count(void) {
    tn_simd_init();

    int count = 137; /* not a multiple of 64 */
    tn_i8 original[137];
    fill_ternary(original, count, 9999);

    tn_u8 packed[35]; /* ceil(137/4) = 35 */
    pack_weights(packed, original, count);

    tn_i8 scalar_out[137], simd_out[137];
    unpack_ternary_block(scalar_out, packed, count);
    tn_unpack_block(simd_out, packed, count);

    for (int i = 0; i < count; i++) {
        TEST_ASSERT(simd_out[i] == scalar_out[i],
                    "AVX2 unpack odd count matches scalar");
    }
}

/* ================================================================
 * 3. Packed Matmul Matches Unpacked (Per-Matrix Scale)
 * ================================================================ */

static void test_packed_matmul_per_matrix(void) {
    int n = 64, d = 8;
    float scale = 1.5f;

    tn_i8 weights[64 * 8];
    fill_ternary(weights, n * d, 111);

    float x[64];
    fill_input(x, n, 222);

    /* Unpacked matmul (reference) */
    float ref_out[8];
    ternary_matmul(ref_out, x, weights, n, d, scale);

    /* Pack weights and do packed matmul */
    tn_u8 packed[64 * 8 / 4]; /* n*d / 4 */
    size_t row_bytes = packed_bytes(n);
    memset(packed, 0, sizeof(packed));
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            pack_ternary(&packed[i * row_bytes], j, weights[i * n + j]);
        }
    }

    float packed_out[8];
    ternary_matmul_packed(packed_out, x, packed, n, d, &scale, 0);

    for (int i = 0; i < d; i++) {
        float diff = fabsf(ref_out[i] - packed_out[i]);
        TEST_ASSERT(diff < 1e-5f, "packed matmul matches reference (per-matrix)");
    }
}

/* ================================================================
 * 4. Packed Matmul Per-Group Scale
 * ================================================================ */

static void test_packed_matmul_per_group(void) {
    int n = 64, d = 4;
    int group_size = 16;
    int n_groups = (n + group_size - 1) / group_size; /* 4 groups */

    tn_i8 weights[64 * 4];
    fill_ternary(weights, n * d, 333);

    float x[64];
    fill_input(x, n, 444);

    /* Compute per-group reference manually */
    float scales[4 * 4]; /* d * n_groups */
    for (int i = 0; i < d * n_groups; i++) {
        scales[i] = 1.0f + (float)(i % 5) * 0.1f;
    }

    float ref_out[4];
    for (int i = 0; i < d; i++) {
        float total = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            float group_sum = 0.0f;
            int start = g * group_size;
            int end = start + group_size;
            if (end > n) end = n;
            for (int j = start; j < end; j++) {
                tn_i8 w = weights[i * n + j];
                if (w == 1) group_sum += x[j];
                else if (w == -1) group_sum -= x[j];
            }
            total += group_sum * scales[i * n_groups + g];
        }
        ref_out[i] = total;
    }

    /* Pack and compute */
    size_t row_bytes = packed_bytes(n);
    tn_u8 packed[4 * 16]; /* d * row_bytes */
    memset(packed, 0, sizeof(packed));
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            pack_ternary(&packed[i * row_bytes], j, weights[i * n + j]);
        }
    }

    float packed_out[4];
    ternary_matmul_packed(packed_out, x, packed, n, d, scales, group_size);

    for (int i = 0; i < d; i++) {
        float diff = fabsf(ref_out[i] - packed_out[i]);
        TEST_ASSERT(diff < 1e-5f, "packed matmul matches reference (per-group)");
    }
}

/* ================================================================
 * 5. AVX2 Packed Matmul Matches Scalar Packed Matmul
 * ================================================================ */

static void test_avx2_packed_matmul_matches_scalar(void) {
    tn_simd_init();

    int n = 128, d = 16;
    float scale = 2.0f;

    tn_i8 weights[128 * 16];
    fill_ternary(weights, n * d, 555);

    float x[128];
    fill_input(x, n, 666);

    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc(d * row_bytes, 1);

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            pack_ternary(&packed[i * row_bytes], j, weights[i * n + j]);
        }
    }

    /* Scalar reference */
    float scalar_out[16];
    ternary_matmul_packed(scalar_out, x, packed, n, d, &scale, 0);

    /*
     * Test AVX2 kernel directly (not via dispatch).
     *
     * Phase 16-S changed tn_ternary_matmul_packed dispatch to prefer the
     * AVX-512 VNNI kernel over AVX2 when VNNI is available.  The VNNI kernel
     * quantises float activations to int8 internally, introducing ~1%
     * quantisation error relative to the scalar float32 reference — too large
     * for the original 1e-4f tolerance.
     *
     * To keep testing what this test was designed to test (AVX2 float-domain
     * correctness), call ternary_matmul_packed_avx2 directly here.  The VNNI
     * dispatch path is validated separately in test_simd_vnni.c with an
     * int8-appropriate 5% relative tolerance.
     */
    float simd_out[16];
#if TN_HAS_AVX2
    ternary_matmul_packed_avx2(simd_out, x, packed, n, d, &scale, 0);
#else
    ternary_matmul_packed(simd_out, x, packed, n, d, &scale, 0);
#endif

    for (int i = 0; i < d; i++) {
        float diff = fabsf(scalar_out[i] - simd_out[i]);
        TEST_ASSERT(diff < 1e-4f, "AVX2 packed matmul matches scalar");
    }

    free(packed);
}

/* ================================================================
 * 6. Edge Cases
 * ================================================================ */

static void test_single_element(void) {
    tn_i8 w = 1;
    tn_u8 packed[1] = {0};
    pack_ternary(packed, 0, w);

    float x[1] = {3.0f};
    float scale = 2.0f;
    float out[1];
    ternary_matmul_packed(out, x, packed, 1, 1, &scale, 0);

    /* Expected: 3.0 * 1 * 2.0 = 6.0 */
    TEST_ASSERT(fabsf(out[0] - 6.0f) < 1e-6f, "single element packed matmul");
}

static void test_all_zeros_weights(void) {
    int n = 32, d = 4;
    tn_i8 weights[32 * 4];
    memset(weights, 0, sizeof(weights)); /* all zero weights => value 0 */

    /* But pack_ternary maps 0 -> 0b01, so need to encode properly */
    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc(d * row_bytes, 1);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            pack_ternary(&packed[i * row_bytes], j, 0);
        }
    }

    float x[32];
    fill_input(x, n, 888);
    float scale = 1.0f;
    float out[4];
    ternary_matmul_packed(out, x, packed, n, d, &scale, 0);

    for (int i = 0; i < d; i++) {
        TEST_ASSERT(fabsf(out[i]) < 1e-6f, "all-zero weights produce zero output");
    }

    free(packed);
}

static void test_large_dimension(void) {
    tn_simd_init();

    int n = 4096, d = 1;
    tn_i8 *weights = (tn_i8 *)malloc(n);
    fill_ternary(weights, n, 12345);

    float *x = (float *)malloc(n * sizeof(float));
    fill_input(x, n, 54321);

    /* Pack */
    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc(row_bytes, 1);
    for (int j = 0; j < n; j++) {
        pack_ternary(packed, j, weights[j]);
    }

    /* Reference via unpacked matmul */
    float ref_out[1];
    float scale = 1.0f;
    ternary_matmul(ref_out, x, weights, n, d, scale);

    /* Packed matmul via dispatch */
    float packed_out[1];
    tn_ternary_matmul_packed(packed_out, x, packed, n, d, &scale, 0);

    float diff = fabsf(ref_out[0] - packed_out[0]);
    /*
     * Phase 16-S: dispatch now selects VNNI (int8) kernel over AVX2/AVX-512F.
     * VNNI introduces int8 quantisation error: for n=4096 and scale=1.0,
     * max accumulated error ≈ n * (1/127) * scale ≈ 32. Use 5% relative +
     * 1.0 absolute tolerance (same as test_simd_vnni.c VNNI validation).
     */
    float atol = 1.0f + 0.05f * fabsf(ref_out[0]);
    TEST_ASSERT(diff < atol, "large dim packed matmul matches reference");

    free(weights); free(x); free(packed);
}

/* ================================================================
 * 7. Group size equals matrix size (should match per-matrix)
 * ================================================================ */

static void test_group_size_equals_n(void) {
    int n = 32, d = 2;
    float scale = 1.5f;

    tn_i8 weights[32 * 2];
    fill_ternary(weights, n * d, 777);

    float x[32];
    fill_input(x, n, 888);

    size_t row_bytes = packed_bytes(n);
    tn_u8 *packed = (tn_u8 *)calloc(d * row_bytes, 1);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            pack_ternary(&packed[i * row_bytes], j, weights[i * n + j]);
        }
    }

    /* Per-matrix: single scale */
    float per_matrix_out[2];
    ternary_matmul_packed(per_matrix_out, x, packed, n, d, &scale, 0);

    /* Per-group with group_size == n: each row has 1 group, same scale */
    float group_scales[2] = {scale, scale}; /* d * 1 groups */
    float per_group_out[2];
    ternary_matmul_packed(per_group_out, x, packed, n, d, group_scales, n);

    for (int i = 0; i < d; i++) {
        float diff = fabsf(per_matrix_out[i] - per_group_out[i]);
        TEST_ASSERT(diff < 1e-5f,
                    "group_size==n matches per-matrix result");
    }

    free(packed);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    tn_simd_init();

    /* Pack/unpack round-trip */
    RUN_TEST(test_pack_unpack_roundtrip);
    RUN_TEST(test_pack_unpack_single);
    RUN_TEST(test_pack_unpack_odd_count);
    RUN_TEST(test_packed_bytes_calc);

    /* AVX2 unpack correctness */
    RUN_TEST(test_avx2_unpack_matches_scalar);
    RUN_TEST(test_avx2_unpack_odd_count);

    /* Packed matmul correctness */
    RUN_TEST(test_packed_matmul_per_matrix);
    RUN_TEST(test_packed_matmul_per_group);
    RUN_TEST(test_avx2_packed_matmul_matches_scalar);

    /* Edge cases */
    RUN_TEST(test_single_element);
    RUN_TEST(test_all_zeros_weights);
    RUN_TEST(test_large_dimension);
    RUN_TEST(test_group_size_equals_n);

    TEST_SUMMARY();
}
