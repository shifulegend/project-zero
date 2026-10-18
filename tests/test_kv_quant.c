/*
 * tests/test_kv_quant.c — plan 8.5 int8 KV rows: quantisation error bound,
 * dot/axpy against F32, zero rows, and type parsing.
 */
#include "test_harness.h"
#include "kv_cache/kv_quant_ops.h"
#include "sampling/rng.h"

static void test_store_roundtrip_error(void) {
    enum { N = 64 };
    unsigned long long r; rng_seed(&r, 9);
    float src[N]; tn_i8 q[N]; float scale;
    for (int i = 0; i < N; i++) src[i] = (rng_float(&r) - 0.5f) * 8.0f;
    kvq_store_row(q, &scale, src, N);
    float amax = 0; for (int i = 0; i < N; i++) if (fabsf(src[i]) > amax) amax = fabsf(src[i]);
    TEST_ASSERT_FLOAT_EQ(scale, amax / 127.0f, 1e-6, "scale = absmax / 127");
    float maxerr = 0;
    for (int i = 0; i < N; i++) { float e = fabsf((float)q[i] * scale - src[i]); if (e > maxerr) maxerr = e; }
    TEST_ASSERT(maxerr <= 0.5f * scale + 1e-6f, "reconstruction error <= half a step");
}

static void test_dot_and_axpy(void) {
    enum { N = 67 };
    unsigned long long r; rng_seed(&r, 21);
    float k[N], v[N], q[N]; tn_i8 kq[N], vq[N]; float ks, vs;
    for (int i = 0; i < N; i++) { k[i] = rng_float(&r) - 0.5f; v[i] = rng_float(&r) - 0.5f; q[i] = rng_float(&r) - 0.5f; }
    kvq_store_row(kq, &ks, k, N);
    kvq_store_row(vq, &vs, v, N);
    double ref = 0; for (int i = 0; i < N; i++) ref += (double)q[i] * k[i];
    float got = kvq_dot(q, kq, N) * ks;
    TEST_ASSERT(fabs(got - ref) < 0.02, "int8 dot close to F32 dot");
    float out[N] = {0}, ref_out[N] = {0};
    kvq_axpy(out, 0.7f * vs, vq, N);
    for (int i = 0; i < N; i++) ref_out[i] = 0.7f * v[i];
    float md = 0; for (int i = 0; i < N; i++) { float e = fabsf(out[i] - ref_out[i]); if (e > md) md = e; }
    TEST_ASSERT(md < 0.7f * vs, "axpy within one quantisation step");
}

static void test_zero_row_and_parse(void) {
    float z[8] = {0}; tn_i8 q[8]; float s = 1.0f;
    kvq_store_row(q, &s, z, 8);
    TEST_ASSERT(s == 0.0f, "zero row -> zero scale");
    int all0 = 1; for (int i = 0; i < 8; i++) all0 &= q[i] == 0;
    TEST_ASSERT(all0, "zero row -> zero codes");
    TEST_ASSERT_EQ(kv_type_parse("i8"), KV_TYPE_I8, "parse i8");
    TEST_ASSERT_EQ(kv_type_parse("f32"), KV_TYPE_F32, "parse f32");
    TEST_ASSERT_EQ(kv_type_parse("q4"), -1, "reject unknown");
}

int main(void) {
    printf("=== KV int8 (plan 8.5) Tests ===\n");
    RUN_TEST(test_store_roundtrip_error);
    RUN_TEST(test_dot_and_axpy);
    RUN_TEST(test_zero_row_and_parse);
    TEST_SUMMARY();
}
