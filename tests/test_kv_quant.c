/*
 * tests/test_kv_quant.c — plan 8.5 quantized KV rows: int8 error bound,
 * dot/axpy against F32, zero rows, type parsing; int4 (Q4_1-style groups)
 * layout, error bound, constant rows, and dot/axpy against a dequantised
 * reference; AVX2 / AVX-512 int4 kernels against the scalar reference
 * (directly and through tn_simd_init's dispatch).
 */
#include "test_harness.h"
#include "kv_cache/kv_quant_ops.h"
#include "sampling/rng.h"
#include "math/kvq4_kernels.h"
#include "math/simd_dispatch.h"
#include "math/cpu_features.h"
#include "core/platform.h"

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

static void test_i4_layout_and_roundtrip(void) {
    enum { N = 128 };
    int b = 0, sc = 0;
    TEST_ASSERT_EQ(kvq_row_layout(KV_TYPE_I4, N, &b, &sc), 0, "i4 layout for 128");
    TEST_ASSERT(b == N / 2 && sc == 2 * (N / TN_KV_I4_GROUP), "nibbles + (scale,min) per group");
    TEST_ASSERT_EQ(kvq_row_layout(KV_TYPE_I4, 48, &b, &sc), -1, "i4 needs a multiple of 32");
    TEST_ASSERT_EQ(kvq_row_layout(KV_TYPE_I8, 48, &b, &sc), 0, "i8 any width");
    TEST_ASSERT_EQ(kvq_row_layout(KV_TYPE_F32, 48, &b, &sc), -1, "f32 is not quantized");
    TEST_ASSERT_EQ(kv_type_parse("i4"), KV_TYPE_I4, "parse i4");

    unsigned long long r; rng_seed(&r, 33);
    float src[N], deq[N], scales[2 * (N / TN_KV_I4_GROUP)];
    tn_i8 q[N / 2];
    for (int i = 0; i < N; i++) src[i] = (rng_float(&r) - 0.3f) * 6.0f;    /* skewed */
    kvq_store(KV_TYPE_I4, q, scales, src, N);
    /* Dequantise via axpy with a = 1 into zeros. */
    for (int i = 0; i < N; i++) deq[i] = 0.0f;
    kvq_axpy_row(KV_TYPE_I4, deq, 1.0f, q, scales, N);
    int ok = 1;
    for (int g = 0; g < N / TN_KV_I4_GROUP; g++)
        for (int i = 0; i < TN_KV_I4_GROUP; i++) {
            int k = g * TN_KV_I4_GROUP + i;
            if (fabsf(deq[k] - src[k]) > 0.5f * scales[2 * g] + 1e-5f) ok = 0;
        }
    TEST_ASSERT(ok, "i4 reconstruction within half a step of its group");

    float qv[N];
    for (int i = 0; i < N; i++) qv[i] = rng_float(&r) - 0.5f;
    double ref = 0;
    for (int i = 0; i < N; i++) ref += (double)qv[i] * deq[i];
    TEST_ASSERT(fabs(kvq_dot_row(KV_TYPE_I4, qv, q, scales, N) - ref) < 1e-3,
                "i4 dot == dot with the dequantised row");

    float c[32], cs[2], back[32] = {0};
    tn_i8 cq[16];
    for (int i = 0; i < 32; i++) c[i] = 2.5f;
    kvq_store(KV_TYPE_I4, cq, cs, c, 32);
    kvq_axpy_row(KV_TYPE_I4, back, 1.0f, cq, cs, 32);
    int exact = 1;
    for (int i = 0; i < 32; i++) exact &= fabsf(back[i] - 2.5f) < 1e-6f;
    TEST_ASSERT(cs[0] == 0.0f && exact, "constant group is exact (zero scale, min = value)");

    float i8out[N] = {0};
    kvq_axpy_row(KV_TYPE_I8, i8out, 1.0f, q, scales, 4);  /* dispatch smoke: i8 path uses scales[0] */
    TEST_ASSERT(isfinite(i8out[0]), "i8 dispatch");
}

/* Compares one SIMD int4 kernel pair with the scalar reference on random rows
 * of several sizes (row values up to +-50 so d and m both matter). */
static void check_kvq4_kernels(const char *name, tn_kvq4_dot_fn dot, tn_kvq4_axpy_fn axpy) {
    enum { MAXN = 256 };
    static const int sizes[] = { 32, 64, 128, 256 };
    unsigned long long r; rng_seed(&r, 77);
    int ok_dot = 1, ok_axpy = 1;
    for (int si = 0; si < 4; si++) {
        int n = sizes[si];
        float src[MAXN], q[MAXN], scales[2 * MAXN / TN_KV_I4_GROUP];
        tn_i8 packed[MAXN / 2];
        for (int rep = 0; rep < 20; rep++) {
            float amp = rep < 10 ? 1.0f : 50.0f;
            for (int i = 0; i < n; i++) {
                src[i] = (rng_float(&r) - 0.3f) * amp;
                q[i] = rng_float(&r) - 0.5f;
            }
            kvq_store(KV_TYPE_I4, packed, scales, src, n);
            float ref = kvq4_dot_scalar(q, (const uint8_t *)packed, scales, n);
            float got = dot(q, (const uint8_t *)packed, scales, n);
            float tol = 1e-5f * (float)n * amp + 1e-5f;
            if (fabsf(ref - got) > tol) ok_dot = 0;
            float o_ref[MAXN], o_got[MAXN];
            for (int i = 0; i < n; i++) o_ref[i] = o_got[i] = q[i];
            kvq4_axpy_scalar(o_ref, 0.37f, (const uint8_t *)packed, scales, n);
            axpy(o_got, 0.37f, (const uint8_t *)packed, scales, n);
            for (int i = 0; i < n; i++)
                if (fabsf(o_ref[i] - o_got[i]) > 1e-5f * amp + 1e-6f) ok_axpy = 0;
        }
    }
    char m1[96], m2[96];
    snprintf(m1, sizeof(m1), "%s int4 dot matches scalar", name);
    snprintf(m2, sizeof(m2), "%s int4 axpy matches scalar", name);
    TEST_ASSERT(ok_dot, m1);
    TEST_ASSERT(ok_axpy, m2);
}

static void test_i4_simd_kernels(void) {
    const TnCpuFeatures *cpu = tn_cpu_features_detect();
#if TN_HAS_AVX2
    if (cpu->avx2) check_kvq4_kernels("AVX2", kvq4_dot_avx2, kvq4_axpy_avx2);
    else printf("  (skip AVX2: CPU lacks it)\n");
#endif
#if TN_HAS_AVX512
    if (cpu->avx512f) check_kvq4_kernels("AVX-512", kvq4_dot_avx512, kvq4_axpy_avx512);
    else printf("  (skip AVX-512: CPU lacks it)\n");
#endif
    (void)cpu;
    tn_simd_init();                                /* whatever this CPU selects */
    check_kvq4_kernels("dispatched", tn_kvq4_dot, tn_kvq4_axpy);
}

int main(void) {
    printf("=== KV int8 / int4 (plan 8.5) Tests ===\n");
    RUN_TEST(test_store_roundtrip_error);
    RUN_TEST(test_dot_and_axpy);
    RUN_TEST(test_zero_row_and_parse);
    RUN_TEST(test_i4_layout_and_roundtrip);
    RUN_TEST(test_i4_simd_kernels);
    TEST_SUMMARY();
}
