/*
 * tests/test_simd_tier_kernel.c — a forced SIMD tier must change the kernel
 * that parallel_ternary_matmul_packed runs (2026-11-16).
 *
 * On a -march=native build for an AVX-512 VNNI host, parallel_matmul.c used to
 * quantise the activations to int8 and take the VNNI "preq" kernel for EVERY
 * tier, so TN_FORCE_BACKEND=scalar|avx2|avx512f (and the calibration sweep that
 * relies on it) silently ran VNNI. The two kernel families give different
 * numbers: the float kernels agree with the scalar reference to rounding error,
 * the int8 kernel differs by activation-quantisation error (~0.3% at n=1024).
 * That difference is the observable: no hook into the kernels is needed.
 *
 * Skips when the build or the CPU has no VNNI path (then the tiers are honoured
 * by construction: the function pointer chosen by tn_simd_init is the only path).
 */
#include "test_harness.h"
#include "core/platform.h"
#include "math/cpu_features.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "math/ternary_matmul_packed.h"
#include "threading/thread_pool.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { N = 1024, D = 64 };

static void fill_x(float *x, unsigned seed) {
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((float)(seed & 0x7FFFu) / 16384.0f) - 1.0f;
    }
}

static void fill_packed(tn_u8 *w, unsigned seed) {
    size_t row_bytes = ((size_t)N + 3) >> 2;
    memset(w, 0, (size_t)D * row_bytes);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < N; j++) {
            seed    = seed * 1103515245u + 12345u;
            int enc = (int)((seed >> 16) % 3); /* {0,1,2} = {-1,0,+1} */
            w[(size_t)i * row_bytes + (size_t)(j >> 2)] |= (tn_u8)(enc << ((j & 3) << 1));
        }
}

static double max_rel_diff(const float *a, const float *b) {
    double m = 0.0, big = 1e-30;
    for (int i = 0; i < D; i++) {
        if (fabs(a[i]) > big) big = fabs(a[i]);
    }
    for (int i = 0; i < D; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m / big;
}

/* Runs the parallel packed matmul with TN_FORCE_BACKEND=tier; returns 0 when
 * the tier is not available on this build/CPU (name not honoured). */
static void run_tier(const char *tier, float *out, const float *x, const tn_u8 *w, ThreadPool *tp) {
    if (tier)
        setenv("TN_FORCE_BACKEND", tier, 1);
    else
        unsetenv("TN_FORCE_BACKEND");
    tn_simd_init();
    parallel_ternary_matmul_packed(out, x, w, N, D, 0.37f, tp);
}

static void test_forced_tiers_change_the_kernel(void) {
#if TN_HAS_AVX512VNNI
    const TnCpuFeatures *cpu = tn_cpu_features_detect();
    if (!cpu->avx512vnni) {
        printf("CPU has no AVX-512 VNNI -- skipping\n");
        return;
    }

    float x[N], ref[D], out_vnni[D], out[D];
    tn_u8 *w = (tn_u8 *)malloc((size_t)D * (((size_t)N + 3) >> 2));
    TEST_ASSERT(w != NULL, "alloc");
    if (!w) return;
    fill_x(x, 7u);
    fill_packed(w, 11u);
    float scale = 0.37f;
    ternary_matmul_packed(ref, x, w, N, D, &scale, 0); /* scalar float reference */

    ThreadPool *tp = threadpool_create(2);
    TEST_ASSERT(tp != NULL, "thread pool");
    if (!tp) {
        free(w);
        return;
    }

    run_tier("vnni", out_vnni, x, w, tp);
    double vnni_vs_ref = max_rel_diff(out_vnni, ref);
    /* The int8 kernel is measurably different from the float reference, otherwise
     * this test could not tell the families apart. */
    TEST_ASSERT(vnni_vs_ref > 1e-4,
                "vnni (int8) output differs from the float reference by quantisation error");

    static const char *const FLOAT_TIERS[] = {"scalar", "avx2", "avx512f"};
    for (int i = 0; i < 3; i++) {
        run_tier(FLOAT_TIERS[i], out, x, w, tp);
        double to_ref  = max_rel_diff(out, ref);
        double to_vnni = max_rel_diff(out, out_vnni);
        printf("  tier %-8s: rel diff vs float reference %.2e, vs vnni %.2e\n", FLOAT_TIERS[i],
               to_ref, to_vnni);
        TEST_ASSERT(to_ref < 1e-4,
                    "a forced float tier runs a float kernel (matches the scalar reference)");
        TEST_ASSERT(to_vnni > 1e-4, "a forced float tier does not run the VNNI int8 kernel");
    }

    run_tier(NULL, out, x, w, tp); /* default tier */
    TEST_ASSERT(max_rel_diff(out, out_vnni) < 1e-6,
                "the default tier on a VNNI host is the VNNI kernel");

    threadpool_destroy(tp);
    free(w);
    unsetenv("TN_FORCE_BACKEND");
    tn_simd_init();
#else
    printf("build has no AVX-512 VNNI path -- skipping\n");
#endif
}

int main(void) {
    RUN_TEST(test_forced_tiers_change_the_kernel);
    TEST_SUMMARY();
}
