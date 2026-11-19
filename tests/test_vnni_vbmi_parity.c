/*
 * tests/test_vnni_vbmi_parity.c — the VBMI build of the VNNI ternary kernels
 * must compute exactly what the SSE-unpack build computes (2026-11-18).
 *
 * ternary_matmul_packed_vnni.c is compiled twice (ternary_matmul_packed_vnni_vbmi.c
 * adds the 3-instruction VBMI unpack and the _vbmi name suffix). Both unpack the
 * same 2-bit codes and run the same VNNI accumulation, so the outputs must be
 * bit-identical for every size, including tails that are not a multiple of 64.
 * Skips when the build has no VBMI variant or the CPU's VBMI is not execution-
 * verified (core/cpu_features.c): then tn_simd_init never binds the _vbmi pair.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/cpu_features.h"
#include "core/platform.h"
#include "math/quantize_i8.h"
#include "math/ternary_matmul_packed.h"
#include "test_harness.h"

#if TN_HAS_AVX512VNNI && TN_HAS_AVX512VBMI

static unsigned lcg(unsigned *s) {
    *s = *s * 1103515245u + 12345u;
    return (*s >> 16) & 0x7FFF;
}

static void run_size(int n, int d, unsigned seed) {
    size_t row_bytes = ((size_t)n + 3) >> 2;
    float *x         = (float *)malloc((size_t)n * sizeof(float));
    tn_u8 *w         = (tn_u8 *)calloc((size_t)d * row_bytes, 1);
    float *a         = (float *)calloc((size_t)d, sizeof(float));
    float *b         = (float *)calloc((size_t)d, sizeof(float));
    int8_t *q        = (int8_t *)malloc((size_t)n);
    TEST_ASSERT(x && w && a && b && q, "alloc");
    if (!(x && w && a && b && q)) {
        free(x);
        free(w);
        free(a);
        free(b);
        free(q);
        return;
    }

    for (int i = 0; i < n; i++) x[i] = ((float)lcg(&seed) / 16384.0f) - 1.0f;
    for (int r = 0; r < d; r++)
        for (int j = 0; j < n; j++)
            w[(size_t)r * row_bytes + (size_t)(j >> 2)] |=
                (tn_u8)((lcg(&seed) % 3) << ((j & 3) << 1));
    const float scale = 0.37f;

    ternary_matmul_packed_vnni(a, x, w, n, d, &scale, 0);
    ternary_matmul_packed_vnni_vbmi(b, x, w, n, d, &scale, 0);
    TEST_ASSERT(memcmp(a, b, (size_t)d * sizeof(float)) == 0,
                "vnni and vnni_vbmi outputs are bit-identical");

    float act   = quantize_row_to_i8(x, q, n);
    int32_t sum = sum_i8(q, n);
    memset(a, 0, (size_t)d * sizeof(float));
    memset(b, 0, (size_t)d * sizeof(float));
    ternary_matmul_packed_vnni_preq(a, q, act, sum, w, n, d, &scale, 0);
    ternary_matmul_packed_vnni_preq_vbmi(b, q, act, sum, w, n, d, &scale, 0);
    TEST_ASSERT(memcmp(a, b, (size_t)d * sizeof(float)) == 0,
                "vnni_preq and vnni_preq_vbmi outputs are bit-identical");

    free(x);
    free(w);
    free(a);
    free(b);
    free(q);
}

static void test_parity(void) {
    const TnCpuFeatures *cpu = tn_cpu_features_detect();
    if (!cpu->avx512vnni || !cpu->avx512vbmi) {
        printf("CPU has no verified AVX-512 VNNI+VBMI -- skipping\n");
        return;
    }
    static const int sizes[][2] = {{64, 8},         {128, 16},  {2560, 64},
                                   {2560 + 40, 24}, {6912, 32}, {1000, 5}};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        run_size(sizes[i][0], sizes[i][1], 100u + i);
}

#else
static void test_parity(void) {
    printf("build has no AVX-512 VNNI+VBMI variant -- skipping\n");
}
#endif

int main(void) {
    RUN_TEST(test_parity);
    TEST_SUMMARY();
}
