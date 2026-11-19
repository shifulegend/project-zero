/*
 * test_cls_kernels.c — INT8 / INT4 classifier (LM head) row kernels (include/math/cls_kernels.h)
 *
 * Every implementation (generic AVX2/scalar, AVX-512, and whichever tn_simd_init() bound) must equal a
 * double-precision reference to float accumulation accuracy, for ragged shapes (n not a multiple of the 32-element
 * block, odd n for INT4, partial row ranges) and for the real BitNet and Mamba head shapes.
 * 2026-11-18: the kernels were rewritten to reduce once per row instead of once per block.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/platform.h"
#include "math/cls_kernels.h"
#include "math/simd_dispatch.h"
#include "test_harness.h"

static unsigned st = 99u;
static unsigned rnd32(void) {
    st = st * 1664525u + 1013904223u;
    return st >> 8;
}
static float frnd(void) {
    return (float)(rnd32() & 0xFFFF) / 32768.0f - 1.0f;
}

/* Returns the exact result; *mag receives the sum of absolute terms, the scale float accumulation error follows
 * (a result that nearly cancels is not a larger kernel error). */
static double ref_row(int is4, const tn_u8 *row, const float *x, const float *rs, int n,
                      double *mag) {
    double total = 0.0;
    *mag         = 0.0;
    const int nb = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    for (int b = 0; b < nb; b++) {
        double v = 0.0;
        for (int j = b * TN_CLS_QUANT_BLOCK; j < n && j < (b + 1) * TN_CLS_QUANT_BLOCK; j++) {
            int code = is4 ? ((j & 1) ? (row[j / 2] >> 4) & 0xF : row[j / 2] & 0xF) : row[j];
            v += (double)(code - (is4 ? 8 : 128)) * (double)x[j];
            *mag += fabs((double)(code - (is4 ? 8 : 128)) * (double)x[j]) * (double)rs[b];
        }
        total += v * (double)rs[b];
    }
    return total;
}

static double run_case(int is4, tn_cls_rows_fn fn, int n, int d, int start, int end) {
    const int nb    = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    const size_t rb = is4 ? ((size_t)n + 1) / 2 : (size_t)n;
    tn_u8 *w        = malloc(rb * (size_t)d);
    float *sc       = malloc(sizeof(float) * (size_t)nb * (size_t)d),
          *x        = malloc(sizeof(float) * (size_t)n);
    float *out      = malloc(sizeof(float) * (size_t)d);
    for (size_t i = 0; i < rb * (size_t)d; i++) w[i] = (tn_u8)rnd32();
    for (int i = 0; i < nb * d; i++) sc[i] = 0.01f + 0.05f * fabsf(frnd());
    for (int i = 0; i < n; i++) x[i] = frnd() * 3.0f;
    for (int i = 0; i < d; i++) out[i] = -777.0f; /* untouched rows must stay untouched */
    fn(out, x, w, sc, n, nb, start, end);
    double worst   = 0.0;
    int outside_ok = 1;
    for (int i = 0; i < d; i++) {
        if (i < start || i >= end) {
            outside_ok &= (out[i] == -777.0f);
            continue;
        }
        double mag, r = ref_row(is4, w + (size_t)i * rb, x, sc + (size_t)i * nb, n, &mag);
        double e =
            fabs((double)out[i] - r) / (mag + 1e-3); /* relative to the sum of absolute terms */
        if (e > worst) worst = e;
    }
    free(w);
    free(sc);
    free(x);
    free(out);
    return outside_ok ? worst : 1e9;
}

typedef struct {
    const char *name;
    int is4;
    tn_cls_rows_fn fn;
} Impl;

static int impls(Impl *v) {
    int k  = 0;
    v[k++] = (Impl){"i8 generic", 0, tn_cls_i8_rows_generic};
    v[k++] = (Impl){"i4 generic", 1, tn_cls_i4_rows_generic};
#if TN_HAS_AVX512
    v[k++] = (Impl){"i8 avx512", 0, tn_cls_i8_rows_avx512};
    v[k++] = (Impl){"i4 avx512", 1, tn_cls_i4_rows_avx512};
#endif
    tn_simd_init(); /* what the engine actually runs */
    v[k++] = (Impl){"i8 dispatched", 0, tn_cls_i8_rows};
    v[k++] = (Impl){"i4 dispatched", 1, tn_cls_i4_rows};
    return k;
}

static void test_shapes_match_reference(void) {
    Impl v[8];
    int ni                = impls(v);
    const int shapes[][2] = {{32, 8},   {64, 5},    {100, 13}, {33, 7}, {2560, 37},
                             {768, 29}, {2563, 11}, {31, 4},   {1, 3}};
    for (int k = 0; k < ni; k++) {
        double worst = 0.0;
        for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            double e = run_case(v[k].is4, v[k].fn, shapes[s][0], shapes[s][1], 0, shapes[s][1]);
            if (e > worst) worst = e;
        }
        printf("    %-14s worst error / sum of |terms| = %.2e\n", v[k].name, worst);
        TEST_ASSERT(worst < 5e-7, v[k].name);
    }
}

static void test_partial_row_ranges(void) {
    Impl v[8];
    int ni = impls(v);
    for (int k = 0; k < ni; k++) {
        double e = run_case(v[k].is4, v[k].fn, 2560, 40, 7, 31); /* rows 7..30 only */
        TEST_ASSERT(e < 5e-7, v[k].name);
    }
}

static void test_real_head_shapes(void) {
    Impl v[8];
    int ni = impls(v);
    for (int k = 0; k < ni; k++) {
        TEST_ASSERT(run_case(v[k].is4, v[k].fn, 2560, 512, 0, 512) < 5e-7,
                    v[k].name); /* BitNet 2B-4T hidden size */
        TEST_ASSERT(run_case(v[k].is4, v[k].fn, 768, 512, 0, 512) < 5e-7,
                    v[k].name); /* Mamba-130M hidden size */
    }
}

int main(void) {
    printf("=== classifier row kernels ===\n");
    RUN_TEST(test_shapes_match_reference);
    RUN_TEST(test_partial_row_ranges);
    RUN_TEST(test_real_head_shapes);
    TEST_SUMMARY();
}
