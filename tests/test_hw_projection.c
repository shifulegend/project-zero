/*
 * tests/test_hw_projection.c — the hardware profile's pre-load projections against a verbatim
 * copy of the code they were lifted out of (tn_hardware_profile_init's inline classifier switch
 * and L3-aware ceiling): tn_hw_classifier_bytes for every format, and tn_hw_cached_ceiling over
 * 200,000 generated (L3, bandwidth, weights, classifier) tuples including the exact boundaries
 * (L3 == classifier, == ternary bytes, == total). Exact double equality. Also the bandwidth probe's
 * buffer sizing over 400,000 (L3, free RAM, threads) tuples including the 512 MB / 2 GB clamps.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "runtime/hardware_profile.h"
#include "test_harness.h"

#define MODEL_TERNARY_BYTES (522ULL * 1024 * 1024)
#define MODEL_NORM_BYTES (1ULL * 1024 * 1024)
#define MODEL_DIM 2560
#define MODEL_VOCAB 128256

static double ref_cls_bytes(TnClassifierFormat fmt) {
    double cls_bytes;
    switch (fmt) {
        case TN_CLS_INT4:
            cls_bytes = (double)MODEL_VOCAB * MODEL_DIM / 2.0;
            break;
        case TN_CLS_INT8:
            cls_bytes = (double)MODEL_VOCAB * MODEL_DIM;
            break;
        default:
            cls_bytes = (double)MODEL_VOCAB * MODEL_DIM * 2.0;
            break;
    }
    return cls_bytes;
}

static double ref_ceiling(double l3_usable, double bw, double weight_bytes, double cls_bytes) {
    double dram_per_tok = weight_bytes;
    double ceiling;
    if (l3_usable > 0.0 && l3_usable < dram_per_tok) {
        double cls_cached = 0.0;
        if (cls_bytes < l3_usable) {
            cls_cached = cls_bytes;
        }
        double l3_for_layers  = l3_usable - cls_cached;
        double ternary_cached = 0.0;
        if (l3_for_layers > 0.0 && l3_for_layers < (double)MODEL_TERNARY_BYTES) {
            ternary_cached = l3_for_layers;
        } else if (l3_for_layers >= (double)MODEL_TERNARY_BYTES) {
            ternary_cached = (double)MODEL_TERNARY_BYTES;
        }
        dram_per_tok -= (cls_cached + ternary_cached);
    } else if (l3_usable >= dram_per_tok) {
        dram_per_tok = 0.0;
    }
    if (dram_per_tok > 0.0) {
        double seconds_per_tok = dram_per_tok / (bw * 1e9);
        ceiling                = 1.0 / seconds_per_tok;
    } else {
        ceiling = 999.0;
    }
    return ceiling;
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

/* The bandwidth probe's buffer sizing, verbatim from probe_dram_bandwidth before it was lifted. */
static size_t ref_bw_buffer(size_t l3, size_t free_ram_hint, int n_threads) {
    size_t total = (l3 > 0) ? 4 * l3 : 0;
    if (total < 512ULL * 1024 * 1024) total = 512ULL * 1024 * 1024;
    if (total > 2048ULL * 1024 * 1024) total = 2048ULL * 1024 * 1024;
    if (free_ram_hint > 0 && total > free_ram_hint / 4) total = free_ram_hint / 4;
    size_t per_thread = (total / (size_t)n_threads) & ~(size_t)4095;
    if (per_thread < 4096) return 0;
    return per_thread;
}

static void test_bw_buffer(void) {
    static const size_t EDGE[] = {0,
                                  1,
                                  4095,
                                  4096,
                                  8191,
                                  64ULL << 20,
                                  128ULL << 20,
                                  129ULL << 20,
                                  512ULL << 20,
                                  513ULL << 20,
                                  2048ULL << 20,
                                  4096ULL << 20,
                                  (size_t)1 << 40};
    int bad                    = 0;
    for (int t = 0; t < 400000; t++) {
        size_t l3   = rnd() % 4 == 0 ? EDGE[rnd() % (sizeof EDGE / sizeof EDGE[0])]
                                     : (size_t)(rnd() % (1ULL << 33));
        size_t free = rnd() % 4 == 0 ? EDGE[rnd() % (sizeof EDGE / sizeof EDGE[0])]
                                     : (size_t)(rnd() % (1ULL << 36));
        int threads = 1 + (int)(rnd() % 8);
        if (tn_hw_bw_buffer_bytes(l3, free, threads) != ref_bw_buffer(l3, free, threads)) bad++;
    }
    TEST_ASSERT(bad == 0, "probe buffer size equals the reference for every tuple");
}

static void test_classifier_bytes(void) {
    TnClassifierFormat f[] = {TN_CLS_BF16, TN_CLS_INT8, TN_CLS_INT4, (TnClassifierFormat)99};
    for (size_t i = 0; i < sizeof f / sizeof f[0]; i++)
        TEST_ASSERT(tn_hw_classifier_bytes(f[i]) == ref_cls_bytes(f[i]), "classifier bytes match");
}

static void test_ceiling(void) {
    int bad                   = 0;
    TnClassifierFormat fmts[] = {TN_CLS_BF16, TN_CLS_INT8, TN_CLS_INT4};
    for (int t = 0; t < 200000; t++) {
        TnClassifierFormat fmt = fmts[rnd() % 3];
        double cls             = ref_cls_bytes(fmt);
        double weights         = (double)MODEL_TERNARY_BYTES + cls + (double)MODEL_NORM_BYTES;
        double l3;
        switch (rnd() % 8) {
            case 0:
                l3 = 0.0;
                break;
            case 1:
                l3 = cls;
                break;
            case 2:
                l3 = (double)MODEL_TERNARY_BYTES;
                break;
            case 3:
                l3 = weights;
                break;
            case 4:
                l3 = cls + (double)MODEL_TERNARY_BYTES;
                break;
            case 5:
                l3 = weights * 2.0;
                break;
            default:
                l3 = (double)(rnd() % (uint64_t)(weights * 1.5));
                break;
        }
        double bw = 1.0 + (double)(rnd() % 4000) / 10.0;
        if (rnd() % 10 == 0) weights = (double)(rnd() % 8000000000ull) + 1.0; /* a loaded model */
        if (tn_hw_cached_ceiling(l3, bw, weights, cls) != ref_ceiling(l3, bw, weights, cls)) bad++;
    }
    TEST_ASSERT(bad == 0, "cached ceiling equals the reference for every tuple");
}

int main(void) {
    printf("=== hardware profile projections ===\n");
    RUN_TEST(test_classifier_bytes);
    RUN_TEST(test_ceiling);
    RUN_TEST(test_bw_buffer);
    TEST_SUMMARY();
}
