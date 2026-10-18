/*
 * tests/test_cache_probe.c — Phase 27.1 cache geometry + 27.3 tiled
 * attention kernel equivalence with the standard softmax path.
 */
#include "test_harness.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"
#include "math/simd_dispatch.h"
#include "sampling/rng.h"

static void test_parse_size(void) {
    TEST_ASSERT_EQ(tn_cache_parse_size("48K"), 49152, "48K");
    TEST_ASSERT_EQ(tn_cache_parse_size("2048K\n"), 2097152, "2048K with newline");
    TEST_ASSERT_EQ(tn_cache_parse_size("260M"), 272629760LL, "260M");
    TEST_ASSERT_EQ(tn_cache_parse_size("64"), 64, "plain bytes");
    TEST_ASSERT_EQ(tn_cache_parse_size("abc"), -1, "garbage");
    TEST_ASSERT_EQ(tn_cache_parse_size("12Q"), -1, "bad unit");
    TEST_ASSERT_EQ(tn_cache_parse_size(NULL), -1, "NULL");
}

static void test_probe_sane(void) {
    const CacheInfo *ci = tn_cache_info();
    TEST_ASSERT(ci->l1d_bytes >= 4096 && ci->l1d_bytes <= (1 << 22), "L1d plausible");
    TEST_ASSERT(ci->l2_bytes >= ci->l1d_bytes, "L2 >= L1d");
    TEST_ASSERT(ci->line_bytes >= 16 && ci->line_bytes <= 512, "line size plausible");
    TEST_ASSERT(tn_cache_info() == ci, "probed once, cached");
    TEST_ASSERT(tn_cache_rows_fitting_l1(1 << 30, 0.5) == 1, "at least one row");
    TEST_ASSERT(tn_cache_rows_fitting_l1(512, 0.5) == ci->l1d_bytes / 2 / 512, "rows = fraction * L1d / row");
}

static void standard(float *out, const float *q, const float *K, const float *V, int hd,
                     const SlidingWindow *sw, int pos, int n, float sc) {
    float *att = (float *)malloc((size_t)n * sizeof(float));
    for (int t = 0; t < n; t++) {
        int lg = (pos >= n) ? (pos - n + 1 + t) : t;
        att[t] = tn_vec_dot(q, K + (size_t)sw_map_position(sw, lg) * hd, hd) * sc;
    }
    tn_softmax(att, n);
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < n; t++) {
        int lg = (pos >= n) ? (pos - n + 1 + t) : t;
        tn_vec_saxpy(out, att[t], V + (size_t)sw_map_position(sw, lg) * hd, hd);
    }
    free(att);
}

static void check_attn(int window, int pinned, int pos, int tile) {
    enum { HD = 64 };
    unsigned long long rng;
    rng_seed(&rng, (unsigned long long)(window * 131 + pos));
    float *K = (float *)malloc((size_t)window * HD * sizeof(float));
    float *V = (float *)malloc((size_t)window * HD * sizeof(float));
    float q[HD], a[HD], b[HD];
    for (int i = 0; i < window * HD; i++) { K[i] = rng_float(&rng) * 4 - 2; V[i] = rng_float(&rng) * 2 - 1; }
    for (int i = 0; i < HD; i++) q[i] = rng_float(&rng) * 2 - 1;
    SlidingWindow sw;
    sw_init(&sw, window, pinned);
    int n = sw_valid_count(&sw, pos);
    standard(a, q, K, V, HD, &sw, pos, n, 0.125f);
    attn_head_tiled(b, q, K, V, HD, &sw, pos, n, 0.125f, tile);
    float md = 0;
    for (int i = 0; i < HD; i++) { float e = fabsf(a[i] - b[i]); if (e > md) md = e; }
    char msg[96];
    snprintf(msg, sizeof(msg), "window=%d pos=%d tile=%d maxdiff=%g", window, pos, tile, md);
    TEST_ASSERT(md < 1e-5f, msg);
    free(K); free(V);
}

static void test_tiled_matches_standard(void) {
    check_attn(1, 1, 0, 8);          /* single position */
    check_attn(100, 100, 99, 8);     /* tiles with a ragged tail */
    check_attn(300, 300, 299, 1024); /* one tile */
    check_attn(64, 8, 200, 16);      /* wrapped sliding window with pinned prefix */
}

static void test_kernel_config(void) {
    TEST_ASSERT_EQ(attn_kernel_parse("tiled"), ATTN_KERNEL_TILED, "parse tiled");
    TEST_ASSERT_EQ(attn_kernel_parse("standard"), ATTN_KERNEL_STANDARD, "parse standard");
    TEST_ASSERT_EQ(attn_kernel_parse("flash"), -1, "reject unknown");
    attn_kernel_configure(ATTN_KERNEL_TILED, 3, 64);
    TEST_ASSERT_EQ(attn_tile_get(), 8, "tile clamped up");
    attn_kernel_configure(ATTN_KERNEL_TILED, 100000, 64);
    TEST_ASSERT_EQ(attn_tile_get(), ATTN_TILE_MAX, "tile clamped down");
    attn_kernel_configure(ATTN_KERNEL_STANDARD, 0, 64);
    TEST_ASSERT(attn_tile_get() >= 8, "auto tile from L1d");
    attn_parallel_min_ctx_set(-5);
    TEST_ASSERT_EQ(attn_parallel_min_ctx_get(), 0, "negative -> never");
    attn_parallel_min_ctx_set(ATTN_PARALLEL_MIN_CTX_DEFAULT);
}

int main(void) {
    tn_simd_init();
    printf("=== Cache Probe + Tiled Attention (Phase 27) Tests ===\n");
    RUN_TEST(test_parse_size);
    RUN_TEST(test_probe_sane);
    RUN_TEST(test_tiled_matches_standard);
    RUN_TEST(test_kernel_config);
    TEST_SUMMARY();
}
