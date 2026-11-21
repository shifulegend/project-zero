/*
 * tests/test_rope_yarn_corr.c — rope_yarn_corr_dims / rope_corr_from_config return, bit for bit,
 * what the four inline copies in the attention files computed (the formula below is that copy,
 * kept here as the oracle), over a grid of head sizes, context lengths, bases and betas.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/clamp.h"
#include "core/config.h"
#include "math/rope.h"
#include "test_harness.h"

/* The old inline code, unchanged. */
static void old_corr(const Config *cfg, int head_dim, float corr[2]) {
    corr[0] = 0.0f;
    corr[1] = 0.0f;
    if (cfg->rope_yarn_ext_factor != 0.0f) {
        static const float M_PI_F = 3.14159265358979323846f;
        float log_base            = logf(cfg->rope_theta);
        float start               = floorf(
            (float)head_dim *
            logf((float)cfg->rope_orig_ctx_len / (cfg->rope_yarn_beta_fast * 2.0f * M_PI_F)) /
            (2.0f * log_base));
        float end =
            ceilf((float)head_dim *
                  logf((float)cfg->rope_orig_ctx_len / (cfg->rope_yarn_beta_slow * 2.0f * M_PI_F)) /
                  (2.0f * log_base));
        corr[0] = tn_clampf(start, 0.0f, (float)(head_dim - 1));
        corr[1] = tn_clampf(end, 0.0f, (float)(head_dim - 1));
    }
}

/* One grid point against the old code (and, when the extension is on, the direct helper). */
static int mismatch_at(int dim, int ctx, float base, float fast, float slow, float ext) {
    Config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.rope_orig_ctx_len    = ctx;
    cfg.rope_theta           = base;
    cfg.rope_yarn_beta_fast  = fast;
    cfg.rope_yarn_beta_slow  = slow;
    cfg.rope_yarn_ext_factor = ext;
    float want[2], got[2], direct[2];
    old_corr(&cfg, dim, want);
    rope_corr_from_config(&cfg, dim, got);
    int bad = memcmp(want, got, sizeof want) != 0;
    if (ext != 0.0f) {
        rope_yarn_corr_dims(dim, ctx, base, fast, slow, direct);
        bad |= memcmp(want, direct, sizeof want) != 0;
    }
    return bad;
}

static void test_matches_old_inline_code(void) {
    static const int dims[]    = {2, 8, 32, 64, 80, 96, 128, 160, 192, 256, 576};
    static const int ctxs[]    = {512, 2048, 4096, 8192, 32768, 131072, 262144};
    static const float bases[] = {1000.0f, 10000.0f, 500000.0f, 1000000.0f, 10000000.0f};
    static const float fasts[] = {1.0f, 8.0f, 32.0f, 64.0f};
    static const float slows[] = {0.5f, 1.0f, 2.0f, 4.0f};
    static const float exts[]  = {0.0f, 1.0f, 0.1f};
#define N(a) (sizeof(a) / sizeof((a)[0]))
    size_t total = N(dims) * N(ctxs) * N(bases) * N(fasts) * N(slows) * N(exts);
    long bad     = 0;
    for (size_t i = 0; i < total; i++) { /* mixed-radix walk over the grid */
        size_t r = i;
        int dim  = dims[r % N(dims)];
        r /= N(dims);
        int ctx = ctxs[r % N(ctxs)];
        r /= N(ctxs);
        float base = bases[r % N(bases)];
        r /= N(bases);
        float fast = fasts[r % N(fasts)];
        r /= N(fasts);
        float slow = slows[r % N(slows)];
        r /= N(slows);
        bad += mismatch_at(dim, ctx, base, fast, slow, exts[r % N(exts)]);
    }
    TEST_ASSERT(total > 10000, "grid size");
    TEST_ASSERT(bad == 0, "helper == the old inline computation, bit for bit");
#undef N
}

int main(void) {
    printf("=== rope YaRN correction ===\n");
    RUN_TEST(test_matches_old_inline_code);
    TEST_SUMMARY();
}
