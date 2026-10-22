/*
 * tests/test_watermark.c — Phase 33: green-list watermark bias + detection.
 */
#include "test_harness.h"
#include "sampling/watermark.h"
#include "sampling/rng.h"
#include "sampling/sampler_chain.h"
#include "math/simd_dispatch.h"
#include <math.h>

#define VOCAB 512

static void test_defaults_and_sanitize(void) {
    WatermarkConfig wm;
    watermark_config_default(&wm);
    TEST_ASSERT(!wm.enabled, "disabled by default");
    TEST_ASSERT_FLOAT_EQ(wm.gamma, WATERMARK_DEFAULT_GAMMA, 1e-6, "default gamma");
    TEST_ASSERT_FLOAT_EQ(wm.delta, WATERMARK_DEFAULT_DELTA, 1e-6, "default delta");
    wm.gamma = 5.0f; wm.delta = -3.0f;
    watermark_config_sanitize(&wm);
    TEST_ASSERT_FLOAT_EQ(wm.gamma, 0.99f, 1e-6, "gamma clamped high");
    TEST_ASSERT_FLOAT_EQ(wm.delta, 0.0f, 1e-6, "delta clamped low");
    wm.gamma = NAN;
    watermark_config_sanitize(&wm);
    TEST_ASSERT_FLOAT_EQ(wm.gamma, 0.01f, 1e-6, "NaN gamma clamped");
}

static void test_key_from_string(void) {
    TEST_ASSERT_EQ(watermark_key_from_string(NULL), 0u, "NULL -> 0");
    TEST_ASSERT_EQ(watermark_key_from_string(""), 0u, "empty -> 0");
    TEST_ASSERT(watermark_key_from_string("a") != watermark_key_from_string("b"), "distinct keys");
    TEST_ASSERT(watermark_key_from_string("abc") == watermark_key_from_string("abc"), "deterministic");
}

static void test_green_fraction_matches_gamma(void) {
    WatermarkConfig wm;
    watermark_config_default(&wm);
    wm.key = 1234;
    int green = 0, total = 0;
    for (int prev = 0; prev < 64; prev++)
        for (int t = 0; t < VOCAB; t++, total++) green += watermark_is_green(&wm, prev, t);
    double frac = (double)green / total;
    TEST_ASSERT(fabs(frac - wm.gamma) < 0.01, "green fraction ~= gamma");
}

static void test_bias_only_green_when_enabled(void) {
    WatermarkConfig wm;
    watermark_config_default(&wm);
    wm.key = 99;
    float logits[VOCAB] = {0};
    watermark_bias_logits(&wm, logits, VOCAB, 5);
    int changed = 0;
    for (int i = 0; i < VOCAB; i++) changed += logits[i] != 0.0f;
    TEST_ASSERT_EQ(changed, 0, "no-op while disabled");

    wm.enabled = true;
    watermark_bias_logits(&wm, logits, VOCAB, 5);
    int ok = 1;
    for (int i = 0; i < VOCAB; i++) {
        float want = watermark_is_green(&wm, 5, i) ? wm.delta : 0.0f;
        if (logits[i] != want) ok = 0;
    }
    TEST_ASSERT(ok, "exactly green tokens get +delta");
}

/* Random (unwatermarked) sequences must essentially never cross z >= 4;
 * sequences sampled with the bias must almost always do. */
static void test_detect_separates_populations(void) {
    WatermarkConfig wm;
    watermark_config_default(&wm);
    wm.key = watermark_key_from_string("unit-test-key");
    unsigned long long rng;
    rng_seed(&rng, 2026);

    enum { N = 200, TRIALS = 200 };
    int toks[N];
    int false_pos = 0;
    double max_null_z = -1e9;
    for (int t = 0; t < TRIALS; t++) {
        for (int i = 0; i < N; i++) toks[i] = (int)(rng_float(&rng) * VOCAB) % VOCAB;
        WatermarkResult r;
        watermark_detect(&wm, toks, N, &r);
        if (r.z_score > max_null_z) max_null_z = r.z_score;
        if (r.z_score >= 4.0) false_pos++;
    }
    TEST_ASSERT_EQ(false_pos, 0, "no false positives at z>=4 over random text");

    /* Watermarked: sample from a flat distribution + bias via the real chain. */
    wm.enabled = true;
    SamplerSettings ss;
    sampler_settings_default(&ss);
    int detected = 0;
    for (int t = 0; t < 20; t++) {
        int prev = -1;
        for (int i = 0; i < N; i++) {
            float logits[VOCAB];
            for (int v = 0; v < VOCAB; v++) logits[v] = 0.0f;
            toks[i] = sampler_chain_sample(logits, VOCAB, &ss, NULL, 0, &wm, prev, &rng);
            prev = toks[i];
        }
        WatermarkResult r;
        watermark_detect(&wm, toks, N, &r);
        if (r.z_score >= 4.0) detected++;
    }
    TEST_ASSERT_EQ(detected, 20, "every watermarked sequence detected");
}

static void test_detect_dedups_repeated_bigrams(void) {
    WatermarkConfig wm;
    watermark_config_default(&wm);
    wm.key = 7;
    /* Find one green bigram and repeat it: must count once, not 100 times. */
    int a = 0, b = 0;
    for (b = 0; b < VOCAB && !watermark_is_green(&wm, a, b); b++) {}
    int toks[201];
    for (int i = 0; i < 201; i++) toks[i] = (i % 2 == 0) ? a : b;
    WatermarkResult r;
    watermark_detect(&wm, toks, 201, &r);
    TEST_ASSERT(r.n_scored <= 2, "repeated bigrams scored once");
    TEST_ASSERT(r.z_score < 4.0, "repetition alone cannot trigger detection");

    watermark_detect(&wm, toks, 1, &r);
    TEST_ASSERT_EQ(r.n_scored, 0, "single token: nothing to score");
    TEST_ASSERT_FLOAT_EQ(r.z_score, 0.0, 1e-9, "z = 0 with no data");
}

int main(void) {
    tn_simd_init();
    printf("=== Watermark (Phase 33) Tests ===\n");
    RUN_TEST(test_defaults_and_sanitize);
    RUN_TEST(test_key_from_string);
    RUN_TEST(test_green_fraction_matches_gamma);
    RUN_TEST(test_bias_only_green_when_enabled);
    RUN_TEST(test_detect_separates_populations);
    RUN_TEST(test_detect_dedups_repeated_bigrams);
    TEST_SUMMARY();
}
