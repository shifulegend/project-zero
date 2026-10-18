/*
 * tests/test_sampler_chain.c — configurable sampling pipeline.
 */
#include "test_harness.h"
#include "sampling/sampler_chain.h"
#include "sampling/logprob.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "math/simd_dispatch.h"
#include <math.h>

#define V 16

static void fill(float *l) { for (int i = 0; i < V; i++) l[i] = (float)i * 0.1f; }

static void test_defaults_neutral(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    TEST_ASSERT_FLOAT_EQ(ss.temperature, 1.0f, 1e-6, "temp 1");
    TEST_ASSERT_FLOAT_EQ(ss.top_p, 1.0f, 1e-6, "top_p 1");
    TEST_ASSERT_EQ(ss.top_k, 0, "top_k off");
    TEST_ASSERT(!sampler_penalties_active(&ss), "penalties off by default");
}

static void test_sanitize(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = -1; ss.top_p = 3; ss.top_k = -5; ss.min_p = 2;
    ss.repetition_penalty = 0; ss.presence_penalty = 9; ss.frequency_penalty = NAN;
    ss.penalty_last_n = -9;
    sampler_settings_sanitize(&ss);
    TEST_ASSERT_FLOAT_EQ(ss.temperature, 0.0f, 1e-6, "temp clamped");
    TEST_ASSERT_FLOAT_EQ(ss.top_p, 1.0f, 1e-6, "top_p reset");
    TEST_ASSERT_EQ(ss.top_k, 0, "top_k clamped");
    TEST_ASSERT_FLOAT_EQ(ss.min_p, 0.0f, 1e-6, "min_p reset");
    TEST_ASSERT_FLOAT_EQ(ss.repetition_penalty, 1.0f, 1e-6, "rep reset");
    TEST_ASSERT_FLOAT_EQ(ss.presence_penalty, 0.0f, 1e-6, "presence reset");
    TEST_ASSERT_FLOAT_EQ(ss.frequency_penalty, 0.0f, 1e-6, "freq reset");
    TEST_ASSERT_EQ(ss.penalty_last_n, -1, "last_n clamped");
}

static void test_greedy_matches_argmax(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = 0.0f;
    float l[V]; fill(l);
    unsigned long long rng = 1;
    TEST_ASSERT_EQ(sampler_chain_sample(l, V, &ss, NULL, 0, NULL, -1, &rng), V - 1, "greedy = argmax");
}

/* Default settings must reproduce the legacy temperature+top-p path exactly. */
static void test_bit_identical_to_legacy(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = 0.7f; ss.top_p = 0.9f;
    for (int seed = 1; seed < 50; seed++) {
        float a[V], b[V]; fill(a); fill(b);
        unsigned long long r1 = (unsigned long long)seed, r2 = r1;
        int x = sampler_chain_sample(a, V, &ss, NULL, 0, NULL, -1, &r1);
        apply_temperature(b, V, 0.7f);
        int y = sample_top_p(b, V, 0.9f, &r2);
        if (x != y) { TEST_ASSERT(0, "diverged from legacy path"); return; }
    }
    TEST_ASSERT(1, "identical to legacy temperature+top-p");
}

static void test_top_k_restricts(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = 5.0f; ss.top_k = 2;
    unsigned long long rng = 3;
    int ok = 1;
    for (int t = 0; t < 200; t++) {
        float l[V]; fill(l);
        int x = sampler_chain_sample(l, V, &ss, NULL, 0, NULL, -1, &rng);
        if (x < V - 2) ok = 0;
    }
    TEST_ASSERT(ok, "top_k=2 only samples the two best tokens");
}

static void test_min_p_restricts(void) {
    float l[V];
    for (int i = 0; i < V; i++) l[i] = 0.0f;
    l[3] = 10.0f; /* dominant */
    sampler_apply_truncation(l, V, 0, 0.5f, 1.0f);
    int kept = 0;
    for (int i = 0; i < V; i++) kept += isfinite(l[i]);
    TEST_ASSERT_EQ(kept, 1, "min_p keeps only tokens near the max");
}

static void test_penalties(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.repetition_penalty = 2.0f;
    ss.frequency_penalty = 0.5f;
    ss.presence_penalty = 0.25f;
    float l[V]; fill(l);
    l[4] = 2.0f; l[5] = -2.0f;
    int hist[] = {4, 4, 5, 9, 4};
    sampler_apply_penalties(l, V, &ss, hist, 5);
    /* token 4: 2.0/2 - 0.5*3 - 0.25 = -0.75 ; token 5: -2*2 - 0.5 - 0.25 = -4.75 */
    TEST_ASSERT_FLOAT_EQ(l[4], -0.75f, 1e-5, "positive logit penalised");
    TEST_ASSERT_FLOAT_EQ(l[5], -4.75f, 1e-5, "negative logit penalised");
    TEST_ASSERT_FLOAT_EQ(l[0], 0.0f, 1e-6, "unseen token untouched");

    /* window: last_n = 1 only sees the final '4' */
    float m[V]; fill(m); m[9] = 1.0f;
    ss.penalty_last_n = 1;
    sampler_apply_penalties(m, V, &ss, hist, 5);
    TEST_ASSERT_FLOAT_EQ(m[9], 1.0f, 1e-6, "outside window untouched");
}

static void test_seed_reproducible(void) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = 1.0f;
    unsigned long long r1, r2;
    rng_seed(&r1, 42); rng_seed(&r2, 42);
    int same = 1;
    for (int t = 0; t < 100; t++) {
        float a[V], b[V]; fill(a); fill(b);
        if (sampler_chain_sample(a, V, &ss, NULL, 0, NULL, -1, &r1) !=
            sampler_chain_sample(b, V, &ss, NULL, 0, NULL, -1, &r2)) same = 0;
    }
    TEST_ASSERT(same, "same seed -> same samples");
}

static void test_logsumexp(void) {
    float l[3] = {0.0f, 0.0f, 0.0f};
    TEST_ASSERT(fabs(logprob_logsumexp(l, 3) - log(3.0)) < 1e-9, "lse of zeros = ln n");
    float big[2] = {1000.0f, 1000.0f};
    TEST_ASSERT(fabs(logprob_logsumexp(big, 2) - (1000.0 + log(2.0))) < 1e-6, "stable for large logits");
    TEST_ASSERT(isinf(logprob_logsumexp(l, 0)), "empty -> -inf");
}

int main(void) {
    tn_simd_init();
    printf("=== Sampler Chain Tests ===\n");
    RUN_TEST(test_defaults_neutral);
    RUN_TEST(test_sanitize);
    RUN_TEST(test_greedy_matches_argmax);
    RUN_TEST(test_bit_identical_to_legacy);
    RUN_TEST(test_top_k_restricts);
    RUN_TEST(test_min_p_restricts);
    RUN_TEST(test_penalties);
    RUN_TEST(test_seed_reproducible);
    RUN_TEST(test_logsumexp);
    TEST_SUMMARY();
}
