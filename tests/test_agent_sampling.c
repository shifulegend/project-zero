/*
 * tests/test_agent_sampling.c — agent_sample_next (the token choice of the agent loop) against a
 * copy of the code it was lifted out of (the same statements, flattened to early returns), over
 * 300,000 generated cases: vocabularies of 1..96 logits (flat, peaked, extreme and negative
 * values), temperatures at and around the greedy cut (<= 0, 1e-7, 1e-6, ordinary, high), top-p at
 * and around its bounds (<= 0, tiny, ordinary, 1, above 1) and random generator states. Equal means
 * the chosen token, every logit left behind (the function rescales them in place) and the generator
 * state afterwards, bit for bit.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agent/agent_loop.h"
#include "math/simd_core_ops.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "test_harness.h"

static int ref_next(float *logits, int vocab_size, float temperature, float top_p,
                    unsigned long long *rng_state) {
    if (temperature <= 0.0f || temperature < 1e-6f) return sample_argmax(logits, vocab_size);
    apply_temperature(logits, vocab_size, temperature);
    if (top_p < 1.0f && top_p > 0.0f) return sample_top_p(logits, vocab_size, top_p, rng_state);
    tn_softmax(logits, vocab_size);
    float r   = rng_float(rng_state);
    float cdf = 0.0f;
    int next  = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) {
            next = i;
            break;
        }
    }
    return next;
}

static uint64_t g_rng = 0x428a2f98d728ae22ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static float gen_logit(int shape) {
    float u = (float)rnd(20001) / 10000.0f - 1.0f; /* -1..1 */
    switch (shape) {
        case 0:
            return 0.0f;
        case 1:
            return u * 3.0f;
        case 2:
            return u * 30.0f;
        case 3:
            return rnd(50) == 0 ? 25.0f : u;
        default:
            return -5.0f + u;
    }
}

static float pick_temperature(void) {
    static const float T[] = {-1.0f, 0.0f, 1e-7f, 5e-7f, 1e-6f, 2e-6f,
                              0.1f,  0.5f, 1.0f,  2.0f,  8.0f};
    return T[rnd(sizeof T / sizeof T[0])];
}

static float pick_top_p(void) {
    static const float P[] = {-1.0f, 0.0f, 1e-6f, 0.05f, 0.3f, 0.9f, 0.999f, 1.0f, 1.5f};
    return P[rnd(sizeof P / sizeof P[0])];
}

static void test_agent_sampling(void) {
    int bad = 0, sampled = 0;
    for (int t = 0; t < 300000; t++) {
        int vocab                = 1 + (int)rnd(96);
        int shape                = (int)rnd(5);
        float temp               = pick_temperature();
        float top_p              = pick_top_p();
        unsigned long long state = ((unsigned long long)rnd(1u << 30) << 20) ^ rnd(1u << 20) ^ 1ull;
        float a[96], b[96];
        for (int i = 0; i < vocab; i++) a[i] = b[i] = gen_logit(shape);
        unsigned long long sa = state, sb = state;
        int na = agent_sample_next(a, vocab, temp, top_p, &sa);
        int nb = ref_next(b, vocab, temp, top_p, &sb);
        sampled += temp >= 1e-6f;
        if (na != nb || sa != sb || memcmp(a, b, (size_t)vocab * sizeof(float)) != 0) bad++;
    }
    TEST_ASSERT(sampled > 100000, "enough cases take the sampled (not greedy) branch");
    TEST_ASSERT(bad == 0, "agent_sample_next equals the reference for every case");
}

int main(void) {
    printf("=== agent_sample_next equivalence ===\n");
    RUN_TEST(test_agent_sampling);
    TEST_SUMMARY();
}
