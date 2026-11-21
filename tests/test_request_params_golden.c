/*
 * tests/test_request_params_golden.c — request_params_resolve over 100,000 generated (request,
 * server defaults, limit) triples: out-of-range sampling values (they get sanitised), every subset
 * of the "set" bits, more stop strings than fit, request-only JSON mode, watermark key and knobs,
 * missing request or defaults, non-positive limits. A hash of every field of the resulting
 * GenerationParams (strings by content), recorded from the function before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "api/request_params.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x0455a098bd41aba5ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_str(const char *s) {
    if (!s) {
        mix("\x01", 1);
        return;
    }
    mix(s, strlen(s) + 1);
}

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

#define PICK(t) ((t)[rnd(sizeof(t) / sizeof((t)[0]))])

static const float k_floats[]    = {-3.0f, -0.5f, 0.0f, 0.3f, 0.7f, 1.0f, 1.5f, 2.0f, 9.0f, 1e9f};
static const int k_ints[]        = {-5, -1, 0, 1, 7, 40, 64, 512, 100000};
static const long long k_seed[]  = {-1, 0, 7, 123456789012LL};
static const char *const k_str[] = {"", "a", "###", "\nUser:", "stop here", "x", "<|end|>", "yy"};
static const char *const k_key[] = {NULL, "", "k", "secret passphrase"};
static const int k_limit[]       = {-5, 0, 1, 7, 100, 4096};

static void gen_defaults(GenerationParams *d) {
    gen_params_default(d);
    d->max_tokens                 = PICK(k_ints);
    d->sampler.temperature        = PICK(k_floats);
    d->sampler.top_p              = PICK(k_floats);
    d->sampler.top_k              = PICK(k_ints);
    d->sampler.min_p              = PICK(k_floats);
    d->sampler.repetition_penalty = PICK(k_floats);
    d->sampler.presence_penalty   = PICK(k_floats);
    d->sampler.frequency_penalty  = PICK(k_floats);
    d->sampler.penalty_last_n     = PICK(k_ints);
    d->seed                       = PICK(k_seed);
    d->json_mode                  = rnd(2);
    d->ignore_eos                 = rnd(2);
    d->raw_prompt                 = rnd(2);
    d->watermark.enabled          = rnd(2);
    d->watermark.key              = rnd(1000);
    d->watermark.gamma            = PICK(k_floats);
    d->watermark.delta            = PICK(k_floats);
    d->system_prompt              = PICK(k_str);
    d->chat_template              = rnd(2) ? PICK(k_str) : NULL;
    d->n_stop                     = (int)rnd(GEN_MAX_STOP + 1);
    for (int i = 0; i < d->n_stop; i++) d->stop[i] = PICK(k_str);
}

static void gen_request(ChatRequest *r) {
    chat_request_init(r);
    r->set_mask                    = rnd(1u << 20);
    r->sampling.temperature        = PICK(k_floats);
    r->sampling.top_p              = PICK(k_floats);
    r->sampling.top_k              = PICK(k_ints);
    r->sampling.min_p              = PICK(k_floats);
    r->sampling.repetition_penalty = PICK(k_floats);
    r->sampling.presence_penalty   = PICK(k_floats);
    r->sampling.frequency_penalty  = PICK(k_floats);
    r->sampling.penalty_last_n     = PICK(k_ints);
    r->sampling.seed               = PICK(k_seed);
    r->output.max_tokens           = PICK(k_ints);
    r->output.json_mode            = rnd(2);
    r->output.ignore_eos           = rnd(2);
    r->output.n_stop               = (int)rnd(CHAT_MAX_STOP + 1);
    for (int i = 0; i < r->output.n_stop; i++) r->output.stop[i] = (char *)PICK(k_str);
    r->watermark.enabled = rnd(2);
    r->watermark.key     = (char *)PICK(k_key);
    r->watermark.gamma   = PICK(k_floats);
    r->watermark.delta   = PICK(k_floats);
    r->chat_template     = (char *)(rnd(2) ? PICK(k_str) : NULL);
}

static void mix_params(const GenerationParams *p) {
    mix(&p->max_tokens, sizeof p->max_tokens);
    mix(&p->sampler, sizeof p->sampler);
    mix(&p->seed, sizeof p->seed);
    mix(&p->json_mode, sizeof p->json_mode);
    mix(&p->ignore_eos, sizeof p->ignore_eos);
    mix(&p->raw_prompt, sizeof p->raw_prompt);
    mix(&p->watermark.enabled, sizeof p->watermark.enabled);
    mix(&p->watermark.key, sizeof p->watermark.key);
    mix(&p->watermark.gamma, sizeof p->watermark.gamma);
    mix(&p->watermark.delta, sizeof p->watermark.delta);
    mix(&p->n_messages, sizeof p->n_messages);
    int null_conv = p->roles == NULL && p->contents == NULL;
    mix(&null_conv, sizeof null_conv);
    mix_str(p->system_prompt);
    mix_str(p->chat_template);
    mix_str(p->assistant_prefix);
    mix(&p->n_stop, sizeof p->n_stop);
    for (int i = 0; i < p->n_stop; i++) mix_str(p->stop[i]);
}

static void test_resolve_golden(void) {
    for (int t = 0; t < 100000; t++) {
        GenerationParams defaults, out;
        ChatRequest req;
        gen_defaults(&defaults);
        gen_request(&req);
        int limit        = PICK(k_limit);
        int use_req      = rnd(10) != 0;
        int use_defaults = rnd(10) != 0;
        memset(&out, 0xA5, sizeof out);
        request_params_resolve(use_req ? &req : NULL, use_defaults ? &defaults : NULL, limit, &out);
        mix_params(&out);
    }
    if (getenv("RP_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "resolved parameters match the recorded hash");
}

int main(void) {
    printf("=== request_params_resolve golden ===\n");
    RUN_TEST(test_resolve_golden);
    TEST_SUMMARY();
}
