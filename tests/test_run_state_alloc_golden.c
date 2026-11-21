/*
 * tests/test_run_state_alloc_golden.c — run_state_alloc_ex over 4,000 generated model shapes and
 * KV options: head counts and sizes (including dim 0, which must fail), context lengths from negative
 * and empty to a few dozen positions, F32 / int8 / int4 KV storage (int4 on head sizes that do and
 * do not divide into its groups, which falls back to int8), skipped KV caches, and the disk-offload
 * path in a temporary directory (and a missing one). For each: the result code, every field the
 * allocator sets, which buffers exist and the RoPE table; the last element of every buffer is
 * written under ASan, so a buffer smaller than the forward pass needs fails loudly. Hash recorded
 * from the allocator before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kv_cache/kv_strategy.h"
#include "runtime/run_state.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x89782fa23aa7dce8ull
#define N_CASES 4000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x6a09e667f3bcc909ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static char g_dir[64];

static void gen_shape(Config *cfg, int *seq) {
    static const int HEAD_DIMS[] = {2, 4, 8, 16, 32, 64, 96};
    memset(cfg, 0, sizeof *cfg);
    cfg->n_heads    = 1 + (int)rnd(4);
    cfg->n_kv_heads = 1 + (int)rnd(4);
    int hd          = HEAD_DIMS[rnd(sizeof HEAD_DIMS / sizeof HEAD_DIMS[0])];
    cfg->dim        = rnd(40) == 0 ? 0 : cfg->n_heads * hd + (rnd(30) == 0 ? 1 : 0);
    cfg->hidden_dim = rnd(30) == 0 ? 0 : cfg->dim * 2;
    cfg->n_layers   = 1 + (int)rnd(6);
    cfg->vocab_size = rnd(30) == 0 ? 0 : 1 + (int)rnd(300);
    cfg->rope_theta = rnd(3) == 0 ? 0.0f : (rnd(2) ? 10000.0f : 500000.0f);
    unsigned k      = rnd(20);
    *seq            = k == 0 ? 0 : k == 1 ? -3 : k == 2 ? 1 : 1 + (int)rnd(80);
}

static void gen_options(KVStrategyOptions *o, const KVStrategyOptions **out) {
    memset(o, 0, sizeof *o);
    unsigned k = rnd(10);
    if (k < 2) {
        *out = NULL;
        return;
    }
    o->kv_type = (int)rnd(3);
    if (k == 9) o->offload_dir = g_dir;
    if (k == 8 && rnd(2)) o->offload_dir = "/nonexistent/pz_kv_dir";
    *out = o;
}

static void touch(const RunState *s, const Config *cfg, int seq) {
    size_t hd                                      = (size_t)(cfg->dim / cfg->n_heads);
    s->x[cfg->dim - 1]                             = 1.0f;
    s->xb[cfg->dim - 1]                            = 1.0f;
    s->xb2[cfg->dim - 1]                           = 1.0f;
    s->q[cfg->dim - 1]                             = 1.0f;
    s->hb[cfg->hidden_dim - 1]                     = 1.0f;
    s->hb2[cfg->hidden_dim - 1]                    = 1.0f;
    s->att[(size_t)cfg->n_heads * (size_t)seq - 1] = 1.0f;
    s->logits[cfg->vocab_size - 1]                 = 1.0f;
    size_t kv = (size_t)cfg->n_layers * (size_t)cfg->n_kv_heads * (size_t)seq * hd;
    if (s->key_cache) {
        s->key_cache[kv - 1]   = 1.0f;
        s->value_cache[kv - 1] = 1.0f;
    }
    if (s->kvs.key_cache_q) {
        size_t rows                                               = kv / hd;
        s->kvs.key_cache_q[rows * (size_t)s->kvs.row_bytes - 1]   = 1;
        s->kvs.value_cache_q[rows * (size_t)s->kvs.row_bytes - 1] = 1;
        s->kvs.key_scale[rows * (size_t)s->kvs.row_scales - 1]    = 1.0f;
        s->kvs.value_scale[rows * (size_t)s->kvs.row_scales - 1]  = 1.0f;
    }
}

static int g_ok, g_quant, g_off, g_f32;

static void one_case(void) {
    Config cfg;
    int seq;
    gen_shape(&cfg, &seq);
    KVStrategyOptions o;
    const KVStrategyOptions *opt;
    gen_options(&o, &opt);
    bool skip = rnd(6) == 0;
    RunState s;
    memset(&s, 0xAB, sizeof s); /* the call must reset everything */
    int rc = (int)run_state_alloc_ex(&s, &cfg, seq, skip, opt);
    mix(&rc, sizeof rc);
    if (rc != TN_OK) {
        mix(&s.x, sizeof s.x); /* a failed allocation leaves nothing behind */
        return;
    }
    touch(&s, &cfg, seq);
    g_ok++;
    g_quant += s.kvs.key_cache_q != NULL;
    g_off += s.kvs.offload_base != NULL;
    g_f32 += s.key_cache != NULL;
    int has[12] = {s.x != NULL,
                   s.att != NULL,
                   s.logits != NULL,
                   s.key_cache != NULL,
                   s.value_cache != NULL,
                   s.kvs.key_cache_q != NULL,
                   s.kvs.key_scale != NULL,
                   s.kvs.offload_base != NULL,
                   s.rope_freq != NULL,
                   s.mla.k_rope_cache != NULL,
                   s.q35.key_cache != NULL,
                   s.hb != NULL};
    mix(has, sizeof has);
    mix(&s.max_seq_len, sizeof s.max_seq_len);
    mix(&s.rope_freq_len, sizeof s.rope_freq_len);
    mix(&s.kvs.type, sizeof s.kvs.type);
    mix(&s.kvs.row_bytes, sizeof s.kvs.row_bytes);
    mix(&s.kvs.row_scales, sizeof s.kvs.row_scales);
    mix(s.rope_freq, (size_t)s.rope_freq_len * sizeof(float));
    run_state_free(&s);
}

static void test_alloc_golden(void) {
    snprintf(g_dir, sizeof g_dir, "/tmp/pz_rs_golden_XXXXXX");
    if (!mkdtemp(g_dir)) {
        TEST_ASSERT(0, "temporary directory");
        return;
    }
    for (int t = 0; t < N_CASES; t++) one_case();
    rmdir(g_dir);
    if (getenv("RUN_STATE_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d ok, %d f32, %d quantized, %d offloaded)\n",
               (unsigned long long)g_hash, g_ok, g_f32, g_quant, g_off);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "run-state layouts match the recorded hash");
}

int main(void) {
    if (!getenv("RUN_STATE_GOLDEN_VERBOSE")) freopen("/dev/null", "w", stderr);
    printf("=== run_state_alloc_ex golden ===\n");
    RUN_TEST(test_alloc_golden);
    TEST_SUMMARY();
}
