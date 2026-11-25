/*
 * tests/test_prompt_store_golden.c — prompt_store_save (and the restore that reads what it stored)
 * over 40 stores of 600 random operations each: token sequences over a four-letter alphabet (so
 * prefixes overlap constantly), lengths from 0 to the context size, tight budgets of 1..9 entries,
 * minimum lengths 0..5, context caps below the sequence length, saves of identical, shorter and
 * longer sequences, and restores of random queries. After every operation the hash covers the whole
 * store: entry count and capacity, bytes in use, the clock, and for each entry its tokens, length,
 * bytes, last-use stamp and K/V rows. Recorded from the function before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/prompt_store.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xabad492d438822d4ull
#define L 2
#define H 2
#define HD 4
#define MAXSEQ 16
#define N_STORES 40
#define N_OPS 600

typedef struct {
    Config cfg;
    RunState s;
    float k[L * H * MAXSEQ * HD];
    float v[L * H * MAXSEQ * HD];
} Fixture;

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0xa54ff53a5f1d36f1ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static void fx_init(Fixture *f, int max_seq) {
    memset(f, 0, sizeof(*f));
    f->cfg.n_layers   = L;
    f->cfg.n_heads    = H;
    f->cfg.n_kv_heads = H;
    f->cfg.dim        = H * HD;
    f->s.key_cache    = f->k;
    f->s.value_cache  = f->v;
    f->s.max_seq_len  = max_seq;
}

/* A recognisable K/V value per (tag, layer, head, position, d). */
static float val(int tag, int l, int h, int p, int d) {
    return (float)(tag * 10000 + l * 1000 + h * 100 + p * 10 + d);
}

/* K and V of every layer and head at position p. */
static void fill_position(Fixture *f, int p, int tag) {
    for (int l = 0; l < L; l++)
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < HD; d++) {
                size_t i = KV_CACHE_IDX(l, h, p, d, H, MAXSEQ, HD);
                f->k[i]  = val(tag, l, h, p, d);
                f->v[i]  = -val(tag, l, h, p, d);
            }
        }
}

static void fill(Fixture *f, PrefixCache *pc, const int *toks, int n, int tag) {
    prefix_cache_reset(pc);
    for (int p = 0; p < n; p++) {
        fill_position(f, p, tag);
        prefix_cache_set(pc, p, toks[p]);
    }
}

static void hash_store(const PromptStore *ps, size_t kv_floats_per_token) {
    mix(&ps->n, sizeof ps->n);
    mix(&ps->cap, sizeof ps->cap);
    mix(&ps->bytes_used, sizeof ps->bytes_used);
    mix(&ps->clock, sizeof ps->clock);
    for (int i = 0; i < ps->n; i++) {
        const PromptStoreEntry *e = &ps->entries[i];
        size_t kv                 = (size_t)e->len * kv_floats_per_token;
        mix(&e->len, sizeof e->len);
        mix(&e->bytes, sizeof e->bytes);
        mix(&e->last_used, sizeof e->last_used);
        mix(e->tokens, (size_t)e->len * sizeof(int));
        mix(e->k, kv * sizeof(float));
        mix(e->v, kv * sizeof(float));
    }
}

/* A sequence: a prefix of an earlier one half the time, so supersede and dedupe are common. */
static int gen_tokens(int *out, const int *prev, int prev_n) {
    int n = (int)rnd(MAXSEQ + 1);
    for (int i = 0; i < n; i++) {
        int from_prev = i < prev_n && rnd(4) != 0;
        out[i]        = from_prev ? prev[i] : (int)rnd(4);
    }
    return n;
}

static unsigned long long g_stored, g_evicted, g_hits;

static void one_store(void) {
    Fixture f;
    int max_seq = rnd(5) == 0 ? 6 + (int)rnd(10) : MAXSEQ;
    fx_init(&f, max_seq);
    PrefixCache pc;
    prefix_cache_init(&pc, MAXSEQ);
    PromptStore ps;
    size_t per_tok     = prompt_store_bytes_per_token(&f.cfg) + sizeof(int);
    size_t entries_fit = 1 + rnd(9);
    int min_tokens     = (int)rnd(6);
    prompt_store_init(&ps, per_tok * entries_fit * 8, min_tokens);
    size_t kv_floats = prompt_store_bytes_per_token(&f.cfg) / (2u * sizeof(float));
    int prev[MAXSEQ] = {0}, prev_n = 0;
    for (int op = 0; op < N_OPS; op++) {
        int toks[MAXSEQ];
        int n = gen_tokens(toks, prev, prev_n);
        if (rnd(4) == 0) {
            PrefixCache live;
            prefix_cache_init(&live, MAXSEQ);
            int match = (int)rnd(4);
            int got   = prompt_store_restore(&ps, &f.cfg, &f.s, &live, toks, n, match);
            mix(&got, sizeof got);
            mix(&live.n, sizeof live.n);
            prefix_cache_free(&live);
        } else {
            fill(&f, &pc, toks, n, op % 90 + 1);
            prompt_store_save(&ps, &f.cfg, &f.s, &pc);
        }
        memcpy(prev, toks, sizeof toks);
        prev_n = n;
        hash_store(&ps, kv_floats);
    }
    g_stored += ps.clock;
    g_evicted += ps.evictions;
    g_hits += ps.hits;
    prompt_store_free(&ps);
    prefix_cache_free(&pc);
}

static void test_prompt_store_golden(void) {
    for (int i = 0; i < N_STORES; i++) one_store();
    if (getenv("PROMPT_STORE_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (clock %llu, evictions %llu, restore hits %llu)\n",
               (unsigned long long)g_hash, g_stored, g_evicted, g_hits);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "store state matches the recorded hash");
}

int main(void) {
    printf("=== prompt_store golden ===\n");
    RUN_TEST(test_prompt_store_golden);
    TEST_SUMMARY();
}
