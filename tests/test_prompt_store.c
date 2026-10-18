/*
 * tests/test_prompt_store.c — Phase 26 multi-entry prompt snapshots: save,
 * restore into the KV cache, supersede, LRU eviction, budget, scope.
 */
#include "test_harness.h"
#include "kv_cache/prompt_store.h"

#define L 2
#define H 2
#define HD 4
#define MAXSEQ 16

typedef struct { Config cfg; RunState s; float k[L * H * MAXSEQ * HD]; float v[L * H * MAXSEQ * HD]; } Fixture;

static void fx_init(Fixture *f) {
    memset(f, 0, sizeof(*f));
    f->cfg.n_layers = L; f->cfg.n_heads = H; f->cfg.n_kv_heads = H; f->cfg.dim = H * HD;
    f->s.key_cache = f->k; f->s.value_cache = f->v; f->s.max_seq_len = MAXSEQ;
}

/* Writes a recognisable value for (layer, head, pos, d) tagged by `tag`. */
static float val(int tag, int l, int h, int p, int d) { return (float)(tag * 10000 + l * 1000 + h * 100 + p * 10 + d); }

static void fill(Fixture *f, PrefixCache *pc, const int *toks, int n, int tag) {
    prefix_cache_reset(pc);
    for (int p = 0; p < n; p++) {
        for (int l = 0; l < L; l++) for (int h = 0; h < H; h++) for (int d = 0; d < HD; d++) {
            size_t i = KV_CACHE_IDX(l, h, p, d, H, MAXSEQ, HD);
            f->k[i] = val(tag, l, h, p, d); f->v[i] = -val(tag, l, h, p, d);
        }
        prefix_cache_set(pc, p, toks[p]);
    }
}

static int kv_matches(const Fixture *f, int n, int tag) {
    for (int p = 0; p < n; p++)
        for (int l = 0; l < L; l++) for (int h = 0; h < H; h++) for (int d = 0; d < HD; d++) {
            size_t i = KV_CACHE_IDX(l, h, p, d, H, MAXSEQ, HD);
            if (f->k[i] != val(tag, l, h, p, d) || f->v[i] != -val(tag, l, h, p, d)) return 0;
        }
    return 1;
}

static void test_save_restore_roundtrip(void) {
    Fixture f; fx_init(&f);
    PrefixCache pc; prefix_cache_init(&pc, MAXSEQ);
    PromptStore ps; prompt_store_init(&ps, 1 << 20, 2);
    int A[] = {1, 2, 3, 4, 5, 6}, B[] = {1, 9, 9, 9, 9};
    fill(&f, &pc, A, 6, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    fill(&f, &pc, B, 5, 2); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 2, "two entries");

    /* Live KV holds B; a prompt continuing A must restore A's rows. */
    int q[] = {1, 2, 3, 4, 5, 6, 7};
    int live = prefix_cache_match(&pc, q, 7);
    TEST_ASSERT_EQ(live, 1, "live cache only shares token 1");
    int got = prompt_store_restore(&ps, &f.cfg, &f.s, &pc, q, 7, live);
    TEST_ASSERT_EQ(got, 6, "restored A's 6 positions");
    TEST_ASSERT(kv_matches(&f, 6, 1), "KV rows are A's");
    TEST_ASSERT_EQ(prefix_cache_match(&pc, q, 7), 6, "live record now A");
    TEST_ASSERT_EQ((int)ps.hits, 1, "hit counted");

    /* No entry better than live -> unchanged. */
    TEST_ASSERT_EQ(prompt_store_restore(&ps, &f.cfg, &f.s, &pc, q, 7, 6), 6, "no better entry");
    prompt_store_free(&ps); prefix_cache_free(&pc);
}

static void test_supersede_and_dedupe(void) {
    Fixture f; fx_init(&f);
    PrefixCache pc; prefix_cache_init(&pc, MAXSEQ);
    PromptStore ps; prompt_store_init(&ps, 1 << 20, 2);
    int A[] = {1, 2, 3, 4, 5, 6, 7, 8};
    fill(&f, &pc, A, 4, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    fill(&f, &pc, A, 8, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 1, "longer continuation replaces the shorter entry");
    TEST_ASSERT_EQ(ps.entries[0].len, 8, "kept the longer one");
    fill(&f, &pc, A, 5, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 1, "prefix of an existing entry is not stored again");
    prompt_store_free(&ps); prefix_cache_free(&pc);
}

static void test_budget_and_lru(void) {
    Fixture f; fx_init(&f);
    PrefixCache pc; prefix_cache_init(&pc, MAXSEQ);
    size_t per = 4 * prompt_store_bytes_per_token(&f.cfg) + 4 * sizeof(int);
    PromptStore ps; prompt_store_init(&ps, per * 2, 2);   /* room for two 4-token entries */
    int A[] = {1, 1, 1, 1}, B[] = {2, 2, 2, 2}, C[] = {3, 3, 3, 3};
    fill(&f, &pc, A, 4, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    fill(&f, &pc, B, 4, 2); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    /* touch A so B is LRU */
    int qa[] = {1, 1, 1, 1, 9};
    prefix_cache_reset(&pc);
    prompt_store_restore(&ps, &f.cfg, &f.s, &pc, qa, 5, 0);
    fill(&f, &pc, C, 4, 3); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 2, "budget holds two");
    TEST_ASSERT_EQ((int)ps.evictions, 1, "one eviction");
    int has_b = 0;
    for (int i = 0; i < ps.n; i++) has_b |= ps.entries[i].tokens[0] == 2;
    TEST_ASSERT(!has_b, "least-recently-used entry (B) evicted");
    TEST_ASSERT(ps.bytes_used <= ps.budget_bytes, "within budget");

    int D[] = {4};
    fill(&f, &pc, D, 1, 4); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 2, "below min_tokens is not stored");
    prompt_store_clear(&ps);
    TEST_ASSERT(ps.n == 0 && ps.bytes_used == 0, "clear empties the store");
    prompt_store_free(&ps); prefix_cache_free(&pc);
}

static void test_disabled_and_scope(void) {
    Fixture f; fx_init(&f);
    PrefixCache pc; prefix_cache_init(&pc, MAXSEQ);
    PromptStore ps; prompt_store_init(&ps, 0, 1);
    int A[] = {1, 2, 3};
    fill(&f, &pc, A, 3, 1); prompt_store_save(&ps, &f.cfg, &f.s, &pc);
    TEST_ASSERT_EQ(ps.n, 0, "budget 0 disables");
    MoEConfig mc; memset(&mc, 0, sizeof(mc));
    TEST_ASSERT(prompt_store_supported(&mc), "dense supported");
    mc.has_mla = 1;
    TEST_ASSERT(!prompt_store_supported(&mc), "MLA excluded");
    TEST_ASSERT(prompt_store_supported(NULL), "NULL mc = dense");
    prompt_store_free(&ps); prefix_cache_free(&pc);
}

int main(void) {
    printf("=== Prompt Store (Phase 26) Tests ===\n");
    RUN_TEST(test_save_restore_roundtrip);
    RUN_TEST(test_supersede_and_dedupe);
    RUN_TEST(test_budget_and_lru);
    RUN_TEST(test_disabled_and_scope);
    TEST_SUMMARY();
}
