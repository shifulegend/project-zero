#include "kv_cache/prompt_store.h"
#include <stdlib.h>
#include <string.h>

void prompt_store_init(PromptStore *ps, size_t budget_bytes, int min_tokens) {
    if (!ps) return;
    memset(ps, 0, sizeof(*ps));
    ps->budget_bytes = budget_bytes;
    ps->min_tokens = min_tokens > 0 ? min_tokens : 1;
}

static void entry_free(PromptStoreEntry *e) {
    free(e->tokens);
    free(e->k);
    free(e->v);
    memset(e, 0, sizeof(*e));
}

static void remove_at(PromptStore *ps, int i) {
    ps->bytes_used -= ps->entries[i].bytes;
    entry_free(&ps->entries[i]);
    ps->entries[i] = ps->entries[ps->n - 1];
    ps->n--;
}

void prompt_store_clear(PromptStore *ps) {
    if (!ps) return;
    while (ps->n > 0) remove_at(ps, ps->n - 1);
}

void prompt_store_free(PromptStore *ps) {
    if (!ps) return;
    prompt_store_clear(ps);
    free(ps->entries);
    ps->entries = NULL;
    ps->cap = 0;
}

int prompt_store_supported(const MoEConfig *mc) {
    return !(mc && (mc->has_mla || mc->has_linear_attn || mc->has_qk_norm));
}

static int kv_heads(const Config *cfg) {
    return cfg->n_kv_heads > 0 ? cfg->n_kv_heads : cfg->n_heads;
}

size_t prompt_store_bytes_per_token(const Config *cfg) {
    if (!cfg) return 0;
    return (size_t)cfg->n_layers * (size_t)kv_heads(cfg) * (size_t)config_head_dim(cfg)
           * 2u * sizeof(float);
}

/* Packs positions [0, len) of every [layer][kv_head] row into a snapshot. */
static void snapshot_kv(const Config *cfg, int max_seq, const float *cache_k,
                        const float *cache_v, float *snap_k, float *snap_v, int len) {
    int nh = kv_heads(cfg), hd = config_head_dim(cfg);
    size_t row = (size_t)len * hd;
    for (int l = 0; l < cfg->n_layers; l++) {
        for (int h = 0; h < nh; h++) {
            size_t off = KV_CACHE_IDX(l, h, 0, 0, nh, max_seq, hd);
            size_t soff = ((size_t)l * nh + h) * row;
            memcpy(snap_k + soff, cache_k + off, row * sizeof(float));
            memcpy(snap_v + soff, cache_v + off, row * sizeof(float));
        }
    }
}

static int lcp(const int *a, int na, const int *b, int nb) {
    int lim = na < nb ? na : nb, i = 0;
    while (i < lim && a[i] == b[i]) i++;
    return i;
}

int prompt_store_restore(PromptStore *ps, const Config *cfg, RunState *s,
                         PrefixCache *live, const int *tokens, int n, int live_match) {
    if (!ps || ps->budget_bytes == 0 || !cfg || !s || !live || !live->tokens || !tokens)
        return live_match;
    int best = -1, best_len = live_match;
    for (int i = 0; i < ps->n; i++) {
        int m = lcp(ps->entries[i].tokens, ps->entries[i].len, tokens, n);
        if (m > best_len) { best_len = m; best = i; }
    }
    if (best < 0) { ps->misses++; return live_match; }
    if (best_len > s->max_seq_len || best_len > live->capacity) return live_match;

    PromptStoreEntry *e = &ps->entries[best];
    /* The snapshot is packed for e->len positions; restore the first best_len
     * of each [layer][head] row. */
    int nh = kv_heads(cfg), hd = config_head_dim(cfg);
    for (int l = 0; l < cfg->n_layers; l++) {
        for (int h = 0; h < nh; h++) {
            size_t off = KV_CACHE_IDX(l, h, 0, 0, nh, s->max_seq_len, hd);
            size_t soff = ((size_t)l * nh + h) * (size_t)e->len * hd;
            memcpy(s->key_cache + off, e->k + soff, (size_t)best_len * hd * sizeof(float));
            memcpy(s->value_cache + off, e->v + soff, (size_t)best_len * hd * sizeof(float));
        }
    }
    memcpy(live->tokens, e->tokens, (size_t)best_len * sizeof(int));
    live->n = best_len;
    e->last_used = ++ps->clock;
    ps->hits++;
    return best_len;
}

static int evict_lru(PromptStore *ps) {
    if (ps->n == 0) return 0;
    int victim = 0;
    for (int i = 1; i < ps->n; i++)
        if (ps->entries[i].last_used < ps->entries[victim].last_used) victim = i;
    remove_at(ps, victim);
    ps->evictions++;
    return 1;
}

void prompt_store_save(PromptStore *ps, const Config *cfg, const RunState *s,
                       const PrefixCache *live) {
    if (!ps || ps->budget_bytes == 0 || !cfg || !s || !live || !live->tokens) return;
    int len = live->n;
    if (len < ps->min_tokens || len > s->max_seq_len) return;
    size_t bytes = (size_t)len * prompt_store_bytes_per_token(cfg) + (size_t)len * sizeof(int);
    if (bytes > ps->budget_bytes) return;

    /* Drop entries this sequence supersedes (same conversation, shorter), and
     * skip the save entirely if an identical/longer entry already covers it. */
    for (int i = 0; i < ps->n;) {
        PromptStoreEntry *e = &ps->entries[i];
        int m = lcp(e->tokens, e->len, live->tokens, len);
        if (m == len && e->len >= len) { e->last_used = ++ps->clock; return; }
        if (m == e->len) { remove_at(ps, i); continue; }
        i++;
    }
    while (ps->bytes_used + bytes > ps->budget_bytes && evict_lru(ps)) {}

    if (ps->n == ps->cap) {
        int nc = ps->cap ? ps->cap * 2 : 8;
        PromptStoreEntry *ne = (PromptStoreEntry *)realloc(ps->entries, (size_t)nc * sizeof(*ne));
        if (!ne) return;
        ps->entries = ne;
        ps->cap = nc;
    }
    PromptStoreEntry e;
    memset(&e, 0, sizeof(e));
    size_t kv_floats = (size_t)len * (prompt_store_bytes_per_token(cfg) / (2u * sizeof(float)));
    e.tokens = (int *)malloc((size_t)len * sizeof(int));
    e.k = (float *)malloc(kv_floats * sizeof(float));
    e.v = (float *)malloc(kv_floats * sizeof(float));
    if (!e.tokens || !e.k || !e.v) { entry_free(&e); return; }
    memcpy(e.tokens, live->tokens, (size_t)len * sizeof(int));
    snapshot_kv(cfg, s->max_seq_len, s->key_cache, s->value_cache, e.k, e.v, len);
    e.len = len;
    e.bytes = bytes;
    e.last_used = ++ps->clock;
    ps->entries[ps->n++] = e;
    ps->bytes_used += bytes;
}
