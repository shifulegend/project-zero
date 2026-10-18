#ifndef TN_PROMPT_STORE_H
#define TN_PROMPT_STORE_H

/*
 * prompt_store.h — Phase 26: multi-entry prompt cache with LRU eviction.
 *
 * Phase 21.5's PrefixCache remembers only what is *currently* in the KV
 * cache, so alternating between two conversations (two browser tabs, two
 * API clients) recomputes each history from scratch. PromptStore keeps
 * snapshots of the K/V rows for several token sequences inside a byte
 * budget; on a new prompt the best-matching snapshot is copied back into
 * the RunState and the live PrefixCache is pointed at it, so only the new
 * suffix is computed.
 *
 * Lookup is a longest-common-prefix scan over entries — at the entry counts
 * a byte budget allows, that is negligible next to one forward pass, and it
 * avoids a radix tree's node-splitting bookkeeping. A sequence that extends
 * an existing entry (the same conversation, one turn later) replaces it.
 *
 * Scope: the generic dense/GQA KV cache (RunState.key_cache/value_cache,
 * layout KV_CACHE_IDX). MLA, Qwen3-MoE and Qwen3.5 hybrid models keep their
 * own caches — prompt_store_supported() returns 0 for them.
 */

#include <stddef.h>
#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "kv_cache/prefix_cache.h"

typedef struct {
    int     *tokens;
    int      len;
    float   *k;          /* [n_layers][n_kv_heads][len][head_dim] */
    float   *v;
    size_t   bytes;
    unsigned long long last_used;
} PromptStoreEntry;

typedef struct {
    PromptStoreEntry *entries;
    int      n, cap;
    size_t   bytes_used;
    size_t   budget_bytes;   /* 0 = store disabled */
    int      min_tokens;     /* sequences shorter than this are not stored */
    unsigned long long clock;
    /* stats */
    unsigned long long hits, misses, evictions;
} PromptStore;

/* budget_bytes 0 disables the store (every call becomes a no-op). */
void prompt_store_init(PromptStore *ps, size_t budget_bytes, int min_tokens);
void prompt_store_free(PromptStore *ps);
void prompt_store_clear(PromptStore *ps);

/* 1 if this model's KV lives in the generic key_cache/value_cache. */
int prompt_store_supported(const MoEConfig *mc);

/* Bytes one cached token costs (K+V, all layers). */
size_t prompt_store_bytes_per_token(const Config *cfg);

/* If an entry shares a longer prefix with tokens[0..n) than `live_match`
 * (the live PrefixCache's match), copy its first `lcp` positions into the
 * RunState, rewrite `live` to those tokens and return lcp; else return
 * live_match unchanged. Positions must fit in s->max_seq_len. */
int prompt_store_restore(PromptStore *ps, const Config *cfg, RunState *s,
                         PrefixCache *live, const int *tokens, int n, int live_match);

/* Snapshot the live sequence (live->tokens[0..live->n)) from the RunState.
 * Replaces entries that are a prefix of it; evicts LRU entries to fit. */
void prompt_store_save(PromptStore *ps, const Config *cfg, const RunState *s,
                       const PrefixCache *live);

#endif /* TN_PROMPT_STORE_H */
