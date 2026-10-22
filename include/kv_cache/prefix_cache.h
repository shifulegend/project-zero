#ifndef TN_PREFIX_CACHE_H
#define TN_PREFIX_CACHE_H

/*
 * prefix_cache.h — Phase 21.5: KV-cache prefix reuse ("context caching").
 *
 * Records which token sits at each KV position of a RunState. A new prompt
 * whose leading tokens match the record can skip re-computing them: the KV
 * entries for those positions are still in the cache. With multi-turn chat
 * (the whole history is re-sent every turn) this turns per-turn prefill from
 * O(conversation) into O(new message).
 *
 * Validity rules (enforced by the caller, generate_ex()):
 *   - positions must not have wrapped the sliding window (set() past
 *     `capacity` invalidates the record);
 *   - anything else that writes the same RunState's KV (vision prefill) must
 *     call prefix_cache_reset();
 *   - recurrent-state models (Qwen3.5/3.6 linear attention) cannot resume
 *     from a partial prefix and must not use it.
 * Standalone: depends only on libc.
 */

typedef struct {
    int *tokens;   /* tokens[i] = token whose K/V is stored at position i */
    int  n;        /* valid record length */
    int  capacity; /* KV window size; positions >= capacity invalidate */
} PrefixCache;

/* Allocates room for `capacity` positions. Returns 0 on success, -1 on OOM
 * or capacity <= 0 (the cache is then inert: match() always returns 0). */
int  prefix_cache_init(PrefixCache *pc, int capacity);
void prefix_cache_free(PrefixCache *pc);
void prefix_cache_reset(PrefixCache *pc);

/* Longest common prefix between the record and tokens[0..n). */
int  prefix_cache_match(const PrefixCache *pc, const int *tokens, int n);

/* Drop everything at positions >= len. */
void prefix_cache_truncate(PrefixCache *pc, int len);

/* Record that `token` was just written at `pos` (drops any record at >= pos
 * first). pos >= capacity, or a gap (pos > n), resets the record. */
void prefix_cache_set(PrefixCache *pc, int pos, int token);

#endif /* TN_PREFIX_CACHE_H */
