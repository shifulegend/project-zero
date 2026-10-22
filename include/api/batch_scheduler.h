#ifndef TN_BATCH_SCHEDULER_H
#define TN_BATCH_SCHEDULER_H

/*
 * api/batch_scheduler — Phase 23 stage 2b (2026-10-22): serve several chat
 * requests at once (--parallel N).
 *
 * One scheduler thread owns N slots, each a RunState (own KV cache) plus its
 * own prefix cache. Every step it gathers one row from each decoding request
 * and a chunk of prompt rows from each request still reading its prompt,
 * runs them through a single transformer_forward_multi_ex() pass (weights
 * stream from memory once for all rows), and hands each request its logits.
 * Requests run the same GenSession code as generate_ex(), so prompt
 * handling, sampling, stops and stats are identical to sequential serving.
 *
 * Memory: slot KV is committed lazily. All slots together may hold at most
 * kv_budget_tokens positions; an idle slot's KV (and with it its prefix
 * cache) is released when a new request needs the room, and a request waits
 * when active requests hold it all.
 *
 * Output never blocks the batch: pieces are queued per request and written
 * to the client by the request's own thread, so a slow client only delays
 * itself. A non-zero return from that thread's callback (cancel, client
 * gone) stops the request at its next token.
 *
 * The scheduler only runs models transformer_forward_multi_supported()
 * accepts; callers check batch_scheduler_supported() first and stay
 * sequential otherwise.
 */

#include "transformer/generate.h"
#include "kv_cache/prefix_cache.h"
#include "kv_cache/prompt_store.h"

typedef struct {
    int       n_slots;           /* concurrent requests (>= 2 to be useful) */
    int       max_batch_rows;    /* rows per forward pass (decode + prefill chunk) */
    long long kv_budget_tokens;  /* positions all slots may hold together; 0 = no limit */
} BatchSchedulerOptions;

typedef struct {
    int n_slots;
    int active;      /* requests generating now */
    int pending;     /* admitted by the caller, waiting for a slot or KV room */
    long long kv_committed_tokens;
    long long kv_budget_tokens;
} BatchSchedulerStats;

typedef struct BatchScheduler BatchScheduler;

/* 1 when this model can be served by the scheduler. */
int batch_scheduler_supported(const TransformerWeights *w, const MoEConfig *mc);

/* Starts the scheduler thread. Slot 0 is slot0 / slot0_cache (shared with
 * the caller's sequential path, which may only use them while the scheduler
 * is idle); slots 1..n-1 are allocated here. pstore (may be NULL) holds
 * prompt snapshots for all slots. Returns NULL on failure (logged). */
BatchScheduler *batch_scheduler_create(const Config *cfg, const TransformerWeights *w,
                                       RunState *slot0, PrefixCache *slot0_cache,
                                       PromptStore *pstore, const MoEConfig *mc,
                                       Tokenizer *tok, ThreadPool *tp,
                                       const BatchSchedulerOptions *opt);

/* Stops the thread: active requests are cancelled, pending ones finish
 * without output, and later batch_scheduler_generate() calls return at
 * once. The memory stays valid (callers may still hold the pointer);
 * destroy frees it. Idempotent; NULL is a no-op. */
void batch_scheduler_stop(BatchScheduler *bs);

/* Stops (if needed) and frees everything, slots included. */
void batch_scheduler_destroy(BatchScheduler *bs);

/* Runs one generation through the scheduler and returns when it is
 * finished. Same contract as generate_ex(): callback runs on the calling
 * thread with each piece; gp->stats (if set) is filled. gp->prefix_cache /
 * gp->prompt_store are ignored — use_cache selects whether the slot's own
 * prefix cache and the shared snapshots may be reused. */
void batch_scheduler_generate(BatchScheduler *bs, const char *prompt,
                              const GenerationParams *gp, int use_cache,
                              TokenCallback callback, void *userdata);

/* Forget every slot's cached prefix (e.g. after a cache reset request).
 * Only call while no request is in the scheduler. */
void batch_scheduler_reset_caches(BatchScheduler *bs);

void batch_scheduler_stats(BatchScheduler *bs, BatchSchedulerStats *out);

#endif /* TN_BATCH_SCHEDULER_H */
