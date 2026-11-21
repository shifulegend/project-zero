/*
 * api/batch_slots.c — set-up and tear-down of the batch scheduler's slots and work buffers.
 * See api/batch_scheduler_internal.h.
 */

#include <stdlib.h>

#include "api/batch_scheduler_internal.h"
#include "runtime/run_state.h"

static void free_slots(BatchScheduler *sched) {
    if (!sched->slots) return;
    for (int i = 0; i < sched->opt.n_slots; i++) {
        BatchSlot *slot = &sched->slots[i];
        if (!slot->owned) continue;
        if (slot->s) {
            run_state_free(slot->s);
            free(slot->s);
        }
        prefix_cache_free(&slot->own_pc);
    }
    free(sched->slots);
    sched->slots = NULL;
}

/* The per-row and per-slot work buffers. Returns 0 when any allocation failed (the caller frees
 * whatever was obtained through release_buffers). */
static int alloc_buffers(BatchScheduler *sched, const Config *cfg) {
    int n = sched->opt.n_slots, rows = sched->opt.max_batch_rows;
    sched->slots     = (BatchSlot *)calloc((size_t)n, sizeof(BatchSlot));
    sched->row_tok   = (int *)malloc((size_t)rows * sizeof(int));
    sched->row_pos   = (int *)malloc((size_t)rows * sizeof(int));
    sched->row_state = (RunState **)malloc((size_t)rows * sizeof(RunState *));
    sched->row_want  = (unsigned char *)malloc((size_t)rows);
    sched->order     = (int *)malloc((size_t)n * sizeof(int));
    sched->count     = (int *)malloc((size_t)n * sizeof(int));
    sched->logits    = (float *)malloc((size_t)n * (size_t)cfg->vocab_size * sizeof(float));
    return sched->slots && sched->row_tok && sched->row_pos && sched->row_state &&
           sched->row_want && sched->order && sched->count && sched->logits &&
           spec_batch_scratch_alloc(&sched->sb, cfg, rows) == TN_OK;
}

static void release_buffers(BatchScheduler *sched) {
    free_slots(sched);
    spec_batch_scratch_free(&sched->sb);
    free(sched->row_tok);
    free(sched->row_pos);
    free(sched->row_state);
    free(sched->row_want);
    free(sched->order);
    free(sched->count);
    free(sched->logits);
}

/* Slot i > 0: its own run state and prefix cache, sharing slot 0's adapter and attention kernel.
 * Returns 0 on allocation failure (slot->s stays NULL or owned for free_slots). */
static int init_owned_slot(BatchSlot *slot, const Config *cfg, const RunState *slot0,
                           const BatchSchedulerOptions *opt) {
    slot->owned = 1;
    slot->s     = (RunState *)malloc(sizeof(RunState));
    if (!slot->s) return 0;
    if (run_state_alloc_ex(slot->s, cfg, slot0->max_seq_len, false, opt->kv) != TN_OK) {
        free(slot->s);
        slot->s = NULL;
        return 0;
    }
    slot->s->active_lora = slot0->active_lora; /* one adapter for every sequence */
    slot->s->attn        = slot0->attn;        /* and one attention kernel */
    prefix_cache_init(&slot->own_pc, slot->s->max_seq_len);
    slot->pc = &slot->own_pc;
    return 1;
}

static int init_slots(BatchScheduler *sched, const Config *cfg, RunState *slot0,
                      PrefixCache *slot0_cache) {
    if (!slot0_cache->tokens) prefix_cache_init(slot0_cache, slot0->max_seq_len);
    for (int i = 0; i < sched->opt.n_slots; i++) {
        BatchSlot *slot = &sched->slots[i];
        if (i == 0) {
            slot->s  = slot0;
            slot->pc = slot0_cache;
        } else if (!init_owned_slot(slot, cfg, slot0, &sched->opt)) {
            return 0;
        }
    }
    return 1;
}

static int start_thread(BatchScheduler *sched, void *(*main)(void *)) {
    pthread_mutex_init(&sched->mu, NULL);
    pthread_cond_init(&sched->cv, NULL);
    int ok                = pthread_create(&sched->thread, NULL, main, sched) == 0;
    sched->thread_started = ok;
    if (!ok) {
        pthread_cond_destroy(&sched->cv);
        pthread_mutex_destroy(&sched->mu);
    }
    return ok;
}

int batch_scheduler_setup(BatchScheduler *sched, const Config *cfg, RunState *slot0,
                          PrefixCache *slot0_cache, void *(*main)(void *)) {
    return alloc_buffers(sched, cfg) && init_slots(sched, cfg, slot0, slot0_cache) &&
           start_thread(sched, main);
}

void batch_scheduler_release(BatchScheduler *sched) {
    release_buffers(sched);
}
