#ifndef TN_BATCH_SCHEDULER_INTERNAL_H
#define TN_BATCH_SCHEDULER_INTERNAL_H

/*
 * api/batch_scheduler_internal.h — the scheduler's private state, shared by batch_scheduler.c
 * (the scheduling loop and public API) and batch_slots.c (set-up and tear-down). Not part of the
 * public API: include api/batch_scheduler.h instead.
 */

#include <pthread.h>

#include "api/batch_scheduler.h"
#include "runtime/run_state.h"
#include "runtime/spec_scratch.h"
#include "transformer/gen_session.h"

/* First capacity (bytes) of a job's output buffer; it doubles from here. */
#define BATCH_JOB_OUT_INIT_CAP 1024

typedef struct BatchJob {
    const char *prompt;
    const GenerationParams *gp;
    int use_cache;
    GenSession g;
    struct BatchScheduler *sched;
    /* Output channel, guarded by sched->mu: NUL-separated pieces. */
    char *out;
    size_t out_len, out_cap;
    int out_pieces;
    int done;
    int abort;
    pthread_cond_t cv;
    struct BatchJob *next;
} BatchJob;

typedef struct {
    RunState *s;
    int owned; /* allocated by the scheduler (slots >= 1) */
    PrefixCache *pc;
    PrefixCache own_pc;
    BatchJob *job; /* NULL when idle */
    unsigned long long admit_seq;
} BatchSlot;

struct BatchScheduler {
    const Config *cfg;
    const TransformerWeights *w;
    const MoEConfig *mc;
    Tokenizer *tok;
    ThreadPool *tp;
    PromptStore *pstore;
    BatchSchedulerOptions opt;

    BatchSlot *slots;
    SpecBatchScratch sb;
    int *row_tok, *row_pos;
    RunState **row_state;
    unsigned char *row_want;
    int *order, *count; /* per step: slots in row order, rows per slot */
    float *logits;      /* [n_slots][vocab]: at most one wanted row per slot */

    pthread_mutex_t mu;
    pthread_cond_t cv; /* new work / stop */
    BatchJob *pend_head, *pend_tail;
    int n_pending, n_active;
    long long kv_committed; /* snapshot for stats, updated by the scheduler under mu */
    unsigned long long seq;
    int stopping;
    pthread_t thread;
    int thread_started;
};

/* Allocate the row/slot buffers, set up every slot (slot 0 is the caller's run state and cache) and
 * start `main` on a new thread. Returns 0 on failure; the caller then calls
 * batch_scheduler_release(). */
int batch_scheduler_setup(BatchScheduler *sched, const Config *cfg, RunState *slot0,
                          PrefixCache *slot0_cache, void *(*main)(void *));

/* Free the slots (those the scheduler owns), scratch and row buffers. */
void batch_scheduler_release(BatchScheduler *sched);

#endif /* TN_BATCH_SCHEDULER_INTERNAL_H */
