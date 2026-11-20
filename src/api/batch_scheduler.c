/*
 * Phase 23 stage 2b — src/api/batch_scheduler.c
 * See include/api/batch_scheduler.h for responsibility.
 */

#include "api/batch_scheduler.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/log.h"
#include "runtime/run_state.h"
#include "runtime/spec_scratch.h"
#include "transformer/forward.h"
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

int batch_scheduler_supported(const TransformerWeights *w, const MoEConfig *mc) {
    return transformer_forward_multi_supported(w, mc);
}

/* ── Output channel ───────────────────────────────────────────────────── */

/* GenSession callback, on the scheduler thread: queue the piece for the
 * request's own thread. Returns non-zero once that thread asked to stop. */
static int job_enqueue(const char *piece, void *userdata) {
    BatchJob *job         = (BatchJob *)userdata;
    BatchScheduler *sched = job->sched;
    size_t n              = piece ? strlen(piece) : 0;
    pthread_mutex_lock(&sched->mu);
    int stop = job->abort;
    if (!stop && piece) {
        if (job->out_len + n + 1 > job->out_cap) {
            size_t cap = job->out_cap ? job->out_cap * 2 : BATCH_JOB_OUT_INIT_CAP;
            while (cap < job->out_len + n + 1) cap *= 2;
            char *nb = (char *)realloc(job->out, cap);
            if (!nb) {
                job->abort = stop = 1;
            } else {
                job->out     = nb;
                job->out_cap = cap;
            }
        }
        if (!stop) {
            memcpy(job->out + job->out_len, piece, n + 1);
            job->out_len += n + 1;
            job->out_pieces++;
            pthread_cond_signal(&job->cv);
        }
    }
    pthread_mutex_unlock(&sched->mu);
    return stop;
}

static void job_finish_locked(BatchJob *job) {
    job->done = 1;
    pthread_cond_signal(&job->cv);
}

/* ── Slots and KV budget ──────────────────────────────────────────────── */

static long long kv_need(const GenSession *g, const Config *cfg) {
    long long need =
        (long long)g->prompt.n_prompt + (g->req.max_tokens > 0 ? g->req.max_tokens : 0);
    return need < cfg->seq_len ? need : cfg->seq_len;
}

static long long kv_committed_total(const BatchScheduler *sched) {
    long long t = 0;
    for (int i = 0; i < sched->opt.n_slots; i++) t += sched->slots[i].s->kv_committed;
    return t;
}

static void slot_release_kv(BatchScheduler *sched, BatchSlot *slot) {
    prefix_cache_reset(slot->pc); /* the cached prefix is about to vanish */
    run_state_kv_release(slot->s, sched->cfg);
}

/* Free slot best suited to the job: the longest reusable prefix, else the
 * least memory already committed (keeps other slots' caches warm). */
static int pick_slot(BatchScheduler *sched, const BatchJob *job, int *match_out) {
    int best = -1, best_match = -1;
    long long best_commit = 0;
    for (int i = 0; i < sched->opt.n_slots; i++) {
        BatchSlot *slot = &sched->slots[i];
        if (slot->job) continue;
        int m       = job->use_cache ? prefix_cache_match(slot->pc, job->g.prompt.prompt_tokens,
                                                          job->g.prompt.n_prompt)
                                     : 0;
        long long c = slot->s->kv_committed;
        if (best < 0 || m > best_match || (m == best_match && c < best_commit)) {
            best        = i;
            best_match  = m;
            best_commit = c;
        }
    }
    *match_out = best_match;
    return best;
}

/* Makes room for job on slot `idx`; returns 1 when it may start now. */
static int kv_make_room(BatchScheduler *sched, int idx, int match, const BatchJob *job) {
    long long budget = sched->opt.kv_budget_tokens;
    if (budget <= 0) return 1;
    long long need = kv_need(&job->g, sched->cfg);
    for (;;) {
        BatchSlot *own  = &sched->slots[idx];
        long long own_c = own->s->kv_committed;
        long long after = kv_committed_total(sched) - own_c + (own_c > need ? own_c : need);
        if (after <= budget) return 1;
        /* Release the idle slot holding the most memory. */
        int victim = -1;
        for (int i = 0; i < sched->opt.n_slots; i++) {
            BatchSlot *slot = &sched->slots[i];
            if (i == idx || slot->job || slot->s->kv_committed <= 0) continue;
            if (victim < 0 || slot->s->kv_committed > sched->slots[victim].s->kv_committed)
                victim = i;
        }
        if (victim >= 0) {
            slot_release_kv(sched, &sched->slots[victim]);
            continue;
        }
        if (own_c > need && match == 0 && run_state_kv_release(own->s, sched->cfg) > 0) {
            prefix_cache_reset(own->pc);
            continue;
        }
        /* Active requests hold the rest: wait for one to finish, unless
         * none is running (then nothing would ever free it). */
        return sched->n_active == 0;
    }
}

/* ── Scheduler thread ─────────────────────────────────────────────────── */

static void start_job(BatchScheduler *sched, int idx, BatchJob *job) {
    BatchSlot *slot      = &sched->slots[idx];
    slot->s->current_pos = 0;                          /* every request is a fresh conversation */
    if (!job->use_cache) prefix_cache_reset(slot->pc); /* KV is about to be overwritten */
    gen_session_start(&job->g, slot->s, job->use_cache ? slot->pc : NULL,
                      job->use_cache ? sched->pstore : NULL);
    slot->job       = job;
    slot->admit_seq = ++sched->seq;
}

/* Called with sched->mu held; may drop it while preparing prompts. */
static void admit(BatchScheduler *sched, BatchJob **staged) {
    for (;;) {
        int free_slots = 0;
        for (int i = 0; i < sched->opt.n_slots; i++)
            if (!sched->slots[i].job) free_slots++;
        if (!free_slots) return;
        if (!*staged) {
            BatchJob *j = sched->pend_head;
            if (!j) return;
            sched->pend_head = j->next;
            if (!sched->pend_head) sched->pend_tail = NULL;
            sched->n_pending--;
            j->next = NULL;
            if (j->abort || sched->stopping) { /* gone before it started */
                job_finish_locked(j);
                continue;
            }
            pthread_mutex_unlock(&sched->mu);
            int ok = gen_session_prepare(&j->g, sched->cfg, sched->w, sched->mc, sched->tok,
                                         sched->tp, j->prompt, j->gp, job_enqueue, j) == 0;
            if (!ok) {
                gen_session_end(&j->g);
                pthread_mutex_lock(&sched->mu);
                job_finish_locked(j);
                continue;
            }
            pthread_mutex_lock(&sched->mu);
            *staged = j;
            sched->n_pending++; /* still waiting, from the caller's view */
        }
        int match = 0;
        int idx   = pick_slot(sched, *staged, &match);
        if (!kv_make_room(sched, idx, match, *staged)) return; /* wait for a finish */
        BatchJob *j = *staged;
        *staged     = NULL;
        sched->n_pending--;
        sched->n_active++;
        pthread_mutex_unlock(&sched->mu);
        start_job(sched, idx, j);
        pthread_mutex_lock(&sched->mu);
        sched->kv_committed = kv_committed_total(sched);
    }
}

/* One batched forward pass over every active request. */
static void step(BatchScheduler *sched) {
    int n_slots = sched->opt.n_slots, cap = sched->opt.max_batch_rows;
    int *order = sched->order, *count = sched->count;
    int n_order = 0, rows = 0;
    memset(count, 0, (size_t)n_slots * sizeof(int));

    /* Decode rows first (one each), so generation never waits behind a
     * long prompt; prefill chunks take what is left, oldest request first. */
    for (int pass = 0; pass < 2 && rows < cap; pass++) {
        for (;;) {
            int pick = -1;
            for (int i = 0; i < n_slots; i++) {
                BatchSlot *slot = &sched->slots[i];
                if (!slot->job || count[i] || slot->job->g.progress.done) continue;
                if (gen_session_needs_logits(&slot->job->g) != (pass == 0)) continue;
                if (pick < 0 || slot->admit_seq < sched->slots[pick].admit_seq) pick = i;
            }
            if (pick < 0 || rows >= cap) break;
            const GenSession *g = &sched->slots[pick].job->g;
            int n =
                gen_session_input_rows(g, sched->row_tok + rows, sched->row_pos + rows, cap - rows);
            if (n <= 0) {
                count[pick] = -1;
                continue;
            }
            for (int k = 0; k < n; k++) {
                sched->row_state[rows + k] = sched->slots[pick].s;
                sched->row_want[rows + k]  = (g->progress.step + k >= g->prompt.n_prompt - 1);
            }
            count[pick]      = n;
            order[n_order++] = pick;
            rows += n;
        }
    }
    if (rows == 0) return;

    /* One row (a lone request decoding): the single-sequence forward is
     * ~5% faster than a one-row batch and is exactly what generate_ex runs. */
    TernaryError err = TN_OK;
    float *single    = NULL;
    if (rows == 1)
        single = transformer_forward(sched->row_tok[0], sched->row_pos[0], sched->cfg, sched->w,
                                     sched->row_state[0], sched->mc, sched->tp);
    else
        err = transformer_forward_multi_ex(sched->row_tok, sched->row_pos, sched->row_state, rows,
                                           sched->row_want, sched->cfg, sched->w, &sched->sb,
                                           sched->mc, sched->tp, sched->logits);
    if (err != TN_OK)
        TN_LOG_ERR("parallel", "[parallel] batched forward failed: %s\n", tn_error_str(err));
    int r = 0, out = 0;
    for (int o = 0; o < n_order; o++) {
        GenSession *g = &sched->slots[order[o]].job->g;
        for (int k = 0; k < count[order[o]]; k++, r++) {
            float *lg = !sched->row_want[r] ? NULL
                        : single            ? single
                                 : sched->logits + (size_t)(out++) * sched->cfg->vocab_size;
            if (err != TN_OK) {
                g->progress.finish = GEN_FINISH_ERROR;
                g->progress.done   = 1;
            } else if (!g->progress.done)
                gen_session_consume(g, lg);
        }
    }
}

static void finish_done(BatchScheduler *sched) {
    for (int i = 0; i < sched->opt.n_slots; i++) {
        BatchSlot *slot = &sched->slots[i];
        if (!slot->job || !slot->job->g.progress.done) continue;
        BatchJob *j = slot->job;
        gen_session_end(&j->g); /* flush, snapshot, stats */
        pthread_mutex_lock(&sched->mu);
        slot->job = NULL;
        sched->n_active--;
        /* Last look at slot memory for this request: once it is marked done
         * its caller may hand the engine (slot 0 included) to the
         * sequential path, so an idle scheduler never reads slot state. */
        sched->kv_committed = kv_committed_total(sched);
        job_finish_locked(j);
        pthread_mutex_unlock(&sched->mu);
    }
}

static void *scheduler_main(void *arg) {
    BatchScheduler *sched = (BatchScheduler *)arg;
    BatchJob *staged      = NULL;
    pthread_mutex_lock(&sched->mu);
    for (;;) {
        if (sched->stopping) {
            for (int i = 0; i < sched->opt.n_slots; i++)
                if (sched->slots[i].job) sched->slots[i].job->abort = 1;
            if (staged) { /* prepared but never started */
                BatchJob *j = staged;
                staged      = NULL;
                sched->n_pending--;
                pthread_mutex_unlock(&sched->mu);
                gen_session_end(&j->g);
                pthread_mutex_lock(&sched->mu);
                job_finish_locked(j);
            }
        }
        /* A request whose client stopped listening ends now, even mid-prompt. */
        for (int i = 0; i < sched->opt.n_slots; i++) {
            BatchJob *j = sched->slots[i].job;
            if (j && j->abort && !j->g.progress.done) {
                j->g.progress.finish = GEN_FINISH_CANCELLED;
                j->g.progress.done   = 1;
            }
        }
        admit(sched, &staged);
        if (sched->n_active == 0) {
            if (sched->stopping && !sched->pend_head && !staged) break;
            pthread_cond_wait(&sched->cv, &sched->mu);
            continue;
        }
        int any_done = 0;
        for (int i = 0; i < sched->opt.n_slots; i++)
            if (sched->slots[i].job && sched->slots[i].job->g.progress.done) any_done = 1;
        pthread_mutex_unlock(&sched->mu);
        if (!any_done) step(sched);
        finish_done(sched);
        pthread_mutex_lock(&sched->mu);
    }
    pthread_mutex_unlock(&sched->mu);
    return NULL;
}

/* ── Public API ───────────────────────────────────────────────────────── */

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

BatchScheduler *batch_scheduler_create(const Config *cfg, const TransformerWeights *w,
                                       RunState *slot0, PrefixCache *slot0_cache,
                                       PromptStore *pstore, const MoEConfig *mc, Tokenizer *tok,
                                       ThreadPool *tp, const BatchSchedulerOptions *opt) {
    if (!cfg || !w || !slot0 || !slot0_cache || !tok || !opt || opt->n_slots < 1 ||
        opt->max_batch_rows < opt->n_slots || !batch_scheduler_supported(w, mc))
        return NULL;
    BatchScheduler *sched = (BatchScheduler *)calloc(1, sizeof(*sched));
    if (!sched) return NULL;
    sched->cfg    = cfg;
    sched->w      = w;
    sched->mc     = mc;
    sched->tok    = tok;
    sched->tp     = tp;
    sched->pstore = pstore;
    sched->opt    = *opt;
    int n = opt->n_slots, rows = opt->max_batch_rows;
    sched->slots     = (BatchSlot *)calloc((size_t)n, sizeof(BatchSlot));
    sched->row_tok   = (int *)malloc((size_t)rows * sizeof(int));
    sched->row_pos   = (int *)malloc((size_t)rows * sizeof(int));
    sched->row_state = (RunState **)malloc((size_t)rows * sizeof(RunState *));
    sched->row_want  = (unsigned char *)malloc((size_t)rows);
    sched->order     = (int *)malloc((size_t)n * sizeof(int));
    sched->count     = (int *)malloc((size_t)n * sizeof(int));
    sched->logits    = (float *)malloc((size_t)n * (size_t)cfg->vocab_size * sizeof(float));
    int ok           = sched->slots && sched->row_tok && sched->row_pos && sched->row_state &&
             sched->row_want && sched->order && sched->count && sched->logits &&
             spec_batch_scratch_alloc(&sched->sb, cfg, rows) == TN_OK;
    if (ok && !slot0_cache->tokens) prefix_cache_init(slot0_cache, slot0->max_seq_len);
    for (int i = 0; ok && i < n; i++) {
        BatchSlot *slot = &sched->slots[i];
        if (i == 0) {
            slot->s  = slot0;
            slot->pc = slot0_cache;
            continue;
        }
        slot->owned = 1;
        slot->s     = (RunState *)malloc(sizeof(RunState));
        if (!slot->s) {
            ok = 0;
            break;
        }
        memset(slot->s, 0, sizeof(*slot->s));
        if (run_state_alloc(slot->s, cfg, slot0->max_seq_len) != TN_OK) {
            free(slot->s);
            slot->s = NULL;
            ok      = 0;
            break;
        }
        slot->s->active_lora = slot0->active_lora; /* one adapter for every sequence */
        prefix_cache_init(&slot->own_pc, slot->s->max_seq_len);
        slot->pc = &slot->own_pc;
    }
    if (ok) {
        pthread_mutex_init(&sched->mu, NULL);
        pthread_cond_init(&sched->cv, NULL);
        ok                    = pthread_create(&sched->thread, NULL, scheduler_main, sched) == 0;
        sched->thread_started = ok;
        if (!ok) {
            pthread_cond_destroy(&sched->cv);
            pthread_mutex_destroy(&sched->mu);
        }
    }
    if (!ok) {
        TN_LOG_ERR("parallel", "[parallel] could not set up %d slots (out of memory?)\n", n);
        free_slots(sched);
        spec_batch_scratch_free(&sched->sb);
        free(sched->row_tok);
        free(sched->row_pos);
        free(sched->row_state);
        free(sched->row_want);
        free(sched->order);
        free(sched->count);
        free(sched->logits);
        free(sched);
        return NULL;
    }
    return sched;
}

void batch_scheduler_stop(BatchScheduler *sched) {
    if (!sched || !sched->thread_started) return;
    pthread_mutex_lock(&sched->mu);
    sched->stopping = 1;
    pthread_cond_broadcast(&sched->cv);
    pthread_mutex_unlock(&sched->mu);
    pthread_join(sched->thread, NULL);
    sched->thread_started = 0;
}

void batch_scheduler_destroy(BatchScheduler *sched) {
    if (!sched) return;
    batch_scheduler_stop(sched);
    pthread_cond_destroy(&sched->cv);
    pthread_mutex_destroy(&sched->mu);
    free_slots(sched);
    spec_batch_scratch_free(&sched->sb);
    free(sched->row_tok);
    free(sched->row_pos);
    free(sched->row_state);
    free(sched->row_want);
    free(sched->order);
    free(sched->count);
    free(sched->logits);
    free(sched);
}

void batch_scheduler_generate(BatchScheduler *sched, const char *prompt, const GenerationParams *gp,
                              int use_cache, TokenCallback callback, void *userdata) {
    BatchJob job;
    memset(&job, 0, sizeof(job));
    job.prompt    = prompt;
    job.gp        = gp;
    job.use_cache = use_cache;
    job.sched     = sched;
    pthread_cond_init(&job.cv, NULL);
    if (gp && gp->stats) {
        memset(gp->stats, 0, sizeof(*gp->stats));
        gp->stats->finish_reason = GEN_FINISH_CANCELLED; /* until the session reports */
    }

    pthread_mutex_lock(&sched->mu);
    if (sched->stopping) {
        pthread_mutex_unlock(&sched->mu);
        pthread_cond_destroy(&job.cv);
        return;
    }
    if (sched->pend_tail)
        sched->pend_tail->next = &job;
    else
        sched->pend_head = &job;
    sched->pend_tail = &job;
    sched->n_pending++;
    pthread_cond_signal(&sched->cv);

    /* Deliver queued pieces on this thread until the request is finished. */
    char *buf   = NULL;
    size_t cap  = 0;
    int stopped = 0;
    for (;;) {
        while (!job.out_pieces && !job.done) pthread_cond_wait(&job.cv, &sched->mu);
        int pieces = job.out_pieces, done = job.done;
        char *tmp      = buf;
        buf            = job.out;
        job.out        = tmp;
        size_t tcap    = cap;
        cap            = job.out_cap;
        job.out_cap    = tcap;
        job.out_len    = 0;
        job.out_pieces = 0;
        pthread_mutex_unlock(&sched->mu);
        const char *p = buf;
        for (int i = 0; i < pieces; i++) {
            if (!stopped && callback && callback(p, userdata)) stopped = 1;
            p += strlen(p) + 1;
        }
        pthread_mutex_lock(&sched->mu);
        if (stopped) job.abort = 1;
        if (done) break;
    }
    pthread_mutex_unlock(&sched->mu);
    free(buf);
    free(job.out);
    pthread_cond_destroy(&job.cv);
}

void batch_scheduler_reset_caches(BatchScheduler *sched) {
    if (!sched) return;
    for (int i = 0; i < sched->opt.n_slots; i++) prefix_cache_reset(sched->slots[i].pc);
}

void batch_scheduler_stats(BatchScheduler *sched, BatchSchedulerStats *out) {
    memset(out, 0, sizeof(*out));
    if (!sched) return;
    pthread_mutex_lock(&sched->mu);
    out->n_slots             = sched->opt.n_slots;
    out->active              = sched->n_active;
    out->pending             = sched->n_pending;
    out->kv_committed_tokens = sched->kv_committed;
    out->kv_budget_tokens    = sched->opt.kv_budget_tokens;
    pthread_mutex_unlock(&sched->mu);
}
