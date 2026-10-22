/*
 * Phase 23 stage 2b — src/api/batch_scheduler.c
 * See include/api/batch_scheduler.h for responsibility.
 */

#include "api/batch_scheduler.h"
#include "transformer/gen_session.h"
#include "transformer/forward.h"
#include "speculative/spec_scratch.h"
#include "core/run_state.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct BatchJob {
    const char *prompt;
    const GenerationParams *gp;
    int use_cache;
    GenSession g;
    struct BatchScheduler *bs;
    /* Output channel, guarded by bs->mu: NUL-separated pieces. */
    char  *out;
    size_t out_len, out_cap;
    int    out_pieces;
    int    done;
    int    abort;
    pthread_cond_t cv;
    struct BatchJob *next;
} BatchJob;

typedef struct {
    RunState   *s;
    int         owned;        /* allocated by the scheduler (slots >= 1) */
    PrefixCache *pc;
    PrefixCache own_pc;
    BatchJob   *job;          /* NULL when idle */
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
    int       *row_tok, *row_pos;
    RunState **row_state;
    unsigned char *row_want;
    int       *order, *count; /* per step: slots in row order, rows per slot */
    float     *logits;        /* [n_slots][vocab]: at most one wanted row per slot */

    pthread_mutex_t mu;
    pthread_cond_t  cv;       /* new work / stop */
    BatchJob *pend_head, *pend_tail;
    int n_pending, n_active;
    long long kv_committed;   /* snapshot for stats, updated by the scheduler under mu */
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
    BatchJob *job = (BatchJob *)userdata;
    BatchScheduler *bs = job->bs;
    size_t n = piece ? strlen(piece) : 0;
    pthread_mutex_lock(&bs->mu);
    int stop = job->abort;
    if (!stop && piece) {
        if (job->out_len + n + 1 > job->out_cap) {
            size_t cap = job->out_cap ? job->out_cap * 2 : 1024;
            while (cap < job->out_len + n + 1) cap *= 2;
            char *nb = (char *)realloc(job->out, cap);
            if (!nb) { job->abort = stop = 1; }
            else { job->out = nb; job->out_cap = cap; }
        }
        if (!stop) {
            memcpy(job->out + job->out_len, piece, n + 1);
            job->out_len += n + 1;
            job->out_pieces++;
            pthread_cond_signal(&job->cv);
        }
    }
    pthread_mutex_unlock(&bs->mu);
    return stop;
}

static void job_finish_locked(BatchJob *job) {
    job->done = 1;
    pthread_cond_signal(&job->cv);
}

/* ── Slots and KV budget ──────────────────────────────────────────────── */

static long long kv_need(const GenSession *g, const Config *cfg) {
    long long need = (long long)g->n_prompt + (g->max_tokens > 0 ? g->max_tokens : 0);
    return need < cfg->seq_len ? need : cfg->seq_len;
}

static long long kv_committed_total(const BatchScheduler *bs) {
    long long t = 0;
    for (int i = 0; i < bs->opt.n_slots; i++) t += bs->slots[i].s->kv_committed;
    return t;
}

static void slot_release_kv(BatchScheduler *bs, BatchSlot *sl) {
    prefix_cache_reset(sl->pc);   /* the cached prefix is about to vanish */
    run_state_kv_release(sl->s, bs->cfg);
}

/* Free slot best suited to the job: the longest reusable prefix, else the
 * least memory already committed (keeps other slots' caches warm). */
static int pick_slot(BatchScheduler *bs, const BatchJob *job, int *match_out) {
    int best = -1, best_match = -1;
    long long best_commit = 0;
    for (int i = 0; i < bs->opt.n_slots; i++) {
        BatchSlot *sl = &bs->slots[i];
        if (sl->job) continue;
        int m = job->use_cache ? prefix_cache_match(sl->pc, job->g.prompt_tokens, job->g.n_prompt) : 0;
        long long c = sl->s->kv_committed;
        if (best < 0 || m > best_match || (m == best_match && c < best_commit)) {
            best = i; best_match = m; best_commit = c;
        }
    }
    *match_out = best_match;
    return best;
}

/* Makes room for job on slot `idx`; returns 1 when it may start now. */
static int kv_make_room(BatchScheduler *bs, int idx, int match, const BatchJob *job) {
    long long budget = bs->opt.kv_budget_tokens;
    if (budget <= 0) return 1;
    long long need = kv_need(&job->g, bs->cfg);
    for (;;) {
        BatchSlot *own = &bs->slots[idx];
        long long own_c = own->s->kv_committed;
        long long after = kv_committed_total(bs) - own_c + (own_c > need ? own_c : need);
        if (after <= budget) return 1;
        /* Release the idle slot holding the most memory. */
        int victim = -1;
        for (int i = 0; i < bs->opt.n_slots; i++) {
            BatchSlot *sl = &bs->slots[i];
            if (i == idx || sl->job || sl->s->kv_committed <= 0) continue;
            if (victim < 0 || sl->s->kv_committed > bs->slots[victim].s->kv_committed) victim = i;
        }
        if (victim >= 0) { slot_release_kv(bs, &bs->slots[victim]); continue; }
        if (own_c > need && match == 0 && run_state_kv_release(own->s, bs->cfg) > 0) {
            prefix_cache_reset(own->pc);
            continue;
        }
        /* Active requests hold the rest: wait for one to finish, unless
         * none is running (then nothing would ever free it). */
        return bs->n_active == 0;
    }
}

/* ── Scheduler thread ─────────────────────────────────────────────────── */

static void start_job(BatchScheduler *bs, int idx, BatchJob *job) {
    BatchSlot *sl = &bs->slots[idx];
    sl->s->current_pos = 0;       /* every request is a fresh conversation */
    if (!job->use_cache) prefix_cache_reset(sl->pc);   /* KV is about to be overwritten */
    gen_session_start(&job->g, sl->s, job->use_cache ? sl->pc : NULL,
                      job->use_cache ? bs->pstore : NULL);
    sl->job = job;
    sl->admit_seq = ++bs->seq;
}

/* Called with bs->mu held; may drop it while preparing prompts. */
static void admit(BatchScheduler *bs, BatchJob **staged) {
    for (;;) {
        int free_slots = 0;
        for (int i = 0; i < bs->opt.n_slots; i++) if (!bs->slots[i].job) free_slots++;
        if (!free_slots) return;
        if (!*staged) {
            BatchJob *j = bs->pend_head;
            if (!j) return;
            bs->pend_head = j->next;
            if (!bs->pend_head) bs->pend_tail = NULL;
            bs->n_pending--;
            j->next = NULL;
            if (j->abort || bs->stopping) {      /* gone before it started */
                job_finish_locked(j);
                continue;
            }
            pthread_mutex_unlock(&bs->mu);
            int ok = gen_session_prepare(&j->g, bs->cfg, bs->w, bs->mc, bs->tok, bs->tp,
                                         j->prompt, j->gp, job_enqueue, j) == 0;
            if (!ok) {
                gen_session_end(&j->g);
                pthread_mutex_lock(&bs->mu);
                job_finish_locked(j);
                continue;
            }
            pthread_mutex_lock(&bs->mu);
            *staged = j;
            bs->n_pending++;                     /* still waiting, from the caller's view */
        }
        int match = 0;
        int idx = pick_slot(bs, *staged, &match);
        if (!kv_make_room(bs, idx, match, *staged)) return;   /* wait for a finish */
        BatchJob *j = *staged;
        *staged = NULL;
        bs->n_pending--;
        bs->n_active++;
        pthread_mutex_unlock(&bs->mu);
        start_job(bs, idx, j);
        pthread_mutex_lock(&bs->mu);
        bs->kv_committed = kv_committed_total(bs);
    }
}

/* One batched forward pass over every active request. */
static void step(BatchScheduler *bs) {
    int n_slots = bs->opt.n_slots, cap = bs->opt.max_batch_rows;
    int *order = bs->order, *count = bs->count;
    int n_order = 0, rows = 0;
    memset(count, 0, (size_t)n_slots * sizeof(int));

    /* Decode rows first (one each), so generation never waits behind a
     * long prompt; prefill chunks take what is left, oldest request first. */
    for (int pass = 0; pass < 2 && rows < cap; pass++) {
        for (;;) {
            int pick = -1;
            for (int i = 0; i < n_slots; i++) {
                BatchSlot *sl = &bs->slots[i];
                if (!sl->job || count[i] || sl->job->g.done) continue;
                if (gen_session_needs_logits(&sl->job->g) != (pass == 0)) continue;
                if (pick < 0 || sl->admit_seq < bs->slots[pick].admit_seq) pick = i;
            }
            if (pick < 0 || rows >= cap) break;
            const GenSession *g = &bs->slots[pick].job->g;
            int n = gen_session_input_rows(g, bs->row_tok + rows, bs->row_pos + rows, cap - rows);
            if (n <= 0) { count[pick] = -1; continue; }
            for (int k = 0; k < n; k++) {
                bs->row_state[rows + k] = bs->slots[pick].s;
                bs->row_want[rows + k] = (g->step + k >= g->n_prompt - 1);
            }
            count[pick] = n;
            order[n_order++] = pick;
            rows += n;
        }
    }
    if (rows == 0) return;

    /* One row (a lone request decoding): the single-sequence forward is
     * ~5% faster than a one-row batch and is exactly what generate_ex runs. */
    TernaryError err = TN_OK;
    float *single = NULL;
    if (rows == 1)
        single = transformer_forward(bs->row_tok[0], bs->row_pos[0], bs->cfg, bs->w,
                                     bs->row_state[0], bs->mc, bs->tp);
    else
        err = transformer_forward_multi_ex(bs->row_tok, bs->row_pos, bs->row_state, rows,
                                           bs->row_want, bs->cfg, bs->w, &bs->sb, bs->mc,
                                           bs->tp, bs->logits);
    if (err != TN_OK)
        fprintf(stderr, "[parallel] batched forward failed: %s\n", tn_error_str(err));
    int r = 0, out = 0;
    for (int o = 0; o < n_order; o++) {
        GenSession *g = &bs->slots[order[o]].job->g;
        for (int k = 0; k < count[order[o]]; k++, r++) {
            float *lg = !bs->row_want[r] ? NULL
                      : single ? single : bs->logits + (size_t)(out++) * bs->cfg->vocab_size;
            if (err != TN_OK) { g->finish = GEN_FINISH_ERROR; g->done = 1; }
            else if (!g->done) gen_session_consume(g, lg);
        }
    }
}

static void finish_done(BatchScheduler *bs) {
    for (int i = 0; i < bs->opt.n_slots; i++) {
        BatchSlot *sl = &bs->slots[i];
        if (!sl->job || !sl->job->g.done) continue;
        BatchJob *j = sl->job;
        gen_session_end(&j->g);               /* flush, snapshot, stats */
        pthread_mutex_lock(&bs->mu);
        sl->job = NULL;
        bs->n_active--;
        /* Last look at slot memory for this request: once it is marked done
         * its caller may hand the engine (slot 0 included) to the
         * sequential path, so an idle scheduler never reads slot state. */
        bs->kv_committed = kv_committed_total(bs);
        job_finish_locked(j);
        pthread_mutex_unlock(&bs->mu);
    }
}

static void *scheduler_main(void *arg) {
    BatchScheduler *bs = (BatchScheduler *)arg;
    BatchJob *staged = NULL;
    pthread_mutex_lock(&bs->mu);
    for (;;) {
        if (bs->stopping) {
            for (int i = 0; i < bs->opt.n_slots; i++)
                if (bs->slots[i].job) bs->slots[i].job->abort = 1;
            if (staged) {                      /* prepared but never started */
                BatchJob *j = staged;
                staged = NULL;
                bs->n_pending--;
                pthread_mutex_unlock(&bs->mu);
                gen_session_end(&j->g);
                pthread_mutex_lock(&bs->mu);
                job_finish_locked(j);
            }
        }
        /* A request whose client stopped listening ends now, even mid-prompt. */
        for (int i = 0; i < bs->opt.n_slots; i++) {
            BatchJob *j = bs->slots[i].job;
            if (j && j->abort && !j->g.done) { j->g.finish = GEN_FINISH_CANCELLED; j->g.done = 1; }
        }
        admit(bs, &staged);
        if (bs->n_active == 0) {
            if (bs->stopping && !bs->pend_head && !staged) break;
            pthread_cond_wait(&bs->cv, &bs->mu);
            continue;
        }
        int any_done = 0;
        for (int i = 0; i < bs->opt.n_slots; i++)
            if (bs->slots[i].job && bs->slots[i].job->g.done) any_done = 1;
        pthread_mutex_unlock(&bs->mu);
        if (!any_done) step(bs);
        finish_done(bs);
        pthread_mutex_lock(&bs->mu);
    }
    pthread_mutex_unlock(&bs->mu);
    return NULL;
}

/* ── Public API ───────────────────────────────────────────────────────── */

static void free_slots(BatchScheduler *bs) {
    if (!bs->slots) return;
    for (int i = 0; i < bs->opt.n_slots; i++) {
        BatchSlot *sl = &bs->slots[i];
        if (!sl->owned) continue;
        if (sl->s) { run_state_free(sl->s); free(sl->s); }
        prefix_cache_free(&sl->own_pc);
    }
    free(bs->slots);
    bs->slots = NULL;
}

BatchScheduler *batch_scheduler_create(const Config *cfg, const TransformerWeights *w,
                                       RunState *slot0, PrefixCache *slot0_cache,
                                       PromptStore *pstore, const MoEConfig *mc,
                                       Tokenizer *tok, ThreadPool *tp,
                                       const BatchSchedulerOptions *opt) {
    if (!cfg || !w || !slot0 || !slot0_cache || !tok || !opt || opt->n_slots < 1 ||
        opt->max_batch_rows < opt->n_slots || !batch_scheduler_supported(w, mc))
        return NULL;
    BatchScheduler *bs = (BatchScheduler *)calloc(1, sizeof(*bs));
    if (!bs) return NULL;
    bs->cfg = cfg; bs->w = w; bs->mc = mc; bs->tok = tok; bs->tp = tp; bs->pstore = pstore;
    bs->opt = *opt;
    int n = opt->n_slots, rows = opt->max_batch_rows;
    bs->slots = (BatchSlot *)calloc((size_t)n, sizeof(BatchSlot));
    bs->row_tok = (int *)malloc((size_t)rows * sizeof(int));
    bs->row_pos = (int *)malloc((size_t)rows * sizeof(int));
    bs->row_state = (RunState **)malloc((size_t)rows * sizeof(RunState *));
    bs->row_want = (unsigned char *)malloc((size_t)rows);
    bs->order = (int *)malloc((size_t)n * sizeof(int));
    bs->count = (int *)malloc((size_t)n * sizeof(int));
    bs->logits = (float *)malloc((size_t)n * (size_t)cfg->vocab_size * sizeof(float));
    int ok = bs->slots && bs->row_tok && bs->row_pos && bs->row_state && bs->row_want &&
             bs->order && bs->count && bs->logits &&
             spec_batch_scratch_alloc(&bs->sb, cfg, rows) == TN_OK;
    if (ok && !slot0_cache->tokens) prefix_cache_init(slot0_cache, slot0->max_seq_len);
    for (int i = 0; ok && i < n; i++) {
        BatchSlot *sl = &bs->slots[i];
        if (i == 0) { sl->s = slot0; sl->pc = slot0_cache; continue; }
        sl->owned = 1;
        sl->s = (RunState *)malloc(sizeof(RunState));
        if (!sl->s) { ok = 0; break; }
        memset(sl->s, 0, sizeof(*sl->s));
        if (run_state_alloc(sl->s, cfg, slot0->max_seq_len) != TN_OK) {
            free(sl->s); sl->s = NULL; ok = 0; break;
        }
        sl->s->active_lora = slot0->active_lora;   /* one adapter for every sequence */
        prefix_cache_init(&sl->own_pc, sl->s->max_seq_len);
        sl->pc = &sl->own_pc;
    }
    if (ok) {
        pthread_mutex_init(&bs->mu, NULL);
        pthread_cond_init(&bs->cv, NULL);
        ok = pthread_create(&bs->thread, NULL, scheduler_main, bs) == 0;
        bs->thread_started = ok;
        if (!ok) { pthread_cond_destroy(&bs->cv); pthread_mutex_destroy(&bs->mu); }
    }
    if (!ok) {
        fprintf(stderr, "[parallel] could not set up %d slots (out of memory?)\n", n);
        free_slots(bs);
        spec_batch_scratch_free(&bs->sb);
        free(bs->row_tok); free(bs->row_pos); free(bs->row_state); free(bs->row_want);
        free(bs->order); free(bs->count); free(bs->logits);
        free(bs);
        return NULL;
    }
    return bs;
}

void batch_scheduler_stop(BatchScheduler *bs) {
    if (!bs || !bs->thread_started) return;
    pthread_mutex_lock(&bs->mu);
    bs->stopping = 1;
    pthread_cond_broadcast(&bs->cv);
    pthread_mutex_unlock(&bs->mu);
    pthread_join(bs->thread, NULL);
    bs->thread_started = 0;
}

void batch_scheduler_destroy(BatchScheduler *bs) {
    if (!bs) return;
    batch_scheduler_stop(bs);
    pthread_cond_destroy(&bs->cv);
    pthread_mutex_destroy(&bs->mu);
    free_slots(bs);
    spec_batch_scratch_free(&bs->sb);
    free(bs->row_tok); free(bs->row_pos); free(bs->row_state); free(bs->row_want);
    free(bs->order); free(bs->count); free(bs->logits);
    free(bs);
}

void batch_scheduler_generate(BatchScheduler *bs, const char *prompt,
                              const GenerationParams *gp, int use_cache,
                              TokenCallback callback, void *userdata) {
    BatchJob job;
    memset(&job, 0, sizeof(job));
    job.prompt = prompt;
    job.gp = gp;
    job.use_cache = use_cache;
    job.bs = bs;
    pthread_cond_init(&job.cv, NULL);
    if (gp && gp->stats) {
        memset(gp->stats, 0, sizeof(*gp->stats));
        gp->stats->finish_reason = GEN_FINISH_CANCELLED;  /* until the session reports */
    }

    pthread_mutex_lock(&bs->mu);
    if (bs->stopping) {
        pthread_mutex_unlock(&bs->mu);
        pthread_cond_destroy(&job.cv);
        return;
    }
    if (bs->pend_tail) bs->pend_tail->next = &job; else bs->pend_head = &job;
    bs->pend_tail = &job;
    bs->n_pending++;
    pthread_cond_signal(&bs->cv);

    /* Deliver queued pieces on this thread until the request is finished. */
    char  *buf = NULL;
    size_t cap = 0;
    int stopped = 0;
    for (;;) {
        while (!job.out_pieces && !job.done) pthread_cond_wait(&job.cv, &bs->mu);
        int pieces = job.out_pieces, done = job.done;
        char *tmp = buf; buf = job.out; job.out = tmp;
        size_t tcap = cap; cap = job.out_cap; job.out_cap = tcap;
        job.out_len = 0;
        job.out_pieces = 0;
        pthread_mutex_unlock(&bs->mu);
        const char *p = buf;
        for (int i = 0; i < pieces; i++) {
            if (!stopped && callback && callback(p, userdata)) stopped = 1;
            p += strlen(p) + 1;
        }
        pthread_mutex_lock(&bs->mu);
        if (stopped) job.abort = 1;
        if (done) break;
    }
    pthread_mutex_unlock(&bs->mu);
    free(buf);
    free(job.out);
    pthread_cond_destroy(&job.cv);
}

void batch_scheduler_reset_caches(BatchScheduler *bs) {
    if (!bs) return;
    for (int i = 0; i < bs->opt.n_slots; i++) prefix_cache_reset(bs->slots[i].pc);
}

void batch_scheduler_stats(BatchScheduler *bs, BatchSchedulerStats *out) {
    memset(out, 0, sizeof(*out));
    if (!bs) return;
    pthread_mutex_lock(&bs->mu);
    out->n_slots = bs->opt.n_slots;
    out->active = bs->n_active;
    out->pending = bs->n_pending;
    out->kv_committed_tokens = bs->kv_committed;
    out->kv_budget_tokens = bs->opt.kv_budget_tokens;
    pthread_mutex_unlock(&bs->mu);
}
