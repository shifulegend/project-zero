#include "reasoning/best_of.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char *buf;
    size_t len, cap;
    BestOfAbortFn abort_fn;
    void *abort_ud;
    int oom;
} Collector;

static int collect_cb(const char *piece, void *ud) {
    Collector *c = (Collector *)ud;
    size_t pl    = strlen(piece);
    if (c->len + pl + 1 > c->cap) {
        size_t nc = c->cap ? c->cap * 2 : 1024;
        while (nc < c->len + pl + 1) nc *= 2;
        char *nb = (char *)realloc(c->buf, nc);
        if (!nb) {
            c->oom = 1;
            return 1;
        }
        c->buf = nb;
        c->cap = nc;
    }
    memcpy(c->buf + c->len, piece, pl + 1);
    c->len += pl;
    return c->abort_fn ? c->abort_fn(c->abort_ud) : 0;
}

double best_of_score(const GenerationStats *st, BestOfScore mode) {
    if (!st || st->completion_tokens <= 0) return -1e300;
    if (mode == BEST_OF_SCORE_SUM_LOGPROB) return st->sum_logprob;
    return st->sum_logprob / (double)st->completion_tokens;
}

typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    RunState *s;
    const MoEConfig *mc;
    Tokenizer *tok;
    ThreadPool *tp;
    const GenerationParams *gp;
    BestOfScore mode;
    BestOfAbortFn abort_fn;
    void *userdata;
    const char *prompt;
    PrefixCache *cache; /* temporary shared cache, or NULL to use gp's own */
} BestOfCtx;

typedef struct {
    char *text;
    double score;
    int complete;
    int index; /* -1 until a candidate has been taken */
} BestOfPick;

static int best_of_args_ok(const Config *cfg, const TransformerWeights *w, const RunState *s,
                           const Tokenizer *tok, const GenerationParams *gp,
                           const BestOfResult *out) {
    return cfg && w && s && tok && gp && out;
}

static int best_of_clamp_n(int n) {
    if (n < 1) return 1;
    return n > BEST_OF_N_MAX ? BEST_OF_N_MAX : n;
}

/* Candidate i uses seed base + i; an unset (negative) seed starts from the clock. */
static unsigned long long best_of_base_seed(long long seed) {
    return seed >= 0 ? (unsigned long long)seed : (unsigned long long)time(NULL);
}

/* A candidate that ended on its own (EOS / stop sequence) beats one cut off by
 * max_tokens: likelihood alone rewards repetitive loops that never terminate
 * (2026-10-18, seen live: "Name a color." won with a 512-token ramble). Score
 * breaks ties within each group. */
static int best_of_beats(const BestOfPick *b, int complete, double sc) {
    return b->index < 0 || complete > b->complete || (complete == b->complete && sc > b->score);
}

/* Records candidate `i` as the new winner, taking ownership of col->buf. */
static void best_of_take(BestOfPick *b, BestOfResult *out, int i, int complete, double sc,
                         const GenerationStats *st, Collector *col) {
    b->score    = sc;
    b->complete = complete;
    b->index    = i;
    out->chosen = i;
    out->winner = *st;
    free(b->text);
    b->text = col->buf ? col->buf : strdup("");
}

/* Runs candidate `i`; returns 0, or 1 if collecting its text ran out of memory. */
static int best_of_candidate(const BestOfCtx *c, int i, unsigned long long base, BestOfPick *best,
                             BestOfResult *out, int *stop) {
    GenerationParams cp = *c->gp;
    GenerationStats st;
    cp.seed             = (long long)(base + (unsigned long long)i);
    cp.compute_logprobs = true;
    cp.stats            = &st;
    if (c->cache) cp.prefix_cache = c->cache;
    Collector col = {NULL, 0, 0, c->abort_fn, c->userdata, 0};
    generate_ex(c->cfg, c->w, c->s, c->mc, c->tok, c->tp, c->prompt, &cp, collect_cb, &col);
    if (col.oom) {
        free(col.buf);
        return 1;
    }
    double sc = best_of_score(&st, c->mode);
    if (i < BEST_OF_N_MAX) out->scores[i] = sc;
    out->n = i + 1;
    out->tokens_total += st.completion_tokens;
    int complete = st.finish_reason == GEN_FINISH_STOP;
    if (best_of_beats(best, complete, sc))
        best_of_take(best, out, i, complete, sc, &st, &col);
    else
        free(col.buf);
    *stop = st.finish_reason == GEN_FINISH_CANCELLED || st.finish_reason == GEN_FINISH_ERROR;
    return 0;
}

/* Runs the candidates in turn; stops early when one is cancelled or fails. */
static TernaryError best_of_run(const BestOfCtx *ctx, int n, BestOfPick *best, BestOfResult *out) {
    unsigned long long base = best_of_base_seed(ctx->gp->seed);
    for (int i = 0, stop = 0; i < n && !stop; i++)
        if (best_of_candidate(ctx, i, base, best, out, &stop)) return TN_ERR_OOM;
    return TN_OK;
}

/* Streams the winning text (if any) to the caller and frees it. */
static void best_of_deliver(BestOfPick *best, TokenCallback callback, void *userdata) {
    if (best->text && best->text[0] && callback) callback(best->text, userdata);
    free(best->text);
}

TernaryError generate_best_of(const Config *cfg, const TransformerWeights *w, RunState *s,
                              const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                              const char *prompt, const GenerationParams *gp, int n,
                              BestOfScore mode, TokenCallback callback, void *userdata,
                              BestOfAbortFn abort_fn, BestOfResult *out) {
    if (!best_of_args_ok(cfg, w, s, tok, gp, out)) return TN_ERR_INVALID_ARGS;
    n = best_of_clamp_n(n);
    memset(out, 0, sizeof(*out));
    out->chosen = -1;

    /* Candidates share the prompt: without a caller-provided prefix cache,
     * use a temporary one so only the first candidate prefills it. */
    PrefixCache tmp_cache;
    int own_cache = 0;
    if (!gp->prefix_cache && prefix_cache_init(&tmp_cache, s->max_seq_len) == 0) own_cache = 1;

    BestOfCtx ctx   = {cfg, w,    s,        mc,       tok,    tp,
                       gp,  mode, abort_fn, userdata, prompt, own_cache ? &tmp_cache : NULL};
    BestOfPick best = {NULL, -1e300, 0, -1};
    TernaryError rc = best_of_run(&ctx, n, &best, out);
    if (own_cache) prefix_cache_free(&tmp_cache);
    if (rc != TN_OK) {
        free(best.text);
        return rc;
    }
    best_of_deliver(&best, callback, userdata);
    if (gp->stats) *gp->stats = out->winner;
    return TN_OK;
}
