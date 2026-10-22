#include "reasoning/best_of.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char  *buf;
    size_t len, cap;
    BestOfAbortFn abort_fn;
    void  *abort_ud;
    int    oom;
} Collector;

static int collect_cb(const char *piece, void *ud) {
    Collector *c = (Collector *)ud;
    size_t pl = strlen(piece);
    if (c->len + pl + 1 > c->cap) {
        size_t nc = c->cap ? c->cap * 2 : 1024;
        while (nc < c->len + pl + 1) nc *= 2;
        char *nb = (char *)realloc(c->buf, nc);
        if (!nb) { c->oom = 1; return 1; }
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

TernaryError generate_best_of(const Config *cfg, const TransformerWeights *w,
                              RunState *s, const MoEConfig *mc,
                              Tokenizer *tok, ThreadPool *tp, const char *prompt,
                              const GenerationParams *gp, int n, BestOfScore mode,
                              TokenCallback callback, void *userdata,
                              BestOfAbortFn abort_fn, BestOfResult *out) {
    if (!cfg || !w || !s || !tok || !gp || !out) return TN_ERR_INVALID_ARGS;
    if (n < 1) n = 1;
    if (n > BEST_OF_N_MAX) n = BEST_OF_N_MAX;
    memset(out, 0, sizeof(*out));
    out->chosen = -1;

    /* Candidates share the prompt: without a caller-provided prefix cache,
     * use a temporary one so only the first candidate prefills it. */
    PrefixCache tmp_cache;
    int own_cache = 0;
    if (!gp->prefix_cache && prefix_cache_init(&tmp_cache, s->max_seq_len) == 0) own_cache = 1;

    char *best_text = NULL;
    double best_score = -1e300;
    int best_complete = 0;
    unsigned long long base = gp->seed >= 0 ? (unsigned long long)gp->seed
                                            : (unsigned long long)time(NULL);
    for (int i = 0; i < n; i++) {
        GenerationParams cp = *gp;
        GenerationStats st;
        cp.seed = (long long)(base + (unsigned long long)i);
        cp.compute_logprobs = true;
        cp.stats = &st;
        if (own_cache) cp.prefix_cache = &tmp_cache;
        Collector col = { NULL, 0, 0, abort_fn, userdata, 0 };
        generate_ex(cfg, w, s, mc, tok, tp, prompt, &cp, collect_cb, &col);
        if (col.oom) {
            free(col.buf); free(best_text);
            if (own_cache) prefix_cache_free(&tmp_cache);
            return TN_ERR_OOM;
        }
        double sc = best_of_score(&st, mode);
        if (i < BEST_OF_N_MAX) out->scores[i] = sc;
        out->n = i + 1;
        out->tokens_total += st.completion_tokens;
        /* A candidate that ended on its own (EOS / stop sequence) beats one
         * cut off by max_tokens: likelihood alone rewards repetitive loops
         * that never terminate (2026-10-18, seen live: "Name a color." won
         * with a 512-token ramble). Score breaks ties within each group. */
        int complete = st.finish_reason == GEN_FINISH_STOP;
        if (out->chosen < 0 || complete > best_complete ||
            (complete == best_complete && sc > best_score)) {
            best_score = sc;
            best_complete = complete;
            out->chosen = i;
            out->winner = st;
            free(best_text);
            best_text = col.buf ? col.buf : strdup("");
        } else {
            free(col.buf);
        }
        if (st.finish_reason == GEN_FINISH_CANCELLED || st.finish_reason == GEN_FINISH_ERROR) break;
    }
    if (own_cache) prefix_cache_free(&tmp_cache);
    if (best_text && best_text[0] && callback) callback(best_text, userdata);
    free(best_text);
    if (gp->stats) *gp->stats = out->winner;
    return TN_OK;
}
