#ifndef TN_PERPLEXITY_H
#define TN_PERPLEXITY_H

/*
 * perplexity.h — model perplexity on a token stream, computed the way
 * llama.cpp's llama-perplexity does so the two engines can be compared on
 * the same file: the stream is cut into chunks of n_ctx tokens, each chunk
 * starts from position 0 with its first token replaced by BOS (when the
 * model has one), and only the second half of each chunk is scored (the
 * first half is context). PPL = exp(mean NLL); the ± is the standard error
 * of the mean NLL propagated through exp.
 */

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"

typedef struct {
    int n_ctx;       /* tokens per chunk (default 512, capped at the context size) */
    int max_chunks;  /* 0 = every full chunk in the stream */
    int bos_token;   /* < 0: keep the stream's own first token */
} PplOptions;

typedef struct {
    double ppl, ppl_err;   /* perplexity and its standard error */
    double mean_nll;
    int    n_scored;       /* scored tokens */
    int    n_chunks;
    double seconds;
} PplResult;

/* Called after each chunk with the running result (may be NULL). */
typedef void (*PplProgressFn)(int chunk, const PplResult *running, void *user);

/* Pure helpers (unit-tested): NLL of `target` under `logits`, and the
 * accumulation -> PPL/± from a sum and sum of squares. */
double ppl_token_nll(const float *logits, int vocab, int target);
void   ppl_finish(PplResult *r, double sum_nll, double sum_nll2, int n);

TernaryError perplexity_eval(const int *tokens, int n_tokens, const PplOptions *opt,
                             const Config *cfg, const TransformerWeights *w, RunState *s,
                             const MoEConfig *mc, ThreadPool *tp,
                             PplProgressFn progress, void *user, PplResult *out);

#endif /* TN_PERPLEXITY_H */
