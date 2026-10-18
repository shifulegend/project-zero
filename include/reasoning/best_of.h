#ifndef TN_BEST_OF_H
#define TN_BEST_OF_H

/*
 * best_of.h — Phase 30 (test-time compute), best-of-N selection.
 *
 * Generates N independent candidates for one prompt (distinct seeds; with a
 * prefix cache the shared prompt is only prefilled once), scores each and
 * delivers only the winner through the caller's callback. The scorer is
 * pluggable; the built-in one is the policy model's own length-normalised
 * log-likelihood (mean log p(token)), which needs no extra model; candidates
 * that finished on their own outrank ones truncated by max_tokens. A process
 * reward model or tree search can plug in at BestOfScoreFn later.
 */

#include "transformer/generate.h"

#define BEST_OF_MAX_DEFAULT 8

typedef enum {
    BEST_OF_SCORE_MEAN_LOGPROB = 0, /* sum / tokens — default, length-fair */
    BEST_OF_SCORE_SUM_LOGPROB  = 1  /* favours short, confident answers */
} BestOfScore;

typedef struct {
    int     n;            /* candidates actually generated */
    int     chosen;       /* index of the winner (-1 if none) */
    double  scores[64];   /* score per candidate (first min(n, 64)) */
    int     tokens_total; /* completion tokens across all candidates */
    GenerationStats winner;
} BestOfResult;

/* Returns 0 to keep going, nonzero to abandon (e.g. client cancelled). */
typedef int (*BestOfAbortFn)(void *userdata);

/* Score of a finished candidate from its stats. */
double best_of_score(const GenerationStats *st, BestOfScore mode);

/*
 * Runs n (>= 1, clamped to 64) candidates with the settings in *gp (whose
 * seed, if >= 0, becomes seed + i per candidate), then streams the winning
 * text through callback. `gp->stats` (if set) receives the winner's stats.
 * Returns TN_OK, TN_ERR_OOM, or TN_ERR_INVALID_ARGS.
 */
TernaryError generate_best_of(const Config *cfg, const TransformerWeights *w,
                              RunState *s, const MoEConfig *mc,
                              Tokenizer *tok, ThreadPool *tp, const char *prompt,
                              const GenerationParams *gp, int n, BestOfScore mode,
                              TokenCallback callback, void *userdata,
                              BestOfAbortFn abort_fn, BestOfResult *out);

#endif /* TN_BEST_OF_H */
