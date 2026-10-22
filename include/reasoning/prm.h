#ifndef TN_REASONING_PRM_H
#define TN_REASONING_PRM_H

/*
 * reasoning/prm — Phase 30.1 process reward model: scores one reasoning step
 * in [0, 1] with a second GGUF model used as a judge.
 *
 * The judge sees `format` with {question} and {steps} substituted (steps =
 * the partial solution ending in the step being scored) and the score is
 * P(good) / (P(good) + P(bad)) over its next-token distribution, where good
 * and bad are the first tokens of two configurable strings. This covers
 * Math-Shepherd-style PRMs ("+"/"-" after a step tag) and plain instruct
 * models asked "Is the last step correct?" ("Yes"/"No"). Nothing about the
 * judge is hardcoded: format, good, bad come from CLI flags.
 */

#include "cli/model_load.h"
#include "core/error.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

#define PRM_DEFAULT_FORMAT \
    "Question: {question}\nSolution so far:\n{steps}\nIs the last step correct? Answer Yes or No.\nAnswer:"
#define PRM_DEFAULT_GOOD " Yes"
#define PRM_DEFAULT_BAD  " No"

typedef struct PrmModel {
    LoadedModel model;
    Tokenizer   tok;
    ThreadPool *tp;
    char       *format;
    int         good_id, bad_id;
    int        *prev;      /* tokens currently in the judge's KV (prefix reuse) */
    int         n_prev;
    int         loaded;
} PrmModel;

/* Loads the judge and resolves good/bad tokens (NULL = defaults above). */
TernaryError prm_load(PrmModel *pm, const char *path, const char *format,
                      const char *good, const char *bad, int max_seq_len, ThreadPool *tp);
void         prm_free(PrmModel *pm);

/* Score of the last step of `steps` for `question`: [0, 1], or -1 on error. */
float prm_score(PrmModel *pm, const char *question, const char *steps);

/* ── pure helpers (unit-tested without a model) ── */

/* Substitutes {question} and {steps} in fmt. Returns a malloc'd string. */
char *prm_format(const char *fmt, const char *question, const char *steps);
/* P(good) / (P(good) + P(bad)) from raw logits; -1 when an id is invalid. */
float prm_score_from_logits(const float *logits, int vocab, int good_id, int bad_id);

#endif /* TN_REASONING_PRM_H */
