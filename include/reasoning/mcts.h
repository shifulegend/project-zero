#ifndef TN_REASONING_MCTS_H
#define TN_REASONING_MCTS_H

/*
 * reasoning/mcts — Phase 30.2/30.3 step-level Monte Carlo tree search.
 *
 * A node is one reasoning step (text up to `step_delim`, at most
 * `step_tokens` tokens) appended to its parent's path. Each simulation:
 *   select   — from the root, UCB1 (value + c*sqrt(ln N / n)) over children of
 *              fully expanded nodes;
 *   expand   — sample one new step from the selected path (the policy model
 *              continues the assistant turn via GenerationParams.assistant_prefix);
 *   evaluate — score the step: the external PRM (reasoning/prm.h) when given,
 *              else the policy's own geometric-mean token probability;
 *   backup   — add the score to every node on the path.
 * The answer follows the most-visited child from the root (ties: higher
 * value); a non-terminal leaf is then completed greedily. Prompt prefixes are
 * shared through the prefix cache / prompt store when the caller passes them
 * in `base`.
 */

#include "reasoning/mcts_options.h"
#include "reasoning/prm.h"
#include "transformer/generate.h"

typedef struct {
    int    simulations;   /* run */
    int    nodes;         /* tree size (excl. root) */
    int    steps;         /* steps on the chosen path */
    int    tokens_total;  /* policy tokens generated across the search */
    float  path_value;    /* mean score of the chosen path's last searched node */
    int    used_prm;      /* 1 when the external PRM scored steps */
    int    path_tokens;   /* tokens of the chosen searched steps (part of the answer) */
} MctsResult;

typedef int (*MctsAbortFn)(void *userdata);

/*
 * Searches and then streams the chosen answer through `callback`.
 * `question` is the text the PRM sees as {question} (usually the last user
 * message); `prompt` / `base` are what generate_ex() would get. `prm` may be
 * NULL (self-scoring). Returns TN_OK, or an error if nothing could be generated.
 */
TernaryError generate_mcts(const Config *cfg, const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                           const char *prompt, const char *question,
                           const GenerationParams *base, const MctsOptions *opt,
                           PrmModel *prm, TokenCallback callback, void *userdata,
                           MctsAbortFn abort_fn, MctsResult *out);

/* ── pure helpers (unit-tested) ── */
/* UCB1 score; unvisited children score +inf (explored first). */
double mcts_ucb(double value_sum, int visits, int parent_visits, double c);
/* Self-score of a step from its stats: exp(mean log p), in (0, 1]. */
float  mcts_self_score(const GenerationStats *st);

#endif /* TN_REASONING_MCTS_H */
