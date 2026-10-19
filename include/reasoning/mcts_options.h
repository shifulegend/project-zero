#ifndef TN_REASONING_MCTS_OPTIONS_H
#define TN_REASONING_MCTS_OPTIONS_H

/*
 * reasoning/mcts_options — Phase 30 search shape, dependency-free so the API
 * server config can carry the defaults (search itself: reasoning/mcts.h).
 * Ranges (CLI, API and /v1/config share them): simulations 0..4096, expand
 * 1..16, max_depth 1..256, step_tokens 1..4096, c (0, 10], temperature (0, 2].
 */

typedef struct {
    int   simulations;   /* --mcts N: search iterations (0 = off) */
    int   expand;        /* children per node before UCB descends (branching factor) */
    int   max_depth;     /* steps per path */
    int   step_tokens;   /* token cap per step */
    float c_ucb;         /* exploration constant */
    float temperature;   /* sampling temperature for expansions (> 0) */
    const char *step_delim; /* step boundary (default "\n") */
} MctsOptions;

#define MCTS_DEFAULT_EXPAND      3
#define MCTS_DEFAULT_DEPTH       8
#define MCTS_DEFAULT_STEP_TOKENS 64
#define MCTS_DEFAULT_C           1.4f
#define MCTS_DEFAULT_TEMPERATURE 0.8f
#define MCTS_DEFAULT_DELIM       "\n"

#define MCTS_MAX_EXPAND      16
#define MCTS_MAX_DEPTH       256
#define MCTS_MAX_STEP_TOKENS 4096
#define MCTS_MAX_SIMULATIONS 4096

/* Fills unset (<= 0 / NULL) fields with the defaults above. */
void mcts_options_defaults(MctsOptions *o);
/* Clamps every field to its range (simulations also to sim_cap >= 0). */
void mcts_options_clamp(MctsOptions *o, int sim_cap);

#endif /* TN_REASONING_MCTS_OPTIONS_H */
