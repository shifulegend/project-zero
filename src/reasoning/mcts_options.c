/*
 * reasoning/mcts_options.c — Phase 30 search-shape defaults and ranges
 * (see reasoning/mcts_options.h).
 */
#include "reasoning/mcts_options.h"

#include <stddef.h>

void mcts_options_defaults(MctsOptions *o) {
    if (!o) return;
    if (o->expand <= 0) o->expand = MCTS_DEFAULT_EXPAND;
    if (o->max_depth <= 0) o->max_depth = MCTS_DEFAULT_DEPTH;
    if (o->step_tokens <= 0) o->step_tokens = MCTS_DEFAULT_STEP_TOKENS;
    if (o->c_ucb <= 0.0f) o->c_ucb = MCTS_DEFAULT_C;
    if (o->temperature <= 0.0f) o->temperature = MCTS_DEFAULT_TEMPERATURE;
    if (!o->step_delim || !o->step_delim[0]) o->step_delim = MCTS_DEFAULT_DELIM;
}

void mcts_options_clamp(MctsOptions *o, int sim_cap) {
    if (!o) return;
    mcts_options_defaults(o);
    if (o->simulations < 0) o->simulations = 0;
    if (o->simulations > MCTS_MAX_SIMULATIONS) o->simulations = MCTS_MAX_SIMULATIONS;
    if (sim_cap >= 0 && o->simulations > sim_cap) o->simulations = sim_cap;
    if (o->expand > MCTS_MAX_EXPAND) o->expand = MCTS_MAX_EXPAND;
    if (o->max_depth > MCTS_MAX_DEPTH) o->max_depth = MCTS_MAX_DEPTH;
    if (o->step_tokens > MCTS_MAX_STEP_TOKENS) o->step_tokens = MCTS_MAX_STEP_TOKENS;
    if (o->c_ucb > 10.0f) o->c_ucb = 10.0f;
    if (o->temperature > 2.0f) o->temperature = 2.0f;
}
