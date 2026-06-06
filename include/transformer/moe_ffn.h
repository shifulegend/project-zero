#ifndef TN_MOE_FFN_H
#define TN_MOE_FFN_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"
#include <stdint.h>

/**
 * moe_ffn_forward — Execute the MoE FFN for one token in one layer.
 * (full description unchanged)
 */
void moe_ffn_forward(RunState              *s,
                     const TransformerWeights *w,
                     const Config          *cfg,
                     const MoEConfig       *mc,
                     int                    layer,
                     ThreadPool            *tp);

/* ---- Expert hit tracking (Phase 17 diagnostics) ---- */

/**
 * Reset expert hit counters before a new prompt.
 * n_layers × num_experts matrix, all zeroed.
 * No-op when expert tracking is disabled or not compiled for MoE.
 */
void moe_expert_tracking_reset(int n_layers, int num_experts);

/**
 * Print per-layer expert utilisation summary to stdout.
 * Shows: total invocations, unique expert count, top-5 expert IDs.
 * Call after generation completes.
 */
void moe_expert_tracking_print(int n_layers, int num_experts);

/**
 * Free expert tracking tables (call at model unload).
 */
void moe_expert_tracking_free(void);

#endif /* TN_MOE_FFN_H */
