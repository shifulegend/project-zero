#ifndef TN_MOE_FFN_H
#define TN_MOE_FFN_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"
#include "transformer/moe_scheduler.h"
#include <stdint.h>

typedef enum {
    TN_MOE_THREADING_ROWSPLIT_FUSED = 0, /* Fused layer-level row-split GEMV with prefetching (default) */
    TN_MOE_THREADING_ROWSPLIT       = 1, /* Sequential per-expert row-split GEMV */
    TN_MOE_THREADING_LEGACY         = 2  /* Batched multi-expert parallel matmul */
} TnMoeThreadingMode;

void moe_set_threading_mode(const char *mode_str);
TnMoeThreadingMode moe_get_threading_mode(void);
void moe_sort_selected_experts(int *selected_experts, float *selected_scores, int k);

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
