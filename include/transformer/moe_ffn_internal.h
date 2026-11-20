#ifndef TN_TRANSFORMER_MOE_FFN_INTERNAL_H
#define TN_TRANSFORMER_MOE_FFN_INTERNAL_H

/* Shared by the MoE FFN files (moe_ffn.c, moe_ffn_legacy.c, moe_ffn_track.c); not public API. */

#include <stddef.h>
#include <stdint.h>

#include "core/config.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "math/matmul_q4k.h"
#include "runtime/run_state.h"
#include "threading/thread_pool.h"

/* Maximum experts supported without heap allocation (stack-allocated buffers). */
#define MOE_MAX_EXPERTS_STACK 512

/* Size of the stack buffers for router outputs (scores, indices). */
#define MOE_SCORE_BUF_SIZE MOE_MAX_EXPERTS_STACK

/* Counts one invocation of expert `e` in `layer`; no-op while tracking is off or out of range. */
void moe_track_hit(int layer, int e);

/* The expert FFN paths that run when the fused two-dispatch path (moe_experts.c) did not handle the
 * routed experts. Each accumulates sum(score * expert(xb)) into s->xb2. `xb_q8k` is s->xb
 * pre-quantised to Q8K, or NULL. */

/* Q4_K gate / up with any down type: three thread-pool dispatches for all selected experts. */
void moe_legacy_routed_batched(RunState *s, const TransformerWeights *w, const Config *cfg,
                               const MoEConfig *mc, int layer, const int *selected,
                               const float *scores, int top_k, const TnQ8KActBlock *xb_q8k,
                               ThreadPool *tp);

/* One expert after another: fused Q2_K, dequantise-to-F32, or ternary packed weights. */
void moe_legacy_routed_sequential(RunState *s, const TransformerWeights *w, const Config *cfg,
                                  const MoEConfig *mc, int layer, const int *selected,
                                  const float *scores, int top_k, ThreadPool *tp);

/* The always-active shared expert (DeepSeek-V2 style), weight 1.0, into s->xb2. */
void moe_legacy_shared(RunState *s, const TransformerWeights *w, const Config *cfg,
                       const MoEConfig *mc, int layer, const TnQ8KActBlock *xb_q8k, ThreadPool *tp);

#endif
