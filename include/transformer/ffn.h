#ifndef TN_FFN_H
#define TN_FFN_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/spec_scratch.h"
#include "threading/thread_pool.h"

/**
 * Feed-Forward Network forward pass for a single layer.
 *
 * Dense model (mc == NULL or mc->is_moe == false, or layer < first_k_dense_replace):
 *   Runs the standard SwiGLU FFN.
 *
 * MoE model (mc->is_moe == true and layer >= mc->first_k_dense_replace):
 *   Runs moe_ffn_forward() — gates, selects top-k experts, weighted sum.
 *
 * @param s      Run state (modified in-place: x, xb, xb2, hb, hb2)
 * @param w      Transformer weights
 * @param cfg    Model configuration
 * @param mc     MoE configuration (NULL or mc->is_moe=false for dense models)
 * @param layer  Layer index [0, n_layers)
 * @param tp     Thread pool for parallel matmul (NULL for single-threaded)
 */
void ffn_forward(RunState *s, const TransformerWeights *w,
                 const Config *cfg, const MoEConfig *mc,
                 int layer, ThreadPool *tp);

/**
 * Phase 18 (speculative decoding): batched multi-token FFN forward pass.
 * Dense (SwiGLU) layers are fully batched. MoE layers fall back to
 * n_tokens sequential calls to the existing single-token moe_ffn_forward()
 * (token->expert routing batching is deferred -- a materially harder,
 * separate piece of work; see docs/architecture/IMPLEMENTATION_PLAN.md
 * Phase 18), using RunState `s`'s own scratch as temporary per-token
 * workspace -- safe since s->x/xb/xb2/hb/hb2 are otherwise unused for the
 * duration of a batched round (the round's activation chain flows through
 * `sb`, not `s`).
 *
 * @param s        Run state — real KV cache/model-global state, and (MoE
 *                 layers only) reused as temporary single-token scratch.
 * @param sb       N-token-wide scratch (spec_batch_scratch_alloc()).
 * @param n_tokens Batch width; must match sb->n_tokens.
 */
TernaryError ffn_forward_batch(RunState *s, SpecBatchScratch *sb,
                                const TransformerWeights *w, const Config *cfg,
                                const MoEConfig *mc, int layer, int n_tokens, ThreadPool *tp);

#endif /* TN_FFN_H */

