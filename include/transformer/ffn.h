#ifndef TN_FFN_H
#define TN_FFN_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
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

#endif /* TN_FFN_H */

