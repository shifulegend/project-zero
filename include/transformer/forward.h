#ifndef TN_FORWARD_H
#define TN_FORWARD_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"

/**
 * Full transformer forward pass for a single token.
 *
 * For dense models (mc == NULL or mc->is_moe == false):
 *   Runs standard attention + dense FFN for every layer. Zero overhead.
 *
 * For MoE models (mc->is_moe == true):
 *   Runs attention for every layer; routes each MoE layer to moe_ffn_forward().
 *   Dense layers (< mc->first_k_dense_replace) use the dense FFN.
 *
 * @param token  Token ID to process (-1 = skip embed, caller injected s->x)
 * @param pos    Sequence position
 * @param cfg    Model configuration
 * @param w      Transformer weights (const — not modified)
 * @param s      Run state (modified in-place)
 * @param mc     MoE configuration (NULL for dense models — BitNet, Llama)
 * @param tp     Thread pool (NULL for single-threaded)
 * @return       Pointer to s->logits (vocab_size floats)
 */
float *transformer_forward(int token, int pos, const Config *cfg,
                           const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, ThreadPool *tp);

#endif /* TN_FORWARD_H */

