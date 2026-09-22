#ifndef TN_ATTENTION_H
#define TN_ATTENTION_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/spec_scratch.h"
#include "threading/thread_pool.h"

/**
 * Multi-Head Attention forward pass for a single layer.
 *
 * When mc != NULL && mc->has_mla is true, routes to mla_attention_forward()
 * for Multi-head Latent Attention (MLA) layers (Phase 17.8).
 * For all dense and standard MoE models, mc is NULL or has_mla==0 and the
 * standard path below executes unchanged.
 *
 * Steps (standard path):
 *   1. RMSNorm input (s->x → s->xb)
 *   2. Compute Q, K, V projections via ternary matmul
 *   3. Apply RoPE to Q and K
 *   4. Store K, V into the transposed KV cache
 *   5. For each attention head:
 *      a. Compute attention scores (Q · K / sqrt(head_dim))
 *      b. Softmax over scores
 *      c. Weighted sum of V → output
 *   6. Output projection matmul
 *   7. Residual connection: s->x += output
 *
 * Supports Grouped Query Attention (GQA) when n_kv_heads < n_heads.
 */
void attention_forward(RunState *s, const TransformerWeights *w,
                       const Config *cfg, const MoEConfig *mc,
                       int layer, int pos, ThreadPool *tp);

/**
 * Phase 18 (speculative decoding): batched multi-token attention forward
 * pass. Semantically equivalent to calling attention_forward() once per
 * token at positions pos..pos+n_tokens-1 in sequence (teacher-forcing
 * property of causal transformers), but streams each weight matrix from
 * RAM once and reuses it across all n_tokens candidate activations.
 *
 * Dispatches to mla_attention_forward_batch()/qwen3moe_attention_forward_batch()
 * exactly like attention_forward() dispatches to the single-token versions.
 * Returns TN_ERR_UNSUPPORTED (not a crash, not a silent slow fallback) when
 * mc->has_linear_attn is set — see attention.c's attention_forward_batch()
 * header comment for why Qwen3.5/3.6 hybrid models cannot be batched this
 * way.
 *
 * sb must be sized for exactly n_tokens (spec_batch_scratch_alloc()).
 */
TernaryError attention_forward_batch(RunState *s, SpecBatchScratch *sb,
                                      const TransformerWeights *w, const Config *cfg,
                                      const MoEConfig *mc, int layer, int pos,
                                      int n_tokens, ThreadPool *tp);

#endif /* TN_ATTENTION_H */
