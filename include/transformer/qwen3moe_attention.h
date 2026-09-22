#ifndef TN_QWEN3MOE_ATTENTION_H
#define TN_QWEN3MOE_ATTENTION_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/spec_scratch.h"
#include "threading/thread_pool.h"

/**
 * Qwen3-MoE (e.g. Qwen3-30B-A3B) attention forward pass — plain GQA with
 * per-head RMSNorm on Q/K ("QK-norm") before RoPE, standard full rotary
 * embedding. Called from attention_forward() when mc->has_qk_norm is set.
 *
 * A dedicated function (rather than attention_forward()'s generic GQA
 * branch) is required because this arch's head_dim is independent of
 * dim/n_heads — see qwen3moe_attention.c's header comment and
 * qwen3moe_run_state_alloc() for the buffer-sizing rationale.
 */
void qwen3moe_attention_forward(RunState *s, const TransformerWeights *w,
                                 const Config *cfg, const MoEConfig *mc,
                                 int layer, int pos, ThreadPool *tp);

/**
 * Phase 18 (speculative decoding): batched multi-token Qwen3-MoE attention.
 * See attention_forward_batch() (attention.h) for the general contract.
 * QK-norm is a per-position in-place RMSNorm with no cross-position read,
 * so this arch batches the same write-phase/read-phase way as the generic
 * dense/GQA path -- it just needs its own N-wide Q/K/V scratch (allocated
 * locally, not from SpecBatchScratch) since this arch's q_width/kv_width
 * are independent of dim/hidden_dim and could overflow SpecBatchScratch's
 * generic buffers (same reason qwen3moe_attention_forward() cannot reuse
 * RunState's generic scratch — see this file's header comment).
 */
TernaryError qwen3moe_attention_forward_batch(RunState *s, SpecBatchScratch *sb,
                                               const TransformerWeights *w, const Config *cfg,
                                               const MoEConfig *mc, int layer, int pos,
                                               int n_tokens, ThreadPool *tp);

#endif /* TN_QWEN3MOE_ATTENTION_H */
