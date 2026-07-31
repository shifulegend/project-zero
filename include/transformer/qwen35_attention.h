#ifndef TN_QWEN35_ATTENTION_H
#define TN_QWEN35_ATTENTION_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"

/**
 * Qwen3.5/3.6 hybrid attention forward pass — dispatches per-layer to either
 * the full-attention (GQA + QK-norm + partial-rotary + sigmoid output gate)
 * or linear-attention (Gated DeltaNet: causal conv1d + gated delta-rule
 * recurrence + SiLU output gate) sub-path, per q35_layer_is_full_attn().
 * Called from attention_forward() when mc->has_linear_attn is set.
 */
void qwen35_attention_forward(RunState *s, const TransformerWeights *w,
                               const Config *cfg, const MoEConfig *mc,
                               int layer, int pos, ThreadPool *tp);

#endif /* TN_QWEN35_ATTENTION_H */
