#ifndef TN_FORWARD_H
#define TN_FORWARD_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/spec_scratch.h"
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

/**
 * Phase 18 (speculative decoding): batched multi-token forward pass.
 * Semantically equivalent to calling transformer_forward(tokens[i], pos+i,
 * ...) for i in [0, n_tokens) sequentially (the teacher-forcing property of
 * causal transformers), but streams each weight matrix from RAM once and
 * reuses it across all n_tokens candidate activations instead of reading it
 * n_tokens times — see src/transformer/attention.c's attention_forward_batch()
 * for the write-phase/read-phase design and its correctness argument.
 *
 * Returns TN_ERR_UNSUPPORTED (not a crash, not a silent slow fallback) when
 * mc->has_linear_attn is set — Qwen3.5/3.6 hybrid's Gated-DeltaNet layers
 * hold a genuinely recurrent state this batching design cannot satisfy.
 * speculative_generate() checks this once at startup so callers should not
 * normally hit this mid-round.
 *
 * @param tokens     n_tokens token IDs to embed and process
 * @param pos        Sequence position of tokens[0] (tokens[i] is at pos+i)
 * @param sb         N-token-wide scratch, sized for exactly n_tokens
 *                   (spec_batch_scratch_alloc())
 * @param logits_out Caller-provided buffer, n_tokens * cfg->vocab_size
 *                   floats — row i is the distribution predicting the
 *                   token after tokens[0..i].
 */
TernaryError transformer_forward_batch(const int *tokens, int pos, int n_tokens,
                                        const Config *cfg, const TransformerWeights *w,
                                        RunState *s, SpecBatchScratch *sb,
                                        const MoEConfig *mc, ThreadPool *tp,
                                        float *logits_out);

#endif /* TN_FORWARD_H */

