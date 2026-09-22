#ifndef TN_MLA_ATTENTION_H
#define TN_MLA_ATTENTION_H

/**
 * mla_attention.h — Phase 17.8
 *
 * Multi-head Latent Attention (MLA) forward pass for DeepSeek-style models.
 *
 * MLA decomposes Q/K/V via low-rank latent projections to reduce KV cache size:
 *
 *   q_full   = mla_wq   @ x     → split: q_nope (n_heads*nope) + q_rope (n_heads*rope)
 *   kvc      = mla_wkv_a @ x    → split: kv_latent (kv_lora_rank) + k_rope_shared (rope)
 *   kv_full  = mla_wkv_b @ kv_latent → split: k_nope (n_kv_heads*nope) + v (n_kv_heads*v_dim)
 *
 *   attn_score[h][t] = (q_nope[h] · k_nope[h][t] + q_rope[h] · k_rope[t]) / sqrt(nope+rope)
 *
 * k_rope is shared across all heads (applied once, not per-head).
 * Stored in s->k_rope_cache[layer][pos * qk_rope_head_dim].
 */

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/spec_scratch.h"
#include "threading/thread_pool.h"

/**
 * MLA attention forward pass for a single layer.
 * Only called when mc->has_mla == 1.
 * Writes result to s->x (residual-added in place).
 */
void mla_attention_forward(RunState *s, const TransformerWeights *w,
                            const Config *cfg, const MoEConfig *mc,
                            int layer, int pos, ThreadPool *tp);

/**
 * Phase 18 (speculative decoding): batched multi-token MLA attention.
 * See attention_forward_batch() (attention.h) for the general contract.
 * MLA's KV compression is per-position-independent (no cross-position
 * coupling beyond the ordinary causal attention read), so it batches the
 * same write-phase/read-phase way as the generic dense/GQA path.
 */
TernaryError mla_attention_forward_batch(RunState *s, SpecBatchScratch *sb,
                                          const TransformerWeights *w, const Config *cfg,
                                          const MoEConfig *mc, int layer, int pos,
                                          int n_tokens, ThreadPool *tp);

#endif /* TN_MLA_ATTENTION_H */
