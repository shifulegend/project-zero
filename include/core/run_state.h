#ifndef TN_RUN_STATE_H
#define TN_RUN_STATE_H

#include "core/config.h"
#include "core/error.h"
#include "kv_cache/sliding_window.h"

/**
 * KV Cache Memory Layout (TRANSPOSED for cache-line efficiency):
 *   [layer][head][position][dim]
 *
 * This ensures all positions for a single head are contiguous,
 * enabling linear SIMD sweeps during attention score computation.
 */
#define KV_CACHE_IDX(layer, head, pos, d, n_kv_heads, max_seq, head_dim)       \
  ((size_t)(layer) * (n_kv_heads) * (max_seq) * (head_dim) +                   \
   (size_t)(head) * (max_seq) * (head_dim) + (size_t)(pos) * (head_dim) + (d))

typedef struct {
  /* Scratch vectors (dim) */
  float *x;   /* current activation */
  float *xb;  /* scratch after rmsnorm / attention output */
  float *xb2; /* scratch for residual branch */

  /* FFN hidden buffers (hidden_dim) */
  float *hb;  /* hidden state after gate + SiLU */
  float *hb2; /* hidden state after up projection */

  /* Attention per-head buffers */
  float *q; /* query vector (dim) */

  /* Attention scores: n_heads * max_seq_len */
  float *att;

  /* Output logits: vocab_size */
  float *logits;

  /* KV Cache — transposed: [layer][head][pos][head_dim] */
  float *key_cache;
  float *value_cache;

  /* Phase 17.7: MLA k_rope_cache — shared RoPE key cache for MLA models.
   * Layout: k_rope_cache[layer][pos * qk_rope_head_dim]
   * NULL for non-MLA models (has_mla == 0); allocated by mla_run_state_alloc(). */
  float **k_rope_cache;

  /* MLA RoPE frequency table computed for qk_rope_head_dim (NOT full head_dim).
   * freq[i] = 1/pow(rope_theta, 2i/qk_rope_head_dim) for i in [0, rope_dim/2).
   * NULL for non-MLA models. Allocated by mla_run_state_alloc(). */
  float *mla_rope_freq;

  /* Precomputed RoPE frequency table: freq[i] = 1/pow(10000, 2i/head_dim) */
  float *rope_freq;
  int rope_freq_len; /* head_dim / 2 */

  /* State tracking */
  int current_pos;
  int max_seq_len;

  /* Sliding Window state for circular KV cache mapping */
  SlidingWindow sw;
} RunState;

/**
 * Allocate all RunState buffers using 64-byte aligned allocation.
 * KV cache uses transposed [layer][head][pos][dim] layout.
 */
TernaryError run_state_alloc(RunState *s, const Config *cfg, int max_seq_len);

/**
 * Free all RunState buffers.
 */
void run_state_free(RunState *s);

/* ---- MLA extension (Phase 17.7) ----------------------------------------- */
#include "core/moe_config.h"

/**
 * Allocate k_rope_cache for MLA models.
 * Must be called after run_state_alloc() when mc->has_mla is true.
 * k_rope_cache[l] has (max_seq_len * mc->qk_rope_head_dim) floats.
 * No-op (returns TN_OK) when mc == NULL or mc->has_mla == 0.
 */
TernaryError mla_run_state_alloc(RunState *s, const Config *cfg,
                                  const MoEConfig *mc, int max_seq_len);

/**
 * Free k_rope_cache per-layer buffers.
 * Call before run_state_free() when has_mla was true at alloc time.
 */
void mla_run_state_free(RunState *s, int n_layers);

#endif /* TN_RUN_STATE_H */
