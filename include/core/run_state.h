#ifndef TN_RUN_STATE_H
#define TN_RUN_STATE_H

#include "core/config.h"
#include "core/error.h"
#include "kv_cache/sliding_window.h"
#include <stdbool.h>

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

  /* Qwen3.5/3.6 hybrid attention state (see MoEConfig.has_linear_attn).
   * NULL for all other models. This arch's true attn_head_dim doesn't
   * divide evenly from dim/n_heads (unlike MLA's DeepSeek-Lite case), so
   * it CANNOT reuse key_cache/value_cache/rope_freq above — those are sized
   * with config_head_dim(cfg)==dim/n_heads, which is wrong here (truncates
   * to a non-integer). Allocated by q35_run_state_alloc(). Per-layer arrays
   * follow the same NULL-for-inapplicable-layer-type convention as weights.h
   * (q35_key_cache[l] is NULL for linear-attention layers and vice versa). */
  float **q35_key_cache;    /* [n_layers] full-attn: [n_kv_heads][max_seq][attn_head_dim] */
  float **q35_value_cache;  /* [n_layers] full-attn: same shape */
  float **q35_conv_state;   /* [n_layers] linear-attn: [conv_kernel-1][conv_dim] history */
  float **q35_recur_state;  /* [n_layers] linear-attn: [n_v_heads][state_size][state_size] */
  float  *q35_rope_freq;    /* [attn_rope_dim/2] shared partial-rotary freq table */

  /* Qwen3-MoE attention state (see MoEConfig.has_qk_norm). NULL for all
   * other models. Same class of fix as q35_key_cache above: this arch's
   * head_dim is independent of dim/n_heads, so it cannot reuse
   * key_cache/value_cache/rope_freq. Allocated by qwen3moe_run_state_alloc(). */
  float **qwen3moe_key_cache;    /* [n_layers] [n_kv_heads][max_seq][attn_head_dim] */
  float **qwen3moe_value_cache;  /* [n_layers] same shape */
  float  *qwen3moe_rope_freq;    /* [attn_head_dim/2] full-rotary freq table */

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
 * Same as run_state_alloc(), but when skip_kv_cache is true, key_cache/
 * value_cache are left NULL (not calloc'd at all — not a calloc-then-free,
 * see below) instead of their usual [n_layers][n_kv_heads][max_seq][head_dim]
 * sizing. run_state_alloc() is a thin wrapper calling this with false, so
 * every existing caller is unaffected.
 *
 * For Qwen3.5/3.6 hybrid models this buffer is 100% dead weight — the
 * qwen35 forward path never reads it (it uses its own correctly-sized
 * q35_key_cache/q35_value_cache instead, see q35_run_state_alloc) — but at
 * this arch's native 262144 context the generic sizing alone reaches
 * several GB. Freeing it immediately after allocating (the first fix
 * attempted here) still isn't enough: tn_aligned_calloc's explicit
 * `memset(ptr, 0, total)` forces the OS to commit real physical pages for
 * the *entire* buffer immediately (unlike libc calloc's lazy zero-page
 * optimization for large anonymous allocations), so the RSS spike happens
 * during the allocation itself, not in the (brief) window before freeing —
 * confirmed by a real near-OOM during verification, see docs/ai/mistakes.md.
 * Skipping the calloc entirely is the only way to avoid the spike.
 */
TernaryError run_state_alloc_ex(RunState *s, const Config *cfg, int max_seq_len,
                                 bool skip_kv_cache);

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

/* ---- Qwen3.5/3.6 hybrid attention extension ------------------------------ */

/**
 * Allocate per-layer KV cache (full-attention layers) and conv/recurrent
 * state (linear-attention layers) for Qwen3.5/3.6 hybrid models.
 * Must be called after run_state_alloc() when mc->has_linear_attn is true.
 * No-op (returns TN_OK) when mc == NULL or mc->has_linear_attn == 0.
 */
TernaryError q35_run_state_alloc(RunState *s, const Config *cfg,
                                  const MoEConfig *mc, int max_seq_len);

/**
 * Free q35_key_cache/q35_value_cache/q35_conv_state/q35_recur_state/q35_rope_freq.
 * Call before run_state_free() when has_linear_attn was true at alloc time.
 */
void q35_run_state_free(RunState *s, const Config *cfg, const MoEConfig *mc);

/* ---- Qwen3-MoE attention extension --------------------------------------- */

/**
 * Allocate per-layer KV cache and RoPE frequency table for Qwen3-MoE models,
 * correctly sized with mc->attn_head_dim (independent of dim/n_heads).
 * Must be called after run_state_alloc() when mc->has_qk_norm is true.
 * No-op (returns TN_OK) when mc == NULL or mc->has_qk_norm == 0.
 */
TernaryError qwen3moe_run_state_alloc(RunState *s, const Config *cfg,
                                       const MoEConfig *mc, int max_seq_len);

/**
 * Free qwen3moe_key_cache/qwen3moe_value_cache/qwen3moe_rope_freq.
 * Call before run_state_free() when has_qk_norm was true at alloc time.
 */
void qwen3moe_run_state_free(RunState *s, const Config *cfg);

#endif /* TN_RUN_STATE_H */
