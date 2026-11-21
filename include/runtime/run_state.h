#ifndef TN_RUN_STATE_H
#define TN_RUN_STATE_H

#include <stdbool.h>

#include "core/attn_config.h"
#include "core/config.h"
#include "core/error.h"
#include "core/lora.h"
#include "kv_cache/kv_strategy.h"
#include "kv_cache/sliding_window.h"

/**
 * KV Cache Memory Layout (TRANSPOSED for cache-line efficiency):
 *   [layer][head][position][dim]
 *
 * This ensures all positions for a single head are contiguous,
 * enabling linear SIMD sweeps during attention score computation.
 */
#define KV_CACHE_IDX(layer, head, pos, d, n_kv_heads, max_seq, head_dim)                           \
    ((size_t)(layer) * (n_kv_heads) * (max_seq) * (head_dim) +                                     \
     (size_t)(head) * (max_seq) * (head_dim) + (size_t)(pos) * (head_dim) + (d))

/* Row index of a (layer, kv_head, pos) entry in key_scale/value_scale. */
#define KV_SCALE_IDX(layer, head, pos, n_kv_heads, max_seq)                                        \
    (((size_t)(layer) * (n_kv_heads) + (size_t)(head)) * (max_seq) + (size_t)(pos))

/* Phase 17.7: MLA k_rope_cache -- shared RoPE key cache for MLA models and its frequency table
 * (computed for qk_rope_head_dim, NOT full head_dim). Both NULL for non-MLA models
 * (mla.enabled == 0); allocated by mla_run_state_alloc(). */
typedef struct {
    float **k_rope_cache; /* k_rope_cache[layer][pos * qk_rope_head_dim] */
    float *mla_rope_freq; /* freq[i] = 1/pow(rope_theta, 2i/qk_rope_head_dim) */
} RunMla;

/* Qwen3.5/3.6 hybrid attention state (see MoEConfig.has_linear_attn). NULL for all other models.
 * This arch's true attn_head_dim doesn't divide evenly from dim/n_heads, so it cannot reuse
 * key_cache/value_cache/rope_freq (sized with config_head_dim(cfg) == dim/n_heads). Allocated by
 * q35_run_state_alloc(). Per-layer arrays follow the same NULL-for-inapplicable-layer-type
 * convention as weights.h (key_cache[l] is NULL for linear-attention layers and vice versa). */
typedef struct {
    float **key_cache;   /* [n_layers] full-attn: [n_kv_heads][max_seq][attn_head_dim] */
    float **value_cache; /* [n_layers] full-attn: same shape */
    float **conv_state;  /* [n_layers] linear-attn: [conv_kernel-1][conv_dim] history */
    float **recur_state; /* [n_layers] linear-attn: [n_v_heads][state_size][state_size] */
    float *rope_freq;    /* [attn_rope_dim/2] shared partial-rotary freq table */
    /* Single-token attention scratch for both layer kinds (full: q|gate, k, v, concat; linear:
     * qkv, z, beta, alpha, decay, conv out, core out), sized from the model. Per RunState, so two
     * RunStates never share it. */
    float *scratch;
} RunQwen35;

/* Qwen3-MoE attention state (see MoEConfig.has_qk_norm). NULL for all other models. Same class of
 * fix as RunQwen35: this arch's head_dim is independent of dim/n_heads, so it cannot reuse
 * key_cache/value_cache/rope_freq. Allocated by qwen3moe_run_state_alloc(). */
typedef struct {
    float **key_cache;   /* [n_layers] [n_kv_heads][max_seq][attn_head_dim] */
    float **value_cache; /* [n_layers] same shape */
    float *rope_freq;    /* [attn_head_dim/2] full-rotary freq table */
    /* Single-token attention scratch: q [q_width] | k [kv_width] | v [kv_width] | concat [q_width],
     * q_width = n_heads*attn_head_dim, kv_width = n_kv_heads*attn_head_dim. Per RunState, so two
     * RunStates (server slots) never share it, and sized by the model, so no width cap. */
    float *scratch;
} RunQwen3Moe;

/* Plan 8.5: int8/int4 KV storage for the generic dense/GQA cache (--kv-type). When type is
 * quantized, key_cache/value_cache are NULL and key_cache_q/value_cache_q hold the same
 * KV_CACHE_IDX layout as int8, plus one scale set per (layer, kv_head, position) row at index
 * KV_SCALE_IDX. Phase 31: with --kv-offload DIR the generic KV buffers are carved out of one
 * disk-backed mapping (memory/file_backed.h) at offload_base; run_state_free() unmaps it. */
typedef struct {
    int type;
    int row_bytes;  /* quantized: data bytes per head_dim row (i8: hd, i4: hd/2) */
    int row_scales; /* quantized: F32 scales per row (i8: 1, i4: 2*hd/32); row r's data at
                       key_cache_q + r*row_bytes, scales at key_scale + r*row_scales,
                       r = KV_SCALE_IDX(...) */
    tn_i8 *key_cache_q;
    tn_i8 *value_cache_q;
    float *key_scale;
    float *value_scale;
    void *offload_base; /* NULL when not offloaded */
    size_t offload_bytes;
} RunKvStore;

/* 1 when the four quantized-KV buffers (data and scales of K and V) are all allocated. */
static inline int run_kv_store_complete(const RunKvStore *kvs) {
    return kvs->key_cache_q && kvs->value_cache_q && kvs->key_scale && kvs->value_scale;
}

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

    /* Phase 17.7: MLA RoPE state; NULL for non-MLA models (see RunMla). */
    RunMla mla;

    /* Precomputed RoPE frequency table: freq[i] = 1/pow(10000, 2i/head_dim) */
    float *rope_freq;
    int rope_freq_len; /* head_dim / 2 */

    /* Qwen3.5/3.6 hybrid attention state (see RunQwen35); all NULL for other models. */
    RunQwen35 q35;

    /* Qwen3-MoE attention state (see RunQwen3Moe); all NULL for other models. */
    RunQwen3Moe qwen3moe;

    /* State tracking */
    int current_pos;
    int max_seq_len;

    /* Sliding Window state for circular KV cache mapping */
    SlidingWindow sw;

    /* Phase 19: LoRA adapter active for this RunState's generation, or NULL
   * (the default -- zeroed by run_state_alloc_ex()'s memset). Not owned:
   * points at a LoRAWeights the caller loaded and outlives this RunState.
   * Read directly by attention_forward()/ffn_forward() (single-token
   * dense/GQA path only -- see docs/ai/decision-log.md for scope); every
   * other RunState in the process (RAG's embedder, vision prefill, the
   * draft model's own state, every test) simply never sets this and pays
   * zero cost. */
    const LoRAWeights *active_lora;
    AttnConfig attn; /* attention kernel for this state; defaults set by run_state_alloc_ex */
    /* Phase 22-SSM: Mamba recurrent state (MoEConfig.is_ssm), NULL otherwise.
   * Allocated by mamba_state_alloc() after run_state_alloc*(); released by
   * run_state_free(). */
    struct MambaState *mamba;
    /* Quantized and disk-offloaded KV storage (see RunKvStore). */
    RunKvStore kvs;
    /* Positions [0, kv_committed) of the generic KV cache are known resident
   * (run_state_kv_reserve); the buffers are committed lazily otherwise. */
    int kv_committed;
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
 * sizing. `kv_opt` (NULL = F32 in RAM) picks the generic cache's storage type and --kv-offload
 * directory. run_state_alloc() is a thin wrapper calling this with false and NULL.
 *
 * For Qwen3.5/3.6 hybrid models this buffer is 100% dead weight — the
 * qwen35 forward path never reads it (it uses its own correctly-sized
 * q35.key_cache/q35.value_cache instead, see q35_run_state_alloc) — but at
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
TernaryError run_state_alloc_ex(RunState *s, const Config *cfg, int max_seq_len, bool skip_kv_cache,
                                const KVStrategyOptions *kv_opt);

/**
 * Free all RunState buffers.
 */
void run_state_free(RunState *s);

/* Commit the generic KV cache rows for positions [0, n_pos) of every layer
 * and KV head (no-op for rows already committed, disk-backed KV, or NULL
 * caches). Contents are preserved. generate_ex() calls it with prompt +
 * max_tokens so decode never takes first-touch page faults (2026-10-19:
 * otherwise ~100 faults per token on SmolLM2-1.7B land in the decode loop;
 * the tok/s effect was within this VM's +-4% noise, but per-token latency
 * stays flat). RSS still only grows with the positions actually used. */
void run_state_kv_reserve(RunState *s, const Config *cfg, int n_pos);

/* Phase 23 (2026-10-22): give the generic KV cache's committed rows back to
 * the OS (they read back as zero) and reset kv_committed to 0 — an idle
 * parallel-serving slot's KV memory is reclaimed this way. Returns the
 * number of positions released (0 for disk-backed or NULL caches, or where
 * the platform cannot release memory). Contents are lost: callers must
 * invalidate whatever indexes them (the slot's prefix cache) first. */
int run_state_kv_release(RunState *s, const Config *cfg);

/* Phase 31: async read-ahead of layer `layer`'s KV rows [0, n_pos) when the
 * cache is disk-backed (no-op otherwise). Called one layer ahead. */
void run_state_kv_prefetch_layer(const RunState *s, const Config *cfg, int layer, int n_pos);

/* ---- MLA extension (Phase 17.7) ----------------------------------------- */
#include "core/moe_config.h"

/**
 * Allocate k_rope_cache for MLA models.
 * Must be called after run_state_alloc() when mc->has_mla is true.
 * k_rope_cache[l] has (max_seq_len * mc->qk_rope_head_dim) floats.
 * No-op (returns TN_OK) when mc == NULL or mc->has_mla == 0.
 */
TernaryError mla_run_state_alloc(RunState *s, const Config *cfg, const MoEConfig *mc,
                                 int max_seq_len);

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
TernaryError q35_run_state_alloc(RunState *s, const Config *cfg, const MoEConfig *mc,
                                 int max_seq_len);

/**
 * Free q35.key_cache/q35.value_cache/q35.conv_state/q35.recur_state/q35.rope_freq.
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
TernaryError qwen3moe_run_state_alloc(RunState *s, const Config *cfg, const MoEConfig *mc,
                                      int max_seq_len);

/**
 * Free qwen3moe.key_cache/qwen3moe.value_cache/qwen3moe.rope_freq.
 * Call before run_state_free() when has_qk_norm was true at alloc time.
 */
void qwen3moe_run_state_free(RunState *s, const Config *cfg);

#endif /* TN_RUN_STATE_H */
