#include "core/run_state.h"
#include "core/platform.h"
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include "kv_cache/kv_strategy.h"  /* tn_get_free_ram() for the OOM sanity guard */
#include <string.h>
#include <stdlib.h>   /* free() for k_rope_cache pointer array */

/* Deterministic OOM trap: on some platforms (notably macOS) calloc over-commits
 * for absurd sizes and the process is OOM-killed instead of receiving NULL.
 * Reject up front when a single buffer would dwarf available RAM. The 32x
 * headroom is far beyond any runnable configuration, so this never rejects a
 * legitimate allocation — it only traps pathological requests (e.g. INT_MAX
 * context) the same way malloc-returns-NULL already does on Linux. */
static int tn_alloc_too_large(size_t count, size_t elem_size) {
  tn_i64 ram = tn_get_free_ram();
  size_t bytes;
  if (ram <= 0) return 0; /* unknown RAM: fall back to malloc-NULL behavior */
  if (tn_size_mul_overflow(count, elem_size, &bytes)) return 1;
  return bytes > (size_t)ram * 32;
}

TernaryError run_state_alloc(RunState *s, const Config *cfg, int max_seq_len) {
  return run_state_alloc_ex(s, cfg, max_seq_len, false);
}

TernaryError run_state_alloc_ex(RunState *s, const Config *cfg, int max_seq_len,
                                 bool skip_kv_cache) {
  memset(s, 0, sizeof(*s));
  s->max_seq_len = max_seq_len;
  s->current_pos = 0;

  int dim = cfg->dim;
  int hidden_dim = cfg->hidden_dim;
  int kv_dim = config_kv_dim(cfg);
  int head_dim = config_head_dim(cfg);

  /* Scratch buffers — 64-byte aligned for SIMD */
  s->x = (float *)tn_aligned_calloc(dim, sizeof(float), TN_SIMD_ALIGN);
  s->xb = (float *)tn_aligned_calloc(dim, sizeof(float), TN_SIMD_ALIGN);
  s->xb2 = (float *)tn_aligned_calloc(dim, sizeof(float), TN_SIMD_ALIGN);
  s->hb = (float *)tn_aligned_calloc(hidden_dim, sizeof(float), TN_SIMD_ALIGN);
  s->hb2 = (float *)tn_aligned_calloc(hidden_dim, sizeof(float), TN_SIMD_ALIGN);
  s->q = (float *)tn_aligned_calloc(dim, sizeof(float), TN_SIMD_ALIGN);

  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q) {
    run_state_free(s);
    return TN_ERR_OOM;
  }

  /* FM-005 fix: overflow-checked multiplication for att buffer */
  size_t att_count;
  if (tn_size_mul_overflow((size_t)cfg->n_heads, (size_t)max_seq_len,
                           &att_count)) {
    run_state_free(s);
    return TN_ERR_OOM;
  }
  if (tn_alloc_too_large(att_count, sizeof(float))) {
    run_state_free(s);
    return TN_ERR_OOM;
  }
  s->att = (float *)tn_aligned_calloc(att_count, sizeof(float), TN_SIMD_ALIGN);
  s->logits =
      (float *)tn_aligned_calloc(cfg->vocab_size, sizeof(float), TN_SIMD_ALIGN);

  if (!s->att || !s->logits) {
    run_state_free(s);
    return TN_ERR_OOM;
  }

  /* KV Cache — transposed layout: [layer][head][pos][head_dim]
   * FM-002 fix: overflow-checked multiplication for all 4 factors.
   * Skipped entirely (not calloc'd, not calloc'd-then-freed — see
   * run_state_alloc_ex's header comment) when skip_kv_cache is set: the
   * calloc's explicit memset would force real RSS commit for the whole
   * buffer even if freed moments later. */
  if (!skip_kv_cache) {
    size_t kv_cache_size;
    if (tn_size_mul4((size_t)cfg->n_layers, (size_t)cfg->n_kv_heads,
                     (size_t)max_seq_len, (size_t)head_dim, &kv_cache_size)) {
      run_state_free(s);
      return TN_ERR_OOM;
    }
    if (tn_alloc_too_large(kv_cache_size, sizeof(float))) {
      run_state_free(s);
      return TN_ERR_OOM;
    }
    s->key_cache =
        (float *)tn_aligned_calloc(kv_cache_size, sizeof(float), TN_SIMD_ALIGN);
    s->value_cache =
        (float *)tn_aligned_calloc(kv_cache_size, sizeof(float), TN_SIMD_ALIGN);

    if (!s->key_cache || !s->value_cache) {
      run_state_free(s);
      return TN_ERR_OOM;
    }
  }

  /* Precompute RoPE frequency table — eliminates powf() from the hot loop */
  s->rope_freq_len = head_dim / 2;
  s->rope_freq = (float *)tn_aligned_calloc(s->rope_freq_len, sizeof(float),
                                            TN_SIMD_ALIGN);
  if (!s->rope_freq) {
    run_state_free(s);
    return TN_ERR_OOM;
  }
  float theta = cfg->rope_theta > 0 ? cfg->rope_theta : 10000.0f;
  rope_precompute_freqs(s->rope_freq, head_dim, theta);

  (void)kv_dim; /* used implicitly via n_kv_heads * head_dim */
  return TN_OK;
}

void run_state_free(RunState *s) {
  tn_aligned_free(s->x);
  tn_aligned_free(s->xb);
  tn_aligned_free(s->xb2);
  tn_aligned_free(s->hb);
  tn_aligned_free(s->hb2);
  tn_aligned_free(s->q);
  tn_aligned_free(s->att);
  tn_aligned_free(s->logits);
  tn_aligned_free(s->key_cache);
  tn_aligned_free(s->value_cache);
  tn_aligned_free(s->rope_freq);
  tn_aligned_free(s->mla_rope_freq);
  /* k_rope_cache is allocated per-layer by mla_run_state_alloc(); free if present */
  if (s->k_rope_cache) {
    /* number of layers is unknown here; mla_run_state_alloc stores layers in cache[0..n-1].
     * We free the pointer array itself — the per-layer buffers are freed by
     * mla_run_state_free() which must be called before run_state_free(). */
    free(s->k_rope_cache);
    s->k_rope_cache = NULL;
  }
  memset(s, 0, sizeof(*s));
}
