#include "transformer/attention.h"
#include "transformer/mla_attention.h"
#include "transformer/qwen35_attention.h"
#include "math/parallel_matmul.h"
#include "math/matmul_f16.h"
#include "math/matmul_q2_0.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include "core/weights.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if TN_HAS_AVX512
#include <immintrin.h>
#endif

/* Maximum input dimension supported for layer-level preq stack buffer.
 * BitNet-b1.58-2B-4T: dim=2048. Guard against future larger models. */
#define ATTN_PREQ_BUF_SIZE 16384

/*
 * K-3 R-3: Non-temporal store helper for KV cache writes.
 *
 * KV cache arrays are allocated with TN_SIMD_ALIGN=64 byte alignment.
 * head_dim is always a multiple of 16 floats (64 bytes) for this model
 * (head_dim=128 → 512 bytes). Both conditions make _mm512_stream_ps safe.
 *
 * Non-temporal stores bypass the L3 cache write-back buffer, preventing
 * KV cache data from evicting weight rows that are needed for the next
 * layer. This is critical when the model (1.1 GB) dwarfs L3 (12 MB).
 * An _mm_sfence() after the loop ensures the stores reach DRAM ordering.
 */
static void kv_nt_store(float *dst, const float *src, int n) {
#if TN_HAS_AVX512
    int i = 0;
    while (i < n && ((uintptr_t)(dst + i) & 63) != 0) {
        dst[i] = src[i];
        i++;
    }
    int nt_start = i;
    for (; i + 15 < n; i += 16)
        _mm512_stream_ps(dst + i, _mm512_loadu_ps(src + i));
    if (i > nt_start)
        _mm_sfence();
    for (; i < n; i++) dst[i] = src[i];
#else
    memcpy(dst, src, (size_t)n * sizeof(float));
#endif
}

void attention_forward(RunState *s, const TransformerWeights *w,
                       const Config *cfg, const MoEConfig *mc,
                       int layer, int pos, ThreadPool *tp) {
  /* Phase 17.8: Route to MLA forward pass when has_mla=1.
   * This guard is a single pointer deref + int compare — branch predictor handles it
   * in 0-1 cycles for all non-MLA models (BitNet, Llama, Mixtral). */
  if (mc && mc->has_mla) {
    mla_attention_forward(s, w, cfg, mc, layer, pos, tp);
    return;
  }

  /* Qwen3.5/3.6 hybrid attention (Gated DeltaNet + Gated Attention) — a
   * third, mutually-exclusive attention family alongside MLA above. */
  if (mc && mc->has_linear_attn) {
    qwen35_attention_forward(s, w, cfg, mc, layer, pos, tp);
    return;
  }

  int dim = cfg->dim;
  int n_heads = cfg->n_heads;
  int n_kv_heads = cfg->n_kv_heads;
  int head_dim = config_head_dim(cfg);
  int kv_dim = config_kv_dim(cfg);
  int max_seq = s->max_seq_len;
  int kv_mul = n_heads / n_kv_heads; /* GQA multiplier */

  /* Map logical position to physical circular buffer slot */
  int mapped_pos = sw_map_position(&s->sw, pos);

  /* Step 1: RMSNorm — normalize s->x into s->xb */
  tn_rmsnorm(s->xb, s->x, w->rms_att_weight[layer], dim, cfg->rms_norm_eps);

  /* Temp buffers for K and V (reuse RunState scratch buffers):
   * k_buf: kv_dim floats — xb2 is dim floats, kv_dim <= dim, safe
   * v_buf: kv_dim floats — hb2 is hidden_dim floats, >= kv_dim, safe */
  float *k_buf = s->xb2;
  float *v_buf = s->hb2;

  /* Step 2: Compute Q, K, V projections
   * Layer-level preq: quantise s->xb once, reuse for all three projections.
   * Saves 2 redundant quantisations per attention layer (3 calls → 1 quantise). */
  if (w->layers_are_ternary) {
    int8_t stack_buf[ATTN_PREQ_BUF_SIZE];
    int8_t *preq_buf = stack_buf;
    if (dim > ATTN_PREQ_BUF_SIZE) {
      preq_buf = (int8_t *)malloc((size_t)dim * sizeof(int8_t));
    }
    TnPreqActivation preq;
    tn_preq_prepare(&preq, preq_buf, s->xb, dim);
    parallel_ternary_matmul_packed_preq(s->q,  s->xb, (const tn_u8 *)w->wq[layer], dim, dim,    w->sq[layer], &preq, tp);
    parallel_ternary_matmul_packed_preq(k_buf, s->xb, (const tn_u8 *)w->wk[layer], dim, kv_dim, w->sk[layer], &preq, tp);
    parallel_ternary_matmul_packed_preq(v_buf, s->xb, (const tn_u8 *)w->wv[layer], dim, kv_dim, w->sv[layer], &preq, tp);
    if (preq_buf != stack_buf) {
      free(preq_buf);
    }
  } else if (w->layer_weight_type == WEIGHT_TYPE_F16) {
    parallel_matmul_f16(s->q,  s->xb, (const tn_u16 *)w->wq[layer], dim, dim,    tp);
    parallel_matmul_f16(k_buf, s->xb, (const tn_u16 *)w->wk[layer], dim, kv_dim, tp);
    parallel_matmul_f16(v_buf, s->xb, (const tn_u16 *)w->wv[layer], dim, kv_dim, tp);
  } else if (w->layer_weight_type == WEIGHT_TYPE_Q2_0) {
    parallel_matmul_q2_0(s->q,  s->xb, (const uint8_t *)w->wq[layer], dim, dim,    tp);
    parallel_matmul_q2_0(k_buf, s->xb, (const uint8_t *)w->wk[layer], dim, kv_dim, tp);
    parallel_matmul_q2_0(v_buf, s->xb, (const uint8_t *)w->wv[layer], dim, kv_dim, tp);
  } else {
    parallel_matmul_float32(s->q,  s->xb, (const float *)w->wq[layer], dim, dim,    tp);
    parallel_matmul_float32(k_buf, s->xb, (const float *)w->wk[layer], dim, kv_dim, tp);
    parallel_matmul_float32(v_buf, s->xb, (const float *)w->wv[layer], dim, kv_dim, tp);
  }

  /* Step 3: Apply RoPE to Q and K with full YaRN if configured */
  {
    float corr[2] = {0.0f, 0.0f};
    float freq_scale  = cfg->rope_freq_scale;
    float ext_factor  = cfg->rope_yarn_ext_factor;
    float attn_factor = cfg->rope_yarn_attn_factor;
    if (ext_factor != 0.0f) {
        /* Compute per-dimension YaRN blend boundaries */
        static const float M_PI_F = 3.14159265358979323846f;
        float log_base = logf(cfg->rope_theta);
        float start = floorf((float)head_dim * logf((float)cfg->rope_orig_ctx_len
                             / (cfg->rope_yarn_beta_fast * 2.0f * M_PI_F)) / (2.0f * log_base));
        float end   = ceilf ((float)head_dim * logf((float)cfg->rope_orig_ctx_len
                             / (cfg->rope_yarn_beta_slow * 2.0f * M_PI_F)) / (2.0f * log_base));
        corr[0] = start < 0.0f ? 0.0f : (start > (float)(head_dim-1) ? (float)(head_dim-1) : start);
        corr[1] = end   < 0.0f ? 0.0f : (end   > (float)(head_dim-1) ? (float)(head_dim-1) : end  );
    }
    apply_rope(s->q, k_buf, s->rope_freq, head_dim, pos, n_heads, n_kv_heads,
               freq_scale, ext_factor, attn_factor, corr);
  }

  /* Step 4: Store K and V into the transposed KV cache at
   * [layer][kv_head][mapped_pos][:] using non-temporal stores (K-3 R-3).
   * NT stores prevent KV writes from evicting weight data from L3 cache. */
  for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
    size_t cache_offset =
        KV_CACHE_IDX(layer, kv_h, mapped_pos, 0, n_kv_heads, max_seq, head_dim);
    kv_nt_store(&s->key_cache[cache_offset],   &k_buf[kv_h * head_dim], head_dim);
    kv_nt_store(&s->value_cache[cache_offset], &v_buf[kv_h * head_dim], head_dim);
  }

  /* Advance the sliding window write head */
  sw_advance(&s->sw);

  /* Step 5: Compute attention for each query head */
  float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);

  for (int h = 0; h < n_heads; h++) {
    /* Pointer to this head's query vector */
    float *q_head = s->q + h * head_dim;

    /* Resolve which KV head this query head maps to (GQA) */
    int kv_h = h / kv_mul;

    /* Number of valid context tokens to attend to */
    int valid_ctx = sw_valid_count(&s->sw, pos);

    /* Pointer to this head's attention scores (valid_ctx entries) */
    float *att_scores = s->att + h * max_seq;

    /* Step 5a: Compute attention scores: score[t] = Q · K[t] / sqrt(head_dim)
     * We iterate through logical history and map to physical slots */
    for (int t = 0; t < valid_ctx; t++) {
      /* Reconstruct historical logical position.
       * Preserves system prompt pinned slots [0 .. system_prompt_len - 1] */
      int spl = s->sw.system_prompt_len;
      int hist_logical = (t < spl) ? t : (pos - (valid_ctx - 1 - t));
      int mapped_t = sw_map_position(&s->sw, hist_logical);

      size_t k_offset =
          KV_CACHE_IDX(layer, kv_h, mapped_t, 0, n_kv_heads, max_seq, head_dim);
      float *k_vec = &s->key_cache[k_offset];

      /* Dot product — use SIMD-dispatched vec_dot for speed */
      float score = tn_vec_dot(q_head, k_vec, head_dim);
      att_scores[t] = score * inv_sqrt_head_dim;
    }

    /* Step 5b: Softmax over attention scores [0..valid_ctx-1] */
    tn_softmax(att_scores, valid_ctx);

    /* Step 5c: Weighted sum of Value vectors → s->xb[h * head_dim ..
     * (h+1)*head_dim) */
    float *out_head = s->xb + h * head_dim;
    memset(out_head, 0, head_dim * sizeof(float));

    for (int t = 0; t < valid_ctx; t++) {
      /* Map to physical slot — must match the key lookup in Step 5a. */
      int spl = s->sw.system_prompt_len;
      int v_hist = (t < spl) ? t : (pos - (valid_ctx - 1 - t));
      int v_mapped = sw_map_position(&s->sw, v_hist);

      size_t v_offset =
          KV_CACHE_IDX(layer, kv_h, v_mapped, 0, n_kv_heads, max_seq, head_dim);
      float *v_vec = &s->value_cache[v_offset];
      float a = att_scores[t];
      tn_vec_saxpy(out_head, a, v_vec, head_dim);
    }
  }

  /* Step 6: Apply attn_sub_norm (BitNet) if present */
  if (w->rms_attn_sub_norm && w->rms_attn_sub_norm[layer]) {
    tn_rmsnorm(s->xb, s->xb, w->rms_attn_sub_norm[layer], dim, cfg->rms_norm_eps);
  }

  /* Step 7: Output projection — s->xb (concatenated head outputs) → s->xb2 */
  if (w->layers_are_ternary) {
    parallel_ternary_matmul_packed(s->xb2, s->xb, (const tn_u8 *)w->wo[layer], dim, dim, w->so[layer], tp);
  } else if (w->layer_weight_type == WEIGHT_TYPE_F16) {
    parallel_matmul_f16(s->xb2, s->xb, (const tn_u16 *)w->wo[layer], dim, dim, tp);
  } else if (w->layer_weight_type == WEIGHT_TYPE_Q2_0) {
    parallel_matmul_q2_0(s->xb2, s->xb, (const uint8_t *)w->wo[layer], dim, dim, tp);
  } else {
    parallel_matmul_float32(s->xb2, s->xb, (const float *)w->wo[layer], dim, dim, tp);
  }


  /* Step 8: Residual connection — s->x += s->xb2 */
  tn_vec_add(s->x, s->x, s->xb2, dim);
}
