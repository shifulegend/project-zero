/**
 * mla_attention.c — Phase 17.8
 *
 * Multi-head Latent Attention (MLA) forward pass.
 *
 * Buffer order (avoids overflow; hb/hb2 are hidden_dim floats each):
 *   Step 1. RMSNorm → xb
 *   Step 2. KV compress:   kvc      = wkv_a @ xb  → hb2 (576 f)
 *              split: kv_latent = hb2[:lora], k_rope_cur = hb2[lora:]
 *   Step 3. KV expand:     kv_full  = wkv_b @ kv_latent → hb (4096 f)
 *              split: k_nope[kv_h][nope], v[kv_h][v_dim]  → caches
 *   Step 4. Q projection:  q_full   = wq   @ xb  → hb (3072 f) *overwrites* kv_full
 *              (safe: kv_full already stored to cache in step 3)
 *   Step 5. Apply RoPE to k_rope_cur (in hb2) and q_rope per head (in hb)
 *   Step 6. Store k_rope_cur → k_rope_cache
 *   Step 7. Attention: score[h][t] = q_nope·k_nope + q_rope·k_rope; softmax; V-sum → xb2
 *   Step 8. wo proj: xb = wo @ xb2;  residual: x += xb
 */

#include "transformer/mla_attention.h"
#include "math/parallel_matmul.h"
#include "math/matmul_q4k.h"
#include "math/simd_dispatch.h"
#include "core/run_state.h"
#include "core/debug.h"
#include "core/step_timing.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── YaRN RoPE helpers ─────────────────────────────────────────────────── */

/* Compute per-dimension ramp blend: 1=extrapolate (no scale), 0=interpolate (scaled).
 * i0 is the raw dimension index (0..n_dims-1), not half-index. */
static float rope_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / (high - low > 0.001f ? high - low : 0.001f);
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;  /* 1=extrapolate, 0=interpolate */
}

/* Compute YaRN correction dimension bounds for the blend zone.
 * corr[0]=low, corr[1]=high: dims below low are fully extrapolated (no scaling),
 * dims above high are fully interpolated (scaled by freq_scale).
 * Matches ggml_rope_yarn_corr_dims() from llama.cpp exactly. */
static void yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                            float beta_fast, float beta_slow, float corr[2]) {
    static const float M_PI_F = 3.14159265358979323846f;
    float log_base = logf(freq_base);
    float start = floorf((float)n_dims * logf((float)n_ctx_orig / (beta_fast * 2.0f * M_PI_F))
                         / (2.0f * log_base));
    float end   = ceilf ((float)n_dims * logf((float)n_ctx_orig / (beta_slow * 2.0f * M_PI_F))
                         / (2.0f * log_base));
    corr[0] = (start < 0.0f) ? 0.0f : (start > (float)(n_dims - 1) ? (float)(n_dims - 1) : start);
    corr[1] = (end   < 0.0f) ? 0.0f : (end   > (float)(n_dims - 1) ? (float)(n_dims - 1) : end);
}

/*
 * Apply NORMAL-type RoPE with full YaRN frequency interpolation.
 * NORMAL = interleaved pairs (v[2i], v[2i+1]) — NOT NEOX layout.
 *
 * YaRN blends theta per-dimension:
 *   - low-freq dims (i < corr[0]/2): use theta_extrap (no scaling)
 *   - high-freq dims (i > corr[1]/2): use theta_interp = freq_scale * theta_extrap
 *   - mid dims: linear blend
 * attn_factor scales the cos/sin amplitude: compensates for energy change from scaling.
 *
 * Parameters come from Config YaRN fields, set by config_from_gguf().
 */
static void rope_apply_yarn(float *v, const float *freq, int d, int pos,
                             float freq_scale, float ext_factor, float attn_factor,
                             const float corr[2]) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float theta_extrap = (float)pos * freq[i];          /* base angle */
        float theta_interp = freq_scale * theta_extrap;     /* scaled angle */

        float theta;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp = rope_yarn_ramp(corr[0], corr[1], 2 * i) * ext_factor;
            theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            /* energy compensation: mscale *= (1 + 0.1 * ln(1/freq_scale)) */
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        } else {
            theta = theta_interp;
        }

        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = v[2 * i];
        float x1 = v[2 * i + 1];
        v[2 * i]     = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

/* Dispatch helper for MLA projection matmuls.
 * Three paths, selected at runtime from weight flags set by gguf_loader:
 *   1. has_mla_quant=true  → zero-copy Q4_K kernel (best performance for GGUF)
 *   2. is_float=true       → F32 dequantized path  (fallback for non-Q4K GGUF)
 *   3. else                → packed ternary path   (native .bin models)
 * Adding a new quant type: add a flag to TransformerWeights, set in gguf_loader,
 * add a branch here. Never hardcode tensor names or dimension checks. */
#define MLA_MATMUL(out, in, wptr, sptr, n_in, n_out)                       \
    do {                                                                    \
        if (w->has_mla_quant) {                                             \
            parallel_matmul_q4k((out), (in),                               \
                (const uint8_t *)(wptr), (n_in), (n_out), tp);             \
        } else if (is_float) {                                              \
            parallel_matmul_float32((out), (in), (const float *)(wptr),    \
                                    (n_in), (n_out), tp);                   \
        } else {                                                            \
            parallel_ternary_matmul_packed((out), (in),                    \
                (const tn_u8 *)(wptr), (n_in), (n_out), (sptr), tp);      \
        }                                                                   \
    } while (0)

void mla_attention_forward(RunState *s, const TransformerWeights *w,
                            const Config *cfg, const MoEConfig *mc,
                            int layer, int pos, ThreadPool *tp) {
    const int dim      = cfg->dim;
    const int n_heads  = cfg->n_heads;
    const int n_kv_h   = cfg->n_kv_heads;
    const int nope     = mc->qk_nope_head_dim;
    const int rope     = mc->qk_rope_head_dim;
    const int lora     = mc->kv_lora_rank;
    const int v_dim    = mc->v_head_dim;
    const int max_seq  = s->max_seq_len;
    const int kv_mul   = n_heads / n_kv_h;
    const int head_dim = dim / n_heads;        /* KV cache stride */

    /* kq_scale: matches llama.cpp deepseek2.cpp formula.
     * mscale_kq = 1 + rope_yarn_log_mul * ln(1/freq_scale)
     *           = 1 + 0.0707 * ln(40) = 1.2608  (for DeepSeek-V2-Lite)
     * iscale = mscale_kq^2 / sqrt(nope + rope)
     * When YaRN is inactive (ext_factor==0 or log_mul==0), use standard 1/sqrt(d). */
    float iscale;
    {
        float mscale_kq = 1.0f;
        if (cfg->rope_yarn_ext_factor != 0.0f && cfg->rope_yarn_log_mul != 0.0f
                && cfg->rope_freq_scale < 1.0f) {
            /* freq_scale = 1/factor; ln(factor) = ln(1/freq_scale) = -ln(freq_scale) */
            mscale_kq = 1.0f + cfg->rope_yarn_log_mul * logf(1.0f / cfg->rope_freq_scale);
        }
        iscale = mscale_kq * mscale_kq / sqrtf((float)(nope + rope));
    }

    const int q_rows   = n_heads * (nope + rope);  /* 3072 */
    const int kva_rows = lora + rope;               /*  576 */
    const int kvb_rows = n_kv_h * (nope + v_dim);  /* 4096 */

    /* Dispatch to float32 or ternary matmul */
    const bool is_float = !w->layers_are_ternary;

    /* YaRN RoPE parameters from config */
    float corr[2];
    yarn_corr_dims(rope, cfg->rope_orig_ctx_len, cfg->rope_theta,
                   cfg->rope_yarn_beta_fast, cfg->rope_yarn_beta_slow, corr);
    float yarn_freq_scale  = cfg->rope_freq_scale;
    float yarn_ext_factor  = cfg->rope_yarn_ext_factor;
    float yarn_attn_factor = cfg->rope_yarn_attn_factor;

    int mapped_pos = sw_map_position(&s->sw, pos);

    /* Step 1: RMSNorm */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->rms_att_weight[layer], dim, cfg->rms_norm_eps);
    if (t_step) {
        tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM,
                           tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "attn_norm", s->xb, dim);

    /* Step 2: KV compress → hb2 (kva_rows ≤ hidden_dim) */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    MLA_MATMUL(s->hb2, s->xb, w->mla_wkv_a[layer], w->mla_skv_a[layer],
               dim, kva_rows);
    if (t_step) {
        tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION,
                           tn_step_timing_now_ns() - t_step);
    }
    float *kv_latent  = s->hb2;           /* hb2[0..lora-1]  */
    float *k_rope_cur = s->hb2 + lora;    /* hb2[lora..lora+rope-1] */

    /* Step 2b: Apply KV-A norm to kv_latent (before KV-B expansion) */
    if (w->rms_attn_sub_norm && w->rms_attn_sub_norm[layer]) {
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        tn_rmsnorm(kv_latent, kv_latent, w->rms_attn_sub_norm[layer], lora, cfg->rms_norm_eps);
        if (t_step) {
            tn_step_timing_add(TN_STEP_7_KV_A_LATENT_NORM,
                               tn_step_timing_now_ns() - t_step);
        }
    }
    DBG_DUMP(layer, "kv_cmpr", kv_latent, lora);

    /* Step 3: KV expand → hb (or heap buffer if kvb_rows > hidden_dim) */
    float *hb_buf = s->hb;
    float *malloc_hb = NULL;
    int max_rows_needed = (q_rows > kvb_rows) ? q_rows : kvb_rows;
    if (max_rows_needed > cfg->hidden_dim) {
        malloc_hb = (float *)malloc((size_t)max_rows_needed * sizeof(float));
        if (malloc_hb) hb_buf = malloc_hb;
    }

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    MLA_MATMUL(hb_buf, kv_latent, w->mla_wkv_b[layer], w->mla_skv_b[layer],
               lora, kvb_rows);
    if (t_step) {
        tn_step_timing_add(TN_STEP_8_KV_B_EXPANSION,
                           tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "kv", hb_buf, kvb_rows);
    /* Store k_nope and v to caches before hb is overwritten */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int kv_h = 0; kv_h < n_kv_h; kv_h++) {
        float *k_nope_src = hb_buf + kv_h * (nope + v_dim);
        float *v_src      = k_nope_src + nope;
        size_t k_off = KV_CACHE_IDX(layer, kv_h, mapped_pos, 0, n_kv_h, max_seq, head_dim);
        size_t v_off = KV_CACHE_IDX(layer, kv_h, mapped_pos, 0, n_kv_h, max_seq, head_dim);
        memcpy(&s->key_cache[k_off],   k_nope_src, (size_t)nope  * sizeof(float));
        memcpy(&s->value_cache[v_off], v_src,       (size_t)v_dim * sizeof(float));
    }
    if (t_step) {
        tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE,
                           tn_step_timing_now_ns() - t_step);
    }

    /* Step 4: Q projection → hb_buf (overwrites kv_full, already cached) */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    MLA_MATMUL(hb_buf, s->xb, w->mla_wq[layer], w->mla_sq[layer], dim, q_rows);
    if (t_step) {
        tn_step_timing_add(TN_STEP_5_Q_PROJECTION,
                           tn_step_timing_now_ns() - t_step);
    }
    float *q_full = hb_buf;   /* [n_heads][nope + rope] */
    DBG_DUMP(layer, "q", q_full, q_rows);

    /* Step 5: Apply RoPE to k_rope_cur (in hb2, shared) and q_rope per head (in hb).
     * CRITICAL: use mla_rope_freq (computed for rope_dim=64), NOT s->rope_freq
     * (which is computed for head_dim=128 and gives wrong frequencies).
     * Full YaRN per-dimension blending: low-freq dims use unscaled theta,
     * high-freq dims use scaled theta (freq_scale), blend zone in between. */
    const float *rf = s->mla_rope_freq;
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    rope_apply_yarn(k_rope_cur, rf, rope, pos, yarn_freq_scale, yarn_ext_factor,
                    yarn_attn_factor, corr);
    DBG_DUMP(layer, "k_pe", k_rope_cur, rope);
    for (int h = 0; h < n_heads; h++)
        rope_apply_yarn(q_full + h * (nope + rope) + nope, rf, rope, pos,
                        yarn_freq_scale, yarn_ext_factor, yarn_attn_factor, corr);
    if (t_step) {
        tn_step_timing_add(TN_STEP_9_YARN_ROPE,
                           tn_step_timing_now_ns() - t_step);
    }
    /* Dump q_pe: first head's rope portion */
    DBG_DUMP(layer, "q_pe", q_full + nope, rope);

    /* Step 6: Store k_rope_cur to k_rope_cache */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    memcpy(&s->k_rope_cache[layer][mapped_pos * rope], k_rope_cur, (size_t)rope * sizeof(float));
    if (t_step) {
        tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE,
                           tn_step_timing_now_ns() - t_step);
    }

    /* Step 7: Attention */
    sw_advance(&s->sw);
    int valid_ctx = sw_valid_count(&s->sw, pos);

    memset(s->xb2, 0, (size_t)(n_heads * v_dim) * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        float *q_nope_h   = q_full + h * (nope + rope);
        float *q_rope_h   = q_nope_h + nope;
        int    kv_h       = h / kv_mul;
        float *att        = s->att + h * max_seq;

        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t k_off = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, n_kv_h, max_seq, head_dim);
            float score  = tn_vec_dot(q_nope_h, &s->key_cache[k_off], nope);
            score += tn_vec_dot(q_rope_h, &s->k_rope_cache[layer][mapped_t * rope], rope);
            att[t] = score * iscale;
        }
        /* Step 11: dump pre-softmax attention scores for head 0 */
        if (h == 0) { DBG_DUMP(layer, "attn_score_h0", att, valid_ctx); }
        tn_softmax(att, valid_ctx);
        /* Step 11: dump post-softmax attention weights for head 0 */
        if (h == 0) { DBG_DUMP(layer, "attn_soft_h0", att, valid_ctx); }
        if (t_step) {
            tn_step_timing_add(TN_STEP_11_ATTN_SCORE,
                               tn_step_timing_now_ns() - t_step);
        }

        float *out_h = s->xb2 + h * v_dim;
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t v_off = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, n_kv_h, max_seq, head_dim);
            float  a     = att[t];
            float *v_vec = &s->value_cache[v_off];
            for (int d = 0; d < v_dim; d++) out_h[d] += a * v_vec[d];
        }
        if (t_step) {
            tn_step_timing_add(TN_STEP_12_POST_ATTN,
                               tn_step_timing_now_ns() - t_step);
        }
    }

    /* Step 12: V-weighted sum output (pre-W_o projection) */
    DBG_DUMP(layer, "attn_v_out", s->xb2, n_heads * v_dim);

    /* Step 8: wo projection → xb; residual */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    MLA_MATMUL(s->xb, s->xb2, w->wo[layer], w->so[layer], n_heads * v_dim, dim);
    /* Step 12: W_o projection output (pre-residual) */
    DBG_DUMP(layer, "attn_wo_out", s->xb, dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    if (t_step) {
        tn_step_timing_add(TN_STEP_12_POST_ATTN,
                           tn_step_timing_now_ns() - t_step);
    }
    /* Step 12: post-attention residual (x after += xb) */
    DBG_DUMP(layer, "attn_resid", s->x, dim);
    if (malloc_hb) free(malloc_hb);

    if (g_tn_verbose) {
        char tag[48];
        snprintf(tag, sizeof(tag), "L%02d mla-attn out x", layer);
        dbg_vec_stats(tag, s->x, dim);
    }
}
