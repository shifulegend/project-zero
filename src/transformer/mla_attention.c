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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/clamp.h"
#include "core/debug.h"
#include "core/step_timing.h"
#include "math/batched_matmul.h"
#include "math/matmul_q4k.h"
#include "math/parallel_matmul.h"
#include "math/rope.h"
#include "math/simd_core_ops.h"
#include "math/simd_vec_ops.h"
#include "runtime/spec_scratch.h" /* reads the batch scratch fields */

/* ── YaRN RoPE helpers ─────────────────────────────────────────────────── */

/* Compute per-dimension ramp blend: 1=extrapolate (no scale), 0=interpolate (scaled).
 * i0 is the raw dimension index (0..n_dims-1), not half-index. */
static float rope_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / (high - low > 0.001f ? high - low : 0.001f);
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y; /* 1=extrapolate, 0=interpolate */
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
static void rope_apply_yarn(float *v, const float *freq, int d, int pos, float freq_scale,
                            float ext_factor, float attn_factor, const float corr[2]) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float theta_extrap = (float)pos * freq[i];      /* base angle */
        float theta_interp = freq_scale * theta_extrap; /* scaled angle */

        float theta;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp = rope_yarn_ramp(corr[0], corr[1], 2 * i) * ext_factor;
            theta      = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            /* energy compensation: mscale *= (1 + 0.1 * ln(1/freq_scale)) */
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        } else {
            theta = theta_interp;
        }

        float c      = cosf(theta) * mscale;
        float s      = sinf(theta) * mscale;
        float x0     = v[2 * i];
        float x1     = v[2 * i + 1];
        v[2 * i]     = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

/* Dispatch helper for MLA projection matmuls.
 * Three paths, selected at runtime from weight flags set by gguf_loader:
 *   1. has_mla_quant=true  -> zero-copy Q4_K kernel (best performance for GGUF)
 *   2. layers not ternary  -> F32 dequantized path  (fallback for non-Q4K GGUF)
 *   3. else                -> packed ternary path   (native .bin models)
 * Adding a new quant type: add a flag to TransformerWeights, set in gguf_loader,
 * add a branch here. Never hardcode tensor names or dimension checks. */
static void mla_matmul(const TransformerWeights *w, ThreadPool *tp, float *out, const float *in,
                       const void *wptr, float sptr, int n_in, int n_out) {
    if (w->mla.has_quant)
        parallel_matmul_q4k(out, in, (const uint8_t *)wptr, n_in, n_out, tp);
    else if (!w->layers_are_ternary)
        parallel_matmul_float32(out, in, (const float *)wptr, n_in, n_out, tp);
    else
        parallel_ternary_matmul_packed(out, in, (const tn_u8 *)wptr, n_in, n_out, sptr, tp);
}

/* kq_scale: matches llama.cpp deepseek2.cpp formula.
 *   mscale_kq = 1 + rope_yarn_log_mul * ln(1/freq_scale)
 *             = 1 + 0.0707 * ln(40) = 1.2608  (for DeepSeek-V2-Lite)
 *   iscale = mscale_kq^2 / sqrt(nope + rope)
 * When YaRN is inactive (ext_factor==0 or log_mul==0), use standard 1/sqrt(d). */
static float mla_kq_scale(const Config *cfg, int nope, int rope) {
    float mscale_kq = 1.0f;
    if (cfg->rope_yarn_ext_factor != 0.0f && cfg->rope_yarn_log_mul != 0.0f &&
        cfg->rope_freq_scale < 1.0f) {
        /* freq_scale = 1/factor; ln(factor) = ln(1/freq_scale) = -ln(freq_scale) */
        mscale_kq = 1.0f + cfg->rope_yarn_log_mul * logf(1.0f / cfg->rope_freq_scale);
    }
    return mscale_kq * mscale_kq / sqrtf((float)(nope + rope));
}

/* This layer's shapes, from the config and the MLA metadata. */
typedef struct {
    int dim, n_heads, n_kv_h, nope, rope, lora, v_dim, max_seq, kv_mul;
    int head_dim; /* KV cache stride */
    int q_rows, kva_rows, kvb_rows;
    float iscale;
    float corr[2]; /* YaRN blend bounds for the rope dims */
} MlaShape;

static MlaShape mla_dims(const RunState *s, const Config *cfg, const MoEConfig *mc) {
    MlaShape d;
    d.dim      = cfg->dim;
    d.n_heads  = cfg->n_heads;
    d.n_kv_h   = cfg->n_kv_heads;
    d.nope     = mc->mla.qk_nope_head_dim;
    d.rope     = mc->mla.qk_rope_head_dim;
    d.lora     = mc->mla.kv_lora_rank;
    d.v_dim    = mc->mla.v_head_dim;
    d.max_seq  = s->max_seq_len;
    d.kv_mul   = d.n_heads / d.n_kv_h;
    d.head_dim = d.dim / d.n_heads;
    d.q_rows   = d.n_heads * (d.nope + d.rope);
    d.kva_rows = d.lora + d.rope;
    d.kvb_rows = d.n_kv_h * (d.nope + d.v_dim);
    d.iscale   = mla_kq_scale(cfg, d.nope, d.rope);
    rope_yarn_corr_dims(d.rope, cfg->rope_orig_ctx_len, cfg->rope_theta, cfg->rope_yarn_beta_fast,
                        cfg->rope_yarn_beta_slow, d.corr);
    return d;
}

/* Stores one row's k_nope and v (kvb_rows floats, [kv_head][nope + v_dim]) into the K/V caches. */
static void mla_store_kv_row(RunState *s, const MlaShape *d, int layer, int mapped_pos,
                             const float *kv_full) {
    for (int kv_h = 0; kv_h < d->n_kv_h; kv_h++) {
        const float *k_nope_src = kv_full + kv_h * (d->nope + d->v_dim);
        const float *v_src      = k_nope_src + d->nope;
        size_t off = KV_CACHE_IDX(layer, kv_h, mapped_pos, 0, d->n_kv_h, d->max_seq, d->head_dim);
        memcpy(&s->key_cache[off], k_nope_src, (size_t)d->nope * sizeof(float));
        memcpy(&s->value_cache[off], v_src, (size_t)d->v_dim * sizeof(float));
    }
}

/* RoPE on one row's k_rope_cur and on every head's q_rope part of q_full, at `pos`. CRITICAL: uses
 * mla_rope_freq (computed for rope_dim=64), NOT s->rope_freq (computed for head_dim=128, which
 * gives wrong frequencies). Full YaRN per-dimension blending: low-freq dims use unscaled theta,
 * high-freq dims use scaled theta (freq_scale), blend zone in between. */
static void mla_rope_row(const RunState *s, const Config *cfg, const MlaShape *d, float *k_rope_cur,
                         float *q_full, int pos) {
    const float *rf = s->mla.mla_rope_freq;
    rope_apply_yarn(k_rope_cur, rf, d->rope, pos, cfg->rope_freq_scale, cfg->rope_yarn_ext_factor,
                    cfg->rope_yarn_attn_factor, d->corr);
    for (int h = 0; h < d->n_heads; h++)
        rope_apply_yarn(q_full + h * (d->nope + d->rope) + d->nope, rf, d->rope, pos,
                        cfg->rope_freq_scale, cfg->rope_yarn_ext_factor, cfg->rope_yarn_attn_factor,
                        d->corr);
}

/* Scores of one head against the cached K (nope part) and k_rope, scaled; softmax in place. */
static void mla_scores(const RunState *s, const MlaShape *d, int layer, int pos, int valid_ctx,
                       const float *q_nope_h, int kv_h, float *att) {
    const float *q_rope_h = q_nope_h + d->nope;
    for (int t = 0; t < valid_ctx; t++) {
        int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t = sw_map_position(&s->sw, hist);
        size_t k_off = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, d->n_kv_h, d->max_seq, d->head_dim);
        float score  = tn_vec_dot(q_nope_h, &s->key_cache[k_off], d->nope);
        score += tn_vec_dot(q_rope_h, &s->mla.k_rope_cache[layer][mapped_t * d->rope], d->rope);
        att[t] = score * d->iscale;
    }
}

/* out_h += sum_t att[t] * V[t] for one head. */
static void mla_weighted_v(const RunState *s, const MlaShape *d, int layer, int pos, int valid_ctx,
                           int kv_h, const float *att, float *out_h) {
    for (int t = 0; t < valid_ctx; t++) {
        int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t = sw_map_position(&s->sw, hist);
        size_t v_off = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, d->n_kv_h, d->max_seq, d->head_dim);
        float a      = att[t];
        const float *v_vec = &s->value_cache[v_off];
        for (int i = 0; i < d->v_dim; i++) out_h[i] += a * v_vec[i];
    }
}

/* Steps 1-2b (single token): RMSNorm, KV compress into hb2 (kva_rows <= hidden_dim) and the KV-A
 * norm of kv_latent = hb2[0..lora-1]; k_rope_cur is left in hb2[lora..lora+rope-1]. */
static void mla_compress(RunState *s, const TransformerWeights *w, const Config *cfg,
                         const MlaShape *d, int layer, ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->attn.norm[layer], d->dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);
    DBG_DUMP(layer, "attn_norm", s->xb, d->dim);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul(w, tp, s->hb2, s->xb, w->mla.wkv_a[layer], w->mla.skv_a[layer], d->dim, d->kva_rows);
    if (t_step) tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);

    if (w->attn.sub_norm && w->attn.sub_norm[layer]) {
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        tn_rmsnorm(s->hb2, s->hb2, w->attn.sub_norm[layer], d->lora, cfg->rms_norm_eps);
        if (t_step)
            tn_step_timing_add(TN_STEP_7_KV_A_LATENT_NORM, tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "kv_cmpr", s->hb2, d->lora);
}

/* Step 3 (single token): KV expand of kv_latent into hb (kvb_rows <= hidden_dim), then the k_nope
 * and v cache writes, which must happen before hb is reused for Q. */
static void mla_expand_store(RunState *s, const TransformerWeights *w, const MlaShape *d, int layer,
                             int mapped_pos, ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul(w, tp, s->hb, s->hb2, w->mla.wkv_b[layer], w->mla.skv_b[layer], d->lora,
               d->kvb_rows);
    if (t_step) tn_step_timing_add(TN_STEP_8_KV_B_EXPANSION, tn_step_timing_now_ns() - t_step);
    DBG_DUMP(layer, "kv", s->hb, d->kvb_rows);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_store_kv_row(s, d, layer, mapped_pos, s->hb);
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Step 7 (single token): every head's scores, softmax and V-weighted sum into xb2, each part with
 * its own step-timing bracket (accumulated per head) and the head-0 debug dumps. */
static void mla_attend(RunState *s, const MlaShape *d, int layer, int pos, const float *q_full) {
    int valid_ctx = sw_valid_count(&s->sw, pos);
    memset(s->xb2, 0, (size_t)(d->n_heads * d->v_dim) * sizeof(float));
    for (int h = 0; h < d->n_heads; h++) {
        const float *q_nope_h = q_full + h * (d->nope + d->rope);
        int kv_h              = h / d->kv_mul;
        float *att            = s->att + h * d->max_seq;

        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        mla_scores(s, d, layer, pos, valid_ctx, q_nope_h, kv_h, att);
        if (h == 0) DBG_DUMP(layer, "attn_score_h0", att, valid_ctx); /* pre-softmax, head 0 */
        tn_softmax(att, valid_ctx);
        if (h == 0) DBG_DUMP(layer, "attn_soft_h0", att, valid_ctx); /* post-softmax, head 0 */
        if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);

        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        mla_weighted_v(s, d, layer, pos, valid_ctx, kv_h, att, s->xb2 + h * d->v_dim);
        if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
    }
}

/* Steps 4-6 (single token): Q projection into hb (q_rows <= hidden_dim; overwrites kv_full, which
 * is already cached), RoPE on k_rope_cur (in hb2) and the q_rope part of every head (in hb), and
 * the k_rope_cache write. Leaves q_full = hb: [n_heads][nope + rope]. */
static void mla_q_rope_store(RunState *s, const TransformerWeights *w, const Config *cfg,
                             const MlaShape *d, int layer, int pos, int mapped_pos,
                             ThreadPool *tp) {
    float *k_rope_cur = s->hb2 + d->lora;
    int64_t t_step    = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul(w, tp, s->hb, s->xb, w->mla.wq[layer], w->mla.sq[layer], d->dim, d->q_rows);
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
    DBG_DUMP(layer, "q", s->hb, d->q_rows);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_rope_row(s, cfg, d, k_rope_cur, s->hb, pos);
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);
    DBG_DUMP(layer, "k_pe", k_rope_cur, d->rope);
    DBG_DUMP(layer, "q_pe", s->hb + d->nope, d->rope); /* first head's rope portion */

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    memcpy(&s->mla.k_rope_cache[layer][mapped_pos * d->rope], k_rope_cur,
           (size_t)d->rope * sizeof(float));
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

void mla_attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                           const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    const MlaShape d = mla_dims(s, cfg, mc);
    int mapped_pos   = sw_map_position(&s->sw, pos);

    mla_compress(s, w, cfg, &d, layer, tp);
    mla_expand_store(s, w, &d, layer, mapped_pos, tp);
    mla_q_rope_store(s, w, cfg, &d, layer, pos, mapped_pos, tp);

    sw_advance(&s->sw);
    mla_attend(s, &d, layer, pos, s->hb);
    DBG_DUMP(layer, "attn_v_out", s->xb2, d.n_heads * d.v_dim); /* V-weighted sum, pre-W_o */

    /* Step 8: wo projection -> xb; residual */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul(w, tp, s->xb, s->xb2, w->attn.wo[layer], w->attn.so[layer], d.n_heads * d.v_dim,
               d.dim);
    DBG_DUMP(layer, "attn_wo_out", s->xb, d.dim); /* W_o output, pre-residual */
    for (int i = 0; i < d.dim; i++) s->x[i] += s->xb[i];
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
    DBG_DUMP(layer, "attn_resid", s->x, d.dim); /* x after += xb */

    if (g_tn_verbose) {
        char tag[48];
        snprintf(tag, sizeof(tag), "L%02d mla-attn out x", layer);
        dbg_vec_stats(tag, s->x, d.dim);
    }
}

/* ── Phase 18 (speculative decoding): batched multi-token MLA attention ────
 *
 * Same write-phase/read-phase restructuring as attention_forward_batch()
 * (attention.c) -- see that function's header comment for the general
 * design and correctness argument (sw_map_position/sw_valid_count purity).
 *
 * Buffer strategy differs from the single-token path's dim/hidden_dim
 * buffer-reuse trick: the batched-matmul kernels (batched_matmul.h) always
 * write a tightly-packed [n_tokens][output_width] block, so this function
 * uses each projection's own output width as the stride when writing into
 * (and later reading back from) sb->hb/sb->hb2 -- NOT sb's declared
 * dim/hidden_dim width -- exactly as long as output_width <= the buffer's
 * declared width (true for kva_rows/kvb_rows/q_rows <= hidden_dim, the
 * same invariant the single-token path's own comments already document).
 * kv_latent (lora-wide) needs its own small packed gather buffer before
 * the KV-expand batched call, since it lives at stride kva_rows within
 * sb->hb2 after the KV-compress call, not stride lora -- the batched
 * kernels take a single fixed stride (=n), so a value embedded at a
 * different stride must be gathered first. Small, freed at the end of
 * this call (lora/kvb_rows/v_dim are all much smaller than the model's
 * hidden_dim-scale matrices, so this allocation is a minor, one-off cost).
 *
 * Q4_K and F32 weights (has_mla_quant / is_float) fall back to n_tokens
 * sequential per-token MLA_MATMUL calls -- correct, documented, without
 * the bandwidth win for those formats (same Phase 18-B scope as the
 * generic dense/GQA path). */
/* One batch call's packed scratch: kv_latent (stride lora), kv_full (stride kvb_rows) and the
 * attention output (stride n_heads * v_dim). The batched-matmul kernels take a single fixed stride
 * (= n), so a value embedded at a different stride in sb->hb2 is gathered first. Small and freed at
 * the end of the call (lora / kvb_rows / v_dim are all much smaller than the hidden_dim-scale
 * matrices). */
typedef struct {
    float *latent, *kv_full, *attn_out;
    int attn_out_width;
} MlaBatch;

static int mla_batch_alloc(MlaBatch *b, const MlaShape *d, int n_tokens) {
    b->attn_out_width = d->n_heads * d->v_dim;
    b->latent         = (float *)malloc((size_t)n_tokens * d->lora * sizeof(float));
    b->kv_full        = (float *)malloc((size_t)n_tokens * d->kvb_rows * sizeof(float));
    b->attn_out       = (float *)malloc((size_t)n_tokens * b->attn_out_width * sizeof(float));
    return b->latent && b->kv_full && b->attn_out;
}

static void mla_batch_free(MlaBatch *b) {
    free(b->latent);
    free(b->kv_full);
    free(b->attn_out);
}

/* One batched projection (ternary weights only) or n_tokens sequential per-token calls (Q4_K and
 * F32 fall back, the same Phase 18-B scope as the generic dense/GQA path): out is packed at stride
 * n_out, in at stride n_in. */
static void mla_matmul_rows(const TransformerWeights *w, ThreadPool *tp, float *out,
                            const float *in, const void *wptr, float sptr, int n_in, int n_out,
                            int n_tokens) {
    if (!w->layers_are_ternary || w->mla.has_quant) {
        for (int k = 0; k < n_tokens; k++)
            mla_matmul(w, tp, out + (size_t)k * n_out, in + (size_t)k * n_in, wptr, sptr, n_in,
                       n_out);
        return;
    }
    tn_ternary_matmul_packed_batch(out, in, (const tn_u8 *)wptr, n_in, n_out, sptr, n_tokens, tp);
}

/* KV-A norm (when the model has one) of each row's kv_latent in sb->hb2, copied into b->latent. */
static void mla_gather_latent(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                              const MlaShape *d, int layer, int n_tokens, MlaBatch *b) {
    for (int k = 0; k < n_tokens; k++) {
        float *kv_latent_k = sb->hb2 + (size_t)k * d->kva_rows;
        if (w->attn.sub_norm && w->attn.sub_norm[layer])
            tn_rmsnorm(kv_latent_k, kv_latent_k, w->attn.sub_norm[layer], d->lora,
                       cfg->rms_norm_eps);
        memcpy(b->latent + (size_t)k * d->lora, kv_latent_k, (size_t)d->lora * sizeof(float));
    }
}

/* Steps 1-2 over all rows: per-token RMSNorm, KV compress packed at stride kva_rows in sb->hb2,
 * then the KV-A norm plus gather of kv_latent into a tightly packed buffer (stride lora). */
static void mla_rows_compress(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                              const MlaShape *d, int layer, int n_tokens, ThreadPool *tp,
                              MlaBatch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        tn_rmsnorm(sb->xb + (size_t)k * d->dim, sb->x + (size_t)k * d->dim, w->attn.norm[layer],
                   d->dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul_rows(w, tp, sb->hb2, sb->xb, w->mla.wkv_a[layer], w->mla.skv_a[layer], d->dim,
                    d->kva_rows, n_tokens);
    if (t_step) tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_gather_latent(sb, w, cfg, d, layer, n_tokens, b);
    if (t_step) tn_step_timing_add(TN_STEP_7_KV_A_LATENT_NORM, tn_step_timing_now_ns() - t_step);
}

/* Step 3 over all rows: KV expand (packed at stride kvb_rows) and the k_nope / v cache writes per
 * token. */
static void mla_rows_expand_store(RunState *s, const TransformerWeights *w, const MlaShape *d,
                                  int layer, int pos, int n_tokens, ThreadPool *tp, MlaBatch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul_rows(w, tp, b->kv_full, b->latent, w->mla.wkv_b[layer], w->mla.skv_b[layer], d->lora,
                    d->kvb_rows, n_tokens);
    if (t_step) tn_step_timing_add(TN_STEP_8_KV_B_EXPANSION, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        mla_store_kv_row(s, d, layer, sw_map_position(&s->sw, pos + k),
                         b->kv_full + (size_t)k * d->kvb_rows);
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Steps 4-6 over all rows: Q projection packed at stride q_rows in sb->hb (sb->hb was never used
 * for kv_full here, so there is no overwrite-after-caching ordering to worry about), RoPE per
 * token and the k_rope cache writes. */
static void mla_rows_q_rope(RunState *s, SpecBatchScratch *sb, const TransformerWeights *w,
                            const Config *cfg, const MlaShape *d, int layer, int pos, int n_tokens,
                            ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul_rows(w, tp, sb->hb, sb->xb, w->mla.wq[layer], w->mla.sq[layer], d->dim, d->q_rows,
                    n_tokens);
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        mla_rope_row(s, cfg, d, sb->hb2 + (size_t)k * d->kva_rows + d->lora,
                     sb->hb + (size_t)k * d->q_rows, pos + k);
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        int mapped_pos = sw_map_position(&s->sw, pos + k);
        memcpy(&s->mla.k_rope_cache[layer][mapped_pos * d->rope],
               sb->hb2 + (size_t)k * d->kva_rows + d->lora, (size_t)d->rope * sizeof(float));
        sw_advance(&s->sw); /* kept for parity; see attention_forward_batch()'s comment */
    }
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Read phase: attention per token (not batched across k). */
static void mla_rows_attend(RunState *s, SpecBatchScratch *sb, const MlaShape *d, int layer,
                            int pos, int n_tokens, MlaBatch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        int tok_pos         = pos + k;
        const float *q_full = sb->hb + (size_t)k * d->q_rows;
        float *out_k        = b->attn_out + (size_t)k * b->attn_out_width;
        memset(out_k, 0, (size_t)b->attn_out_width * sizeof(float));
        int valid_ctx = sw_valid_count(&s->sw, tok_pos);
        for (int h = 0; h < d->n_heads; h++) {
            int kv_h   = h / d->kv_mul;
            float *att = s->att + h * d->max_seq;
            mla_scores(s, d, layer, tok_pos, valid_ctx, q_full + h * (d->nope + d->rope), kv_h,
                       att);
            tn_softmax(att, valid_ctx);
            mla_weighted_v(s, d, layer, tok_pos, valid_ctx, kv_h, att, out_k + h * d->v_dim);
        }
    }
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);
}

TernaryError mla_attention_forward_batch(RunState *s, SpecBatchScratch *sb,
                                         const TransformerWeights *w, const Config *cfg,
                                         const MoEConfig *mc, int layer, int pos, int n_tokens,
                                         ThreadPool *tp) {
    const MlaShape d = mla_dims(s, cfg, mc);
    MlaBatch b;
    if (!mla_batch_alloc(&b, &d, n_tokens)) {
        mla_batch_free(&b);
        return TN_ERR_OOM;
    }

    /* Whole-batch-call step-timing granularity (Stage 5.5): mirrors mla_attention_forward()'s own
     * step tagging (4, 6, 7, 8, 9, 10, 11, 12) at the equivalent call boundaries. */
    mla_rows_compress(sb, w, cfg, &d, layer, n_tokens, tp, &b);
    mla_rows_expand_store(s, w, &d, layer, pos, n_tokens, tp, &b);
    mla_rows_q_rope(s, sb, w, cfg, &d, layer, pos, n_tokens, tp);
    mla_rows_attend(s, sb, &d, layer, pos, n_tokens, &b);

    /* Step 8: wo projection -> sb->xb2, packed at stride dim; residual */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    mla_matmul_rows(w, tp, sb->xb2, b.attn_out, w->attn.wo[layer], w->attn.so[layer],
                    b.attn_out_width, d.dim, n_tokens);
    tn_vec_add(sb->x, sb->x, sb->xb2, n_tokens * d.dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);

    mla_batch_free(&b);
    return TN_OK;
}
