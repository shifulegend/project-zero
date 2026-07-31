/**
 * qwen35_attention.c
 *
 * Qwen3.5/3.6 hybrid attention: two structurally different per-layer
 * sub-paths selected by q35_layer_is_full_attn() (see moe_config.h).
 *
 * Full-attention layers (1 in every `full_attention_interval`):
 *   GQA with per-head RMSNorm on Q/K ("QK-norm"), partial rotary (only the
 *   first attn_rope_dim of attn_head_dim gets RoPE, NEOX/rotate-half pairing
 *   — NOT this codebase's default interleaved-pair convention, see
 *   rope_apply_neox_partial below), and a sigmoid output gate: the Q
 *   projection produces [Q | gate] doubled-width per head.
 *
 * Linear-attention layers (Gated DeltaNet):
 *   combined q/k/v projection -> short causal depthwise conv1d (kernel=4)
 *   + SiLU -> per-head L2-norm on q/k -> cyclic head-repeat (NOT the usual
 *   block-repeat GQA convention — see q35_head_repeat_index) -> scalar
 *   per-head decay-gated delta-rule recurrence (persistent state matrix,
 *   O(1) per token, no growing KV cache) -> gated RMSNorm (SiLU output
 *   gate) -> output projection.
 *
 * Every matmul in this model is Q2_0 (see matmul_q2_0.c) — the model card's
 * claim of "no high-precision escape hatches" is literal, so this file
 * calls parallel_matmul_q2_0 directly rather than the generic
 * layers_are_ternary/WEIGHT_TYPE_F16/float32 3-way dispatch used elsewhere.
 *
 * Reference: llama.cpp src/models/qwen35.cpp + src/models/delta-net-base.cpp
 * (build_layer_attn / build_layer_attn_linear / build_delta_net_autoregressive),
 * cross-checked against a real Ternary-Bonsai-27B-Q2_0.gguf header dump —
 * see docs/ai/decision-log.md for the full derivation.
 */

#include "transformer/qwen35_attention.h"
#include "math/matmul_q2_0.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include "core/step_timing.h"
#include <math.h>
#include <string.h>

/* Generous fixed stack-buffer bounds for this arch's projection widths.
 * Validated (>0) at config-parse time in moe_config_from_gguf(); these
 * ceilings only need to exceed any real model's dims (Bonsai-27B's largest
 * is q_full=12288, conv_dim=10240) — mirrors ATTN_PREQ_BUF_SIZE's existing
 * "fixed generous buffer, no runtime malloc" convention in attention.c. */
#define Q35_MAX_Q_WIDTH   16384
#define Q35_MAX_KV_WIDTH   8192
#define Q35_MAX_CONV_DIM  16384
#define Q35_MAX_HEADS       128
#define Q35_MAX_HEAD_DIM    512

/* ── Partial-rotary RoPE, NEOX/rotate-half pairing ───────────────────────
 * Pairs dimension i with i+rope_dim/2 (NOT this file's sibling rope.c /
 * mla_attention.c, which both use interleaved (2i,2i+1) pairs — Qwen3.5/3.6
 * (M-RoPE mode GGML_ROPE_TYPE_IMROPE upstream) applies cos/sin in NEOX
 * ordering per ggml's own comment: "cos/sin are applied in NEOX ordering").
 * Only the first rope_dim of the head's dims are touched; the rest pass
 * through unchanged (partial_rotary_factor < 1). No YaRN: this model ships
 * no rope.scaling.* metadata (native 262144 context, no extension needed).
 */
static void rope_apply_neox_partial(float *v, const float *freq, int rope_dim, int pos) {
    int half = rope_dim / 2;
    for (int i = 0; i < half; i++) {
        float theta = (float)pos * freq[i];
        float c = cosf(theta), s = sinf(theta);
        float x0 = v[i], x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/* ── Full attention: GQA + QK-norm + partial rotary + sigmoid gate ──────── */

static void q35_full_attn_forward(RunState *s, const TransformerWeights *w,
                                   const Config *cfg, const MoEConfig *mc,
                                   int layer, int pos, ThreadPool *tp) {
    int dim       = cfg->dim;
    int n_head    = cfg->n_heads;
    int n_kv_h    = cfg->n_kv_heads;
    int head_dim  = mc->attn_head_dim;
    int rope_dim  = mc->attn_rope_dim > 0 ? mc->attn_rope_dim : head_dim;
    int kv_mul    = n_head / n_kv_h;
    int q_width   = 2 * n_head * head_dim;   /* [Q | gate] doubled per head */
    int kv_width  = n_kv_h * head_dim;
    int max_seq   = s->max_seq_len;

    static float q_full[Q35_MAX_Q_WIDTH];
    static float k_buf[Q35_MAX_KV_WIDTH];
    static float v_buf[Q35_MAX_KV_WIDTH];
    static float attn_concat[Q35_MAX_HEADS * Q35_MAX_HEAD_DIM];

    /* Step 1: RMSNorm */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->rms_att_weight[layer], dim, cfg->rms_norm_eps);
    if (t_step)
        tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    /* Step 2: projections */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(q_full, s->xb, (const uint8_t *)w->wq[layer], dim, q_width,  tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
        t_step = tn_step_timing_now_ns();
    }
    parallel_matmul_q2_0(k_buf,  s->xb, (const uint8_t *)w->wk[layer], dim, kv_width, tp);
    parallel_matmul_q2_0(v_buf,  s->xb, (const uint8_t *)w->wv[layer], dim, kv_width, tp);
    if (t_step)
        tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);

    /* Step 3: per-head Q/K RMSNorm, then partial NEOX rotary */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int h = 0; h < n_head; h++) {
        float *q_h = q_full + (size_t)h * 2 * head_dim;
        tn_rmsnorm(q_h, q_h, w->q35_attn_q_norm[layer], head_dim, cfg->rms_norm_eps);
        rope_apply_neox_partial(q_h, s->q35_rope_freq, rope_dim, pos);
    }
    for (int kh = 0; kh < n_kv_h; kh++) {
        float *k_h = k_buf + (size_t)kh * head_dim;
        tn_rmsnorm(k_h, k_h, w->q35_attn_k_norm[layer], head_dim, cfg->rms_norm_eps);
        rope_apply_neox_partial(k_h, s->q35_rope_freq, rope_dim, pos);
    }
    if (t_step)
        tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);

    /* Step 4: write K/V into this layer's own correctly-sized cache */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int mapped_pos = sw_map_position(&s->sw, pos);
    for (int kh = 0; kh < n_kv_h; kh++) {
        size_t off = ((size_t)kh * max_seq + (size_t)mapped_pos) * head_dim;
        memcpy(&s->q35_key_cache[layer][off],   &k_buf[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
        memcpy(&s->q35_value_cache[layer][off], &v_buf[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
    }
    sw_advance(&s->sw);
    if (t_step)
        tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);

    /* Step 5: attention per head (scores+softmax+weighted sum+gate together —
     * one bucket; splitting inside the per-head loop would add timer calls on
     * a per-head granularity for little attribution value) */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int valid_ctx = sw_valid_count(&s->sw, pos);
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_head; h++) {
        float *q_h   = q_full + (size_t)h * 2 * head_dim;
        float *gate_h = q_h + head_dim;
        int    kh    = h / kv_mul;
        float *att   = s->att + (size_t)h * max_seq;

        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
            att[t] = tn_vec_dot(q_h, &s->q35_key_cache[layer][off], head_dim) * inv_sqrt_hd;
        }
        tn_softmax(att, valid_ctx);

        float *out_h = attn_concat + (size_t)h * head_dim;
        memset(out_h, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
            tn_vec_saxpy(out_h, att[t], &s->q35_value_cache[layer][off], head_dim);
        }
        /* Step 6: sigmoid output gate (per-element, matches build_layer_attn) */
        for (int d = 0; d < head_dim; d++) {
            float g = 1.0f / (1.0f + expf(-gate_h[d]));
            out_h[d] *= g;
        }
    }
    if (t_step)
        tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);

    /* Step 7: output projection + residual */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(s->xb, attn_concat, (const uint8_t *)w->wo[layer], n_head * head_dim, dim, tp);
    tn_vec_add(s->x, s->x, s->xb, dim);
    if (t_step)
        tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

/* ── Linear attention: Gated DeltaNet ────────────────────────────────────
 * Cyclic (NOT block) head-repeat: ggml_repeat tiles its source periodically
 * (dst[j] = src[j % src_len]), unlike the block-style repeat_kv used by
 * ordinary GQA elsewhere in this codebase. v-head hv reads q/k-head hv%n_k. */
static inline int q35_head_repeat_index(int v_head, int n_k_heads) {
    return v_head % n_k_heads;
}

static void q35_linear_attn_forward(RunState *s, const TransformerWeights *w,
                                     const Config *cfg, const MoEConfig *mc,
                                     int layer, int pos, ThreadPool *tp) {
    (void)pos;
    int dim        = cfg->dim;
    int conv_k     = mc->ssm_conv_kernel;
    int n_k_heads  = mc->ssm_group_count;
    int n_v_heads  = mc->ssm_time_step_rank;
    int state_sz   = mc->ssm_state_size;      /* head_k_dim == head_v_dim */
    int key_dim    = n_k_heads * state_sz;
    int value_dim  = mc->ssm_inner_size;      /* n_v_heads * state_sz */
    int conv_dim   = key_dim * 2 + value_dim;
    float eps      = cfg->rms_norm_eps;

    static float qkv_mixed[Q35_MAX_CONV_DIM];
    static float z[Q35_MAX_KV_WIDTH];
    static float beta[Q35_MAX_HEADS];
    static float alpha[Q35_MAX_HEADS];
    static float g_exp[Q35_MAX_HEADS];
    static float conv_out[Q35_MAX_CONV_DIM];
    static float core_out[Q35_MAX_KV_WIDTH];

    /* Step 1: RMSNorm */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->rms_att_weight[layer], dim, eps);
    if (t_step)
        tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    /* Step 2: projections — combined q|k|v (pre-conv), separate output gate.
     * Attribution note: the combined qkv projection bills to step 5 and the
     * gate/beta/alpha projections to step 6 — the closest available buckets;
     * DeltaNet has no literal "Q projection"/"KV compression". */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(qkv_mixed, s->xb, (const uint8_t *)w->q35_ssm_qkv[layer],  dim, conv_dim,  tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
        t_step = tn_step_timing_now_ns();
    }
    parallel_matmul_q2_0(z,         s->xb, (const uint8_t *)w->q35_ssm_gate[layer], dim, value_dim, tp);
    parallel_matmul_q2_0(beta,      s->xb, (const uint8_t *)w->q35_ssm_beta[layer], dim, n_v_heads, tp);
    parallel_matmul_q2_0(alpha,     s->xb, (const uint8_t *)w->q35_ssm_alpha[layer],dim, n_v_heads, tp);
    if (t_step)
        tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);

    /* Step 3: per-head scalar decay gate.
     * gate = ssm_a * softplus(alpha + dt_bias); g_exp = exp(gate).
     * ssm_a/dt_bias already carry whatever sign/scale the conversion baked
     * in (build_layer_attn_linear multiplies them directly, no extra
     * negation) — matched verbatim, not re-derived. */
    for (int hv = 0; hv < n_v_heads; hv++) {
        float sig = 1.0f / (1.0f + expf(-beta[hv]));
        beta[hv] = sig;
        float ab = alpha[hv] + w->q35_ssm_dt_bias[layer][hv];
        float softplus = (ab > 20.0f) ? ab : log1pf(expf(ab));
        float gate = w->q35_ssm_a[layer][hv] * softplus;
        g_exp[hv] = expf(gate);
    }

    /* Step 4: depthwise causal conv1d (kernel=conv_k) + SiLU, using the
     * persisted (conv_k-1)-timestep history; then slide the history window.
     *
     * ssm_conv1d.weight's GGUF shape is [conv_k, conv_dim] in ne[] order
     * (ne0=conv_k is the fast/contiguous dimension — confirmed both from
     * the real Ternary-Bonsai-27B.gguf's tensor dims and from llama.cpp's
     * own create_tensor(..., {ssm_d_conv, conv_dim}, ...) — ne0 listed
     * first is always the fastest-varying axis). So per channel c, the
     * conv_k tap weights are contiguous: kernel[c*conv_k + i], NOT
     * kernel[i*conv_dim + c] (channel-major, not tap-major) — the
     * transposed indexing this file originally used silently read the
     * wrong weight for every tap/channel pair except the diagonal-ish
     * overlap, corrupting every linear-attention layer's q/k/v inputs
     * while still producing finite, plausible-looking numbers (see
     * docs/ai/mistakes.md). */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    {
        float *hist = s->q35_conv_state[layer];          /* [conv_k-1][conv_dim], tap-major (this file's own buffer, no GGUF layout constraint) */
        const float *kernel = w->q35_ssm_conv1d[layer];  /* [conv_dim][conv_k], channel-major per GGUF */
        for (int c = 0; c < conv_dim; c++) {
            const float *kc = kernel + (size_t)c * conv_k;
            float acc = 0.0f;
            for (int i = 0; i < conv_k - 1; i++)
                acc += kc[i] * hist[(size_t)i * conv_dim + c];
            acc += kc[conv_k - 1] * qkv_mixed[c];
            conv_out[c] = acc / (1.0f + expf(-acc));  /* SiLU */
        }
        /* Slide history: drop oldest, append the *pre-conv* qkv_mixed sample. */
        for (int i = 0; i < conv_k - 2; i++)
            memcpy(&hist[(size_t)i * conv_dim], &hist[(size_t)(i + 1) * conv_dim],
                   (size_t)conv_dim * sizeof(float));
        if (conv_k - 1 > 0)
            memcpy(&hist[(size_t)(conv_k - 2) * conv_dim], qkv_mixed, (size_t)conv_dim * sizeof(float));
    }
    if (t_step)
        tn_step_timing_add(TN_STEP_22_DELTANET_CONV, tn_step_timing_now_ns() - t_step);

    /* Step 5: split into q/k/v, L2-normalize q/k per head (unit vectors) —
     * billed to the recurrence bucket together with step 6 below */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    float *q_lin = conv_out;               /* [n_k_heads][state_sz] */
    float *k_lin = conv_out + key_dim;     /* [n_k_heads][state_sz] */
    float *v_lin = conv_out + 2 * key_dim; /* [n_v_heads][state_sz] */

    for (int kh = 0; kh < n_k_heads; kh++) {
        float *qh = q_lin + (size_t)kh * state_sz;
        float *kh_ = k_lin + (size_t)kh * state_sz;
        float qn = 0.0f, kn = 0.0f;
        for (int d = 0; d < state_sz; d++) { qn += qh[d]*qh[d]; kn += kh_[d]*kh_[d]; }
        float qs = 1.0f / fmaxf(sqrtf(qn), eps);
        float ks = 1.0f / fmaxf(sqrtf(kn), eps);
        for (int d = 0; d < state_sz; d++) { qh[d] *= qs; kh_[d] *= ks; }
    }

    /* Step 6: gated delta-rule recurrence, per v-head (persistent state) */
    float q_scale = 1.0f / sqrtf((float)state_sz);
    for (int hv = 0; hv < n_v_heads; hv++) {
        int kh = q35_head_repeat_index(hv, n_k_heads);
        const float *q_h = q_lin + (size_t)kh * state_sz;
        const float *k_h = k_lin + (size_t)kh * state_sz;
        const float *v_h = v_lin + (size_t)hv * state_sz;
        float gamma = g_exp[hv];
        float bh    = beta[hv];
        float *S = s->q35_recur_state[layer] + (size_t)hv * state_sz * state_sz; /* [i][d] */

        /* decay */
        for (int idx = 0; idx < state_sz * state_sz; idx++) S[idx] *= gamma;

        /* v_pred[d] = sum_i S[i][d]*k[i]; delta[d] = beta*(v[d]-v_pred[d]) */
        float v_pred[Q35_MAX_HEAD_DIM];
        for (int d = 0; d < state_sz; d++) v_pred[d] = 0.0f;
        for (int i = 0; i < state_sz; i++) {
            float ki = k_h[i];
            if (ki == 0.0f) continue;
            const float *Srow = S + (size_t)i * state_sz;
            for (int d = 0; d < state_sz; d++) v_pred[d] += Srow[d] * ki;
        }
        float delta[Q35_MAX_HEAD_DIM];
        for (int d = 0; d < state_sz; d++) delta[d] = bh * (v_h[d] - v_pred[d]);

        /* rank-1 update: S[i][d] += k[i]*delta[d] */
        for (int i = 0; i < state_sz; i++) {
            float ki = k_h[i];
            float *Srow = S + (size_t)i * state_sz;
            for (int d = 0; d < state_sz; d++) Srow[d] += ki * delta[d];
        }

        /* read out: out[d] = sum_i S[i][d] * (q[i]*q_scale) */
        float *out_h = core_out + (size_t)hv * state_sz;
        for (int d = 0; d < state_sz; d++) out_h[d] = 0.0f;
        for (int i = 0; i < state_sz; i++) {
            float qi = q_h[i] * q_scale;
            if (qi == 0.0f) continue;
            const float *Srow = S + (size_t)i * state_sz;
            for (int d = 0; d < state_sz; d++) out_h[d] += Srow[d] * qi;
        }
    }
    if (t_step)
        tn_step_timing_add(TN_STEP_23_DELTANET_RECURRENCE, tn_step_timing_now_ns() - t_step);

    /* Step 7: gated RMSNorm per head — norm(core_out) * silu(z), both per-head */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int hv = 0; hv < n_v_heads; hv++) {
        float *out_h = core_out + (size_t)hv * state_sz;
        float *z_h   = z + (size_t)hv * state_sz;
        tn_rmsnorm(out_h, out_h, w->q35_ssm_norm[layer], state_sz, eps);
        for (int d = 0; d < state_sz; d++)
            out_h[d] *= z_h[d] / (1.0f + expf(-z_h[d])); /* SiLU(z) */
    }

    /* Step 8: output projection + residual (gated norm above included) */
    parallel_matmul_q2_0(s->xb, core_out, (const uint8_t *)w->q35_ssm_out[layer], value_dim, dim, tp);
    tn_vec_add(s->x, s->x, s->xb, dim);
    if (t_step)
        tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

void qwen35_attention_forward(RunState *s, const TransformerWeights *w,
                               const Config *cfg, const MoEConfig *mc,
                               int layer, int pos, ThreadPool *tp) {
    if (q35_layer_is_full_attn(mc, layer)) {
        q35_full_attn_forward(s, w, cfg, mc, layer, pos, tp);
    } else {
        q35_linear_attn_forward(s, w, cfg, mc, layer, pos, tp);
    }
}
