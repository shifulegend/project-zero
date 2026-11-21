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

#include <math.h>
#include <string.h>

#include "core/platform.h"
#include "core/step_timing.h"
#include "math/matmul_q2_0.h"
#include "math/simd_core_ops.h"
#include "math/simd_vec_ops.h"

/* Stack bound for the per-head v_pred / delta temporaries of the delta rule (state size per head).
 * The projection buffers live in RunState.q35.scratch and have no fixed cap. */
#define Q35_MAX_HEAD_DIM 512

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

/* The full-attention layer's shapes. */
typedef struct {
    int dim, n_head, n_kv_h, head_dim, rope_dim, kv_mul, q_width, kv_width, max_seq;
} Q35Full;

static Q35Full q35_full_dims(const RunState *s, const Config *cfg, const MoEConfig *mc) {
    Q35Full d;
    d.dim      = cfg->dim;
    d.n_head   = cfg->n_heads;
    d.n_kv_h   = cfg->n_kv_heads;
    d.head_dim = mc->attn_head_dim;
    d.rope_dim = mc->attn_rope_dim > 0 ? mc->attn_rope_dim : d.head_dim;
    d.kv_mul   = d.n_head / d.n_kv_h;
    d.q_width  = 2 * d.n_head * d.head_dim; /* [Q | gate] doubled per head */
    d.kv_width = d.n_kv_h * d.head_dim;
    d.max_seq  = s->max_seq_len;
    return d;
}

/* Steps 1-2: RMSNorm, then the Q (with its gate half), K and V projections. */
static void q35_full_project(RunState *s, const TransformerWeights *w, const Config *cfg,
                             const Q35Full *d, int layer, ThreadPool *tp, float *q_full,
                             float *k_buf, float *v_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->attn.norm[layer], d->dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(q_full, s->xb, (const uint8_t *)w->attn.wq[layer], d->dim, d->q_width, tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
        t_step = tn_step_timing_now_ns();
    }
    parallel_matmul_q2_0(k_buf, s->xb, (const uint8_t *)w->attn.wk[layer], d->dim, d->kv_width, tp);
    parallel_matmul_q2_0(v_buf, s->xb, (const uint8_t *)w->attn.wv[layer], d->dim, d->kv_width, tp);
    if (t_step) tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);
}

/* Step 3: per-head Q/K RMSNorm, then partial NEOX rotary. */
static void q35_full_norm_rope(const RunState *s, const TransformerWeights *w, const Config *cfg,
                               const Q35Full *d, int layer, int pos, float *q_full, float *k_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int h = 0; h < d->n_head; h++) {
        float *q_h = q_full + (size_t)h * 2 * d->head_dim;
        tn_rmsnorm(q_h, q_h, w->q35.attn_q_norm[layer], d->head_dim, cfg->rms_norm_eps);
        rope_apply_neox_partial(q_h, s->q35.rope_freq, d->rope_dim, pos);
    }
    for (int kh = 0; kh < d->n_kv_h; kh++) {
        float *k_h = k_buf + (size_t)kh * d->head_dim;
        tn_rmsnorm(k_h, k_h, w->q35.attn_k_norm[layer], d->head_dim, cfg->rms_norm_eps);
        rope_apply_neox_partial(k_h, s->q35.rope_freq, d->rope_dim, pos);
    }
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);
}

/* Step 4: write K/V into this layer's own correctly-sized cache. */
static void q35_full_store_kv(RunState *s, const Q35Full *d, int layer, int pos, const float *k_buf,
                              const float *v_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int mapped_pos = sw_map_position(&s->sw, pos);
    for (int kh = 0; kh < d->n_kv_h; kh++) {
        size_t off = ((size_t)kh * d->max_seq + (size_t)mapped_pos) * d->head_dim;
        memcpy(&s->q35.key_cache[layer][off], &k_buf[(size_t)kh * d->head_dim],
               (size_t)d->head_dim * sizeof(float));
        memcpy(&s->q35.value_cache[layer][off], &v_buf[(size_t)kh * d->head_dim],
               (size_t)d->head_dim * sizeof(float));
    }
    sw_advance(&s->sw);
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Steps 5-6: attention per head (scores + softmax + weighted sum + the per-element sigmoid output
 * gate, which matches build_layer_attn) in one step-timing bucket: splitting inside the per-head
 * loop would add timer calls per head for little attribution value. */
static void q35_full_attend(const RunState *s, const Q35Full *d, int layer, int pos,
                            const float *q_full, float *attn_concat) {
    int64_t t_step    = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int valid_ctx     = sw_valid_count(&s->sw, pos);
    float inv_sqrt_hd = 1.0f / sqrtf((float)d->head_dim);
    for (int h = 0; h < d->n_head; h++) {
        const float *q_h    = q_full + (size_t)h * 2 * d->head_dim;
        const float *gate_h = q_h + d->head_dim;
        int kh              = h / d->kv_mul;
        float *att          = s->att + (size_t)h * d->max_seq;

        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off   = ((size_t)kh * d->max_seq + (size_t)mapped_t) * d->head_dim;
            att[t] = tn_vec_dot(q_h, &s->q35.key_cache[layer][off], d->head_dim) * inv_sqrt_hd;
        }
        tn_softmax(att, valid_ctx);

        float *out_h = attn_concat + (size_t)h * d->head_dim;
        memset(out_h, 0, (size_t)d->head_dim * sizeof(float));
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off   = ((size_t)kh * d->max_seq + (size_t)mapped_t) * d->head_dim;
            tn_vec_saxpy(out_h, att[t], &s->q35.value_cache[layer][off], d->head_dim);
        }
        for (int i = 0; i < d->head_dim; i++) {
            float g = 1.0f / (1.0f + expf(-gate_h[i]));
            out_h[i] *= g;
        }
    }
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);
}

static void q35_full_attn_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                                  const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    const Q35Full d = q35_full_dims(s, cfg, mc);

    /* Per-RunState scratch (q35_run_state_alloc): no shared static, no width cap. */
    float *q_full      = s->q35.scratch;
    float *k_buf       = q_full + d.q_width;
    float *v_buf       = k_buf + d.kv_width;
    float *attn_concat = v_buf + d.kv_width;

    q35_full_project(s, w, cfg, &d, layer, tp, q_full, k_buf, v_buf);
    q35_full_norm_rope(s, w, cfg, &d, layer, pos, q_full, k_buf);
    q35_full_store_kv(s, &d, layer, pos, k_buf, v_buf);
    q35_full_attend(s, &d, layer, pos, q_full, attn_concat);

    /* Step 7: output projection + residual */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(s->xb, attn_concat, (const uint8_t *)w->attn.wo[layer],
                         d.n_head * d.head_dim, d.dim, tp);
    tn_vec_add(s->x, s->x, s->xb, d.dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

/* ── Linear attention: Gated DeltaNet ────────────────────────────────────
 * Cyclic (NOT block) head-repeat: ggml_repeat tiles its source periodically
 * (dst[j] = src[j % src_len]), unlike the block-style repeat_kv used by
 * ordinary GQA elsewhere in this codebase. v-head hv reads q/k-head hv%n_k. */
static inline int q35_head_repeat_index(int v_head, int n_k_heads) {
    return v_head % n_k_heads;
}

/* The linear-attention layer's shapes. */
typedef struct {
    int dim, conv_k, n_k_heads, n_v_heads;
    int state_sz; /* head_k_dim == head_v_dim */
    int key_dim, value_dim, conv_dim;
    float eps;
} Q35Lin;

/* Where each intermediate lives in the per-RunState scratch. */
typedef struct {
    float *qkv_mixed, *z, *beta, *alpha, *g_exp, *conv_out, *core_out;
} Q35LinBuf;

static Q35Lin q35_lin_dims(const Config *cfg, const MoEConfig *mc) {
    Q35Lin d;
    d.dim       = cfg->dim;
    d.conv_k    = mc->ssm_conv_kernel;
    d.n_k_heads = mc->ssm_group_count;
    d.n_v_heads = mc->ssm_time_step_rank;
    d.state_sz  = mc->ssm_state_size;
    d.key_dim   = d.n_k_heads * d.state_sz;
    d.value_dim = mc->ssm_inner_size; /* n_v_heads * state_sz */
    d.conv_dim  = d.key_dim * 2 + d.value_dim;
    d.eps       = cfg->rms_norm_eps;
    return d;
}

/* Linear-attention part of the scratch, after the full-attention part (see q35_run_state_alloc):
 * the full-attention part is q_full (2*n_heads*head_dim) + k + v (kv_width each) + concat. */
static Q35LinBuf q35_lin_buf(RunState *s, const Config *cfg, const MoEConfig *mc, const Q35Lin *d) {
    const size_t full_n = 3 * (size_t)cfg->n_heads * (size_t)mc->attn_head_dim +
                          2 * (size_t)cfg->n_kv_heads * (size_t)mc->attn_head_dim;
    Q35LinBuf b;
    b.qkv_mixed = s->q35.scratch + full_n;
    b.z         = b.qkv_mixed + d->conv_dim;
    b.beta      = b.z + d->value_dim;
    b.alpha     = b.beta + d->n_v_heads;
    b.g_exp     = b.alpha + d->n_v_heads;
    b.conv_out  = b.g_exp + d->n_v_heads;
    b.core_out  = b.conv_out + d->conv_dim;
    return b;
}

/* Steps 1-2: RMSNorm, then the projections: combined q|k|v (pre-conv) and the separate output
 * gate. Attribution note: the combined qkv projection bills to step 5 and the gate/beta/alpha
 * projections to step 6, the closest available buckets; DeltaNet has no literal "Q
 * projection"/"KV compression". */
static void q35_lin_project(RunState *s, const TransformerWeights *w, const Q35Lin *d,
                            const Q35LinBuf *b, int layer, ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->attn.norm[layer], d->dim, d->eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_matmul_q2_0(b->qkv_mixed, s->xb, (const uint8_t *)w->q35.ssm_qkv[layer], d->dim,
                         d->conv_dim, tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
        t_step = tn_step_timing_now_ns();
    }
    parallel_matmul_q2_0(b->z, s->xb, (const uint8_t *)w->q35.ssm_gate[layer], d->dim, d->value_dim,
                         tp);
    parallel_matmul_q2_0(b->beta, s->xb, (const uint8_t *)w->q35.ssm_beta[layer], d->dim,
                         d->n_v_heads, tp);
    parallel_matmul_q2_0(b->alpha, s->xb, (const uint8_t *)w->q35.ssm_alpha[layer], d->dim,
                         d->n_v_heads, tp);
    if (t_step) tn_step_timing_add(TN_STEP_6_KV_A_COMPRESSION, tn_step_timing_now_ns() - t_step);
}

/* Step 3: per-head scalar decay gate. gate = ssm_a * softplus(alpha + dt_bias); g_exp = exp(gate).
 * ssm_a/dt_bias already carry whatever sign/scale the conversion baked in (build_layer_attn_linear
 * multiplies them directly, no extra negation): matched verbatim, not re-derived. */
static void q35_lin_decay_gates(const TransformerWeights *w, const Q35Lin *d, const Q35LinBuf *b,
                                int layer) {
    for (int hv = 0; hv < d->n_v_heads; hv++) {
        float sig      = 1.0f / (1.0f + expf(-b->beta[hv]));
        b->beta[hv]    = sig;
        float ab       = b->alpha[hv] + w->q35.ssm_dt_bias[layer][hv];
        float softplus = (ab > 20.0f) ? ab : log1pf(expf(ab));
        float gate     = w->q35.ssm_a[layer][hv] * softplus;
        b->g_exp[hv]   = expf(gate);
    }
}

/* Step 4: depthwise causal conv1d (kernel=conv_k) + SiLU, using the persisted (conv_k-1)-timestep
 * history; then slide the history window.
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
static void q35_lin_conv(RunState *s, const TransformerWeights *w, const Q35Lin *d,
                         const Q35LinBuf *b, int layer) {
    int64_t t_step   = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    const int conv_k = d->conv_k, conv_dim = d->conv_dim;
    float *hist = s->q35.conv_state[layer]; /* [conv_k-1][conv_dim], tap-major (this file's own
                                               buffer, no GGUF layout constraint) */
    const float *kernel = w->q35.ssm_conv1d[layer]; /* [conv_dim][conv_k], channel-major per GGUF */
    for (int c = 0; c < conv_dim; c++) {
        const float *kc = kernel + (size_t)c * conv_k;
        float acc       = 0.0f;
        for (int i = 0; i < conv_k - 1; i++) acc += kc[i] * hist[(size_t)i * conv_dim + c];
        acc += kc[conv_k - 1] * b->qkv_mixed[c];
        b->conv_out[c] = acc / (1.0f + expf(-acc)); /* SiLU */
    }
    /* Slide history: drop oldest, append the *pre-conv* qkv_mixed sample. */
    for (int i = 0; i < conv_k - 2; i++)
        memcpy(&hist[(size_t)i * conv_dim], &hist[(size_t)(i + 1) * conv_dim],
               (size_t)conv_dim * sizeof(float));
    if (conv_k - 1 > 0)
        memcpy(&hist[(size_t)(conv_k - 2) * conv_dim], b->qkv_mixed,
               (size_t)conv_dim * sizeof(float));
    if (t_step) tn_step_timing_add(TN_STEP_22_DELTANET_CONV, tn_step_timing_now_ns() - t_step);
}

/* Step 5: split conv_out into q/k/v and L2-normalise q and k per head (unit vectors). */
static void q35_lin_qk_norm(const Q35Lin *d, const Q35LinBuf *b) {
    float *q_lin = b->conv_out;              /* [n_k_heads][state_sz] */
    float *k_lin = b->conv_out + d->key_dim; /* [n_k_heads][state_sz] */
    for (int kh = 0; kh < d->n_k_heads; kh++) {
        float *qh    = q_lin + (size_t)kh * d->state_sz;
        float *k_row = k_lin + (size_t)kh * d->state_sz;
        float qn = 0.0f, kn = 0.0f;
        for (int i = 0; i < d->state_sz; i++) {
            qn += qh[i] * qh[i];
            kn += k_row[i] * k_row[i];
        }
        float qs = 1.0f / fmaxf(sqrtf(qn), d->eps);
        float ks = 1.0f / fmaxf(sqrtf(kn), d->eps);
        for (int i = 0; i < d->state_sz; i++) {
            qh[i] *= qs;
            k_row[i] *= ks;
        }
    }
}

/* acc[j] += sum_i S[i][j] * x[i] * scale over the rows whose x[i] is non-zero: S^T x. */
static void q35_lin_mat_t_vec(const float *S, int state_sz, const float *x, float scale,
                              float *acc) {
    for (int i = 0; i < state_sz; i++) {
        float xi = x[i] * scale;
        if (xi == 0.0f) continue;
        const float *Srow = S + (size_t)i * state_sz;
        for (int j = 0; j < state_sz; j++) acc[j] += Srow[j] * xi;
    }
}

/* Step 6 for one v-head: the gated delta-rule recurrence on its persistent state S [i][d]:
 * decay, v_pred = S^T k, delta = beta * (v - v_pred), rank-1 update S += k delta^T, read-out
 * out = S^T (q * q_scale). */
static void q35_lin_recur_head(float *S, int state_sz, const float *q_h, const float *k_h,
                               const float *v_h, float gamma, float bh, float q_scale,
                               float *out_h) {
    for (int idx = 0; idx < state_sz * state_sz; idx++) S[idx] *= gamma;

    float v_pred[Q35_MAX_HEAD_DIM];
    for (int i = 0; i < state_sz; i++) v_pred[i] = 0.0f;
    q35_lin_mat_t_vec(S, state_sz, k_h, 1.0f, v_pred);
    float delta[Q35_MAX_HEAD_DIM];
    for (int i = 0; i < state_sz; i++) delta[i] = bh * (v_h[i] - v_pred[i]);

    for (int i = 0; i < state_sz; i++) {
        float ki    = k_h[i];
        float *Srow = S + (size_t)i * state_sz;
        for (int j = 0; j < state_sz; j++) Srow[j] += ki * delta[j];
    }

    for (int i = 0; i < state_sz; i++) out_h[i] = 0.0f;
    q35_lin_mat_t_vec(S, state_sz, q_h, q_scale, out_h);
}

/* Steps 5-6, billed to the recurrence bucket together: q/k normalisation, then the recurrence of
 * every v-head (v-head hv reads q/k-head hv % n_k_heads, see q35_head_repeat_index). */
static void q35_lin_recurrence(RunState *s, const Q35Lin *d, const Q35LinBuf *b, int layer) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    q35_lin_qk_norm(d, b);
    const float *q_lin = b->conv_out;
    const float *k_lin = b->conv_out + d->key_dim;
    const float *v_lin = b->conv_out + 2 * d->key_dim; /* [n_v_heads][state_sz] */
    float q_scale      = 1.0f / sqrtf((float)d->state_sz);
    for (int hv = 0; hv < d->n_v_heads; hv++) {
        int kh = q35_head_repeat_index(hv, d->n_k_heads);
        q35_lin_recur_head(s->q35.recur_state[layer] + (size_t)hv * d->state_sz * d->state_sz,
                           d->state_sz, q_lin + (size_t)kh * d->state_sz,
                           k_lin + (size_t)kh * d->state_sz, v_lin + (size_t)hv * d->state_sz,
                           b->g_exp[hv], b->beta[hv], q_scale,
                           b->core_out + (size_t)hv * d->state_sz);
    }
    if (t_step)
        tn_step_timing_add(TN_STEP_23_DELTANET_RECURRENCE, tn_step_timing_now_ns() - t_step);
}

/* Steps 7-8: gated RMSNorm per head (norm(core_out) * silu(z), both per head), then the output
 * projection and the residual. */
static void q35_lin_output(RunState *s, const TransformerWeights *w, const Q35Lin *d,
                           const Q35LinBuf *b, int layer, ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int hv = 0; hv < d->n_v_heads; hv++) {
        float *out_h = b->core_out + (size_t)hv * d->state_sz;
        float *z_h   = b->z + (size_t)hv * d->state_sz;
        tn_rmsnorm(out_h, out_h, w->q35.ssm_norm[layer], d->state_sz, d->eps);
        for (int i = 0; i < d->state_sz; i++)
            out_h[i] *= z_h[i] / (1.0f + expf(-z_h[i])); /* SiLU(z) */
    }
    parallel_matmul_q2_0(s->xb, b->core_out, (const uint8_t *)w->q35.ssm_out[layer], d->value_dim,
                         d->dim, tp);
    tn_vec_add(s->x, s->x, s->xb, d->dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

static void q35_linear_attn_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                                    const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    (void)pos;
    const Q35Lin d    = q35_lin_dims(cfg, mc);
    const Q35LinBuf b = q35_lin_buf(s, cfg, mc, &d);
    q35_lin_project(s, w, &d, &b, layer, tp);
    q35_lin_decay_gates(w, &d, &b, layer);
    q35_lin_conv(s, w, &d, &b, layer);
    q35_lin_recurrence(s, &d, &b, layer);
    q35_lin_output(s, w, &d, &b, layer, tp);
}

void qwen35_attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                              const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    if (q35_layer_is_full_attn(mc, layer)) {
        q35_full_attn_forward(s, w, cfg, mc, layer, pos, tp);
    } else {
        q35_linear_attn_forward(s, w, cfg, mc, layer, pos, tp);
    }
}
