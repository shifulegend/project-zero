/**
 * qwen3moe_attention.c
 *
 * Qwen3-MoE (e.g. Qwen3-30B-A3B) attention: plain GQA with per-head
 * RMSNorm on Q/K ("QK-norm") applied before RoPE, standard full
 * (non-partial) rotary embedding — every layer uses this same attention
 * shape (no linear-attention layers, unlike Qwen3.5/3.6's hybrid model),
 * and there's no latent KV compression (unlike DeepSeek-V2's MLA).
 *
 * A dedicated forward function (rather than reusing attention_forward()'s
 * generic GQA branch) is required because this arch's head_dim is
 * independent of dim/n_heads (e.g. Qwen3-30B-A3B: dim=2048, n_heads=32 ->
 * dim/n_heads=64, but the model's actual head_dim is 128) — reusing the
 * generic branch's dim-sized s->q/s->xb/s->xb2 scratch buffers and
 * s->key_cache/value_cache (sized with config_head_dim(cfg)==dim/n_heads)
 * would silently overflow them whenever n_heads*attn_head_dim > dim. Same
 * class of bug already fixed for MLA (mla_run_state.c) and Qwen3.5/3.6
 * (qwen35_run_state.c) — see docs/ai/mistakes.md. Mirrors
 * qwen35_attention.c's q35_full_attn_forward() pattern (fixed generous
 * static buffers, own correctly-sized KV cache) minus the gate-doubling,
 * partial-rotary, and Q2_0-only quant specifics that don't apply here.
 */

#include "transformer/qwen3moe_attention.h"
#include "math/parallel_matmul.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include "transformer/dense_matmul_dispatch.h"
#include <math.h>
#include <string.h>

/* Generous fixed stack-buffer bounds, mirroring qwen35_attention.c's
 * Q35_MAX_* convention (fixed generous buffer, no runtime malloc). Sized
 * well beyond real Qwen3-MoE shapes (e.g. Qwen3-30B-A3B: q_width=32*128=
 * 4096, kv_width=4*128=512) with headroom for larger variants. */
#define QWEN3MOE_MAX_Q_WIDTH   16384
#define QWEN3MOE_MAX_KV_WIDTH   8192
#define QWEN3MOE_MAX_HEADS       128
#define QWEN3MOE_MAX_HEAD_DIM    512
#define QWEN3MOE_PREQ_BUF_SIZE 16384

void qwen3moe_attention_forward(RunState *s, const TransformerWeights *w,
                                 const Config *cfg, const MoEConfig *mc,
                                 int layer, int pos, ThreadPool *tp) {
    int dim      = cfg->dim;
    int n_heads  = cfg->n_heads;
    int n_kv_h   = cfg->n_kv_heads;
    int head_dim = mc->attn_head_dim;
    int kv_mul   = n_heads / n_kv_h;
    int q_width  = n_heads * head_dim;
    int kv_width = n_kv_h  * head_dim;
    int max_seq  = s->max_seq_len;

    static float q_buf[QWEN3MOE_MAX_Q_WIDTH];
    static float k_buf[QWEN3MOE_MAX_KV_WIDTH];
    static float v_buf[QWEN3MOE_MAX_KV_WIDTH];
    static float attn_concat[QWEN3MOE_MAX_HEADS * QWEN3MOE_MAX_HEAD_DIM];

    /* Step 1: RMSNorm */
    tn_rmsnorm(s->xb, s->x, w->rms_att_weight[layer], dim, cfg->rms_norm_eps);

    /* Step 2: Q/K/V projections — 3-way dispatch, same convention as
     * attention.c's generic path (this arch is not Q2_0-locked). */
    if (w->layers_are_ternary) {
        int8_t preq_buf[QWEN3MOE_PREQ_BUF_SIZE];
        TnPreqActivation preq;
        tn_preq_prepare(&preq, preq_buf, s->xb, dim);
        parallel_ternary_matmul_packed_preq(q_buf, s->xb, (const tn_u8 *)w->wq[layer], dim, q_width,  w->sq[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(k_buf, s->xb, (const tn_u8 *)w->wk[layer], dim, kv_width, w->sk[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(v_buf, s->xb, (const tn_u8 *)w->wv[layer], dim, kv_width, w->sv[layer], &preq, tp);
    } else {
        tn_dense_matmul_dispatch(q_buf, s->xb, w->wq[layer], w->wq_type[layer], dim, q_width,  tp);
        tn_dense_matmul_dispatch(k_buf, s->xb, w->wk[layer], w->wk_type[layer], dim, kv_width, tp);
        tn_dense_matmul_dispatch(v_buf, s->xb, w->wv[layer], w->wv_type[layer], dim, kv_width, tp);
    }

    /* Step 3: per-head QK-norm, before RoPE */
    for (int h = 0; h < n_heads; h++) {
        float *qh = q_buf + (size_t)h * head_dim;
        tn_rmsnorm(qh, qh, w->qwen3moe_attn_q_norm[layer], head_dim, cfg->rms_norm_eps);
    }
    for (int kh = 0; kh < n_kv_h; kh++) {
        float *khp = k_buf + (size_t)kh * head_dim;
        tn_rmsnorm(khp, khp, w->qwen3moe_attn_k_norm[layer], head_dim, cfg->rms_norm_eps);
    }

    /* Step 4: full (non-partial) RoPE with YaRN support, sized for this
     * arch's real head_dim via s->qwen3moe_rope_freq
     * (qwen3moe_run_state_alloc) — identical convention to
     * attention_forward()'s generic Step 3. */
    {
        float corr[2] = {0.0f, 0.0f};
        float freq_scale  = cfg->rope_freq_scale;
        float ext_factor  = cfg->rope_yarn_ext_factor;
        float attn_factor = cfg->rope_yarn_attn_factor;
        if (ext_factor != 0.0f) {
            static const float M_PI_F = 3.14159265358979323846f;
            float log_base = logf(cfg->rope_theta);
            float start = floorf((float)head_dim * logf((float)cfg->rope_orig_ctx_len
                                 / (cfg->rope_yarn_beta_fast * 2.0f * M_PI_F)) / (2.0f * log_base));
            float end   = ceilf ((float)head_dim * logf((float)cfg->rope_orig_ctx_len
                                 / (cfg->rope_yarn_beta_slow * 2.0f * M_PI_F)) / (2.0f * log_base));
            corr[0] = start < 0.0f ? 0.0f : (start > (float)(head_dim-1) ? (float)(head_dim-1) : start);
            corr[1] = end   < 0.0f ? 0.0f : (end   > (float)(head_dim-1) ? (float)(head_dim-1) : end  );
        }
        apply_rope(q_buf, k_buf, s->qwen3moe_rope_freq, head_dim, pos, n_heads, n_kv_h,
                   freq_scale, ext_factor, attn_factor, corr);
    }

    /* Step 5: write K/V into this layer's own correctly-sized cache
     * (qwen3moe_run_state_alloc) — flat per-layer buffer, [kv_head][pos][d]. */
    int mapped_pos = sw_map_position(&s->sw, pos);
    for (int kh = 0; kh < n_kv_h; kh++) {
        size_t off = ((size_t)kh * max_seq + (size_t)mapped_pos) * head_dim;
        memcpy(&s->qwen3moe_key_cache[layer][off],   &k_buf[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
        memcpy(&s->qwen3moe_value_cache[layer][off], &v_buf[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
    }
    sw_advance(&s->sw);

    /* Step 6: attention per head (scores + softmax + weighted sum) */
    int valid_ctx = sw_valid_count(&s->sw, pos);
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        float *q_h = q_buf + (size_t)h * head_dim;
        int    kh  = h / kv_mul;
        float *att = s->att + (size_t)h * max_seq;

        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
            att[t] = tn_vec_dot(q_h, &s->qwen3moe_key_cache[layer][off], head_dim) * inv_sqrt_hd;
        }
        tn_softmax(att, valid_ctx);

        float *out_h = attn_concat + (size_t)h * head_dim;
        memset(out_h, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
            tn_vec_saxpy(out_h, att[t], &s->qwen3moe_value_cache[layer][off], head_dim);
        }
    }

    /* Step 7: output projection (q_width -> dim) + residual. wo's input
     * width is q_width (n_heads*head_dim), NOT dim — attn_output.weight is
     * shaped [dim x q_width] for this arch, unlike the generic dense path's
     * [dim x dim] assumption (see weights_from_gguf_qwen3moe). */
    if (w->layers_are_ternary) {
        parallel_ternary_matmul_packed(s->xb, attn_concat, (const tn_u8 *)w->wo[layer], q_width, dim, w->so[layer], tp);
    } else {
        tn_dense_matmul_dispatch(s->xb, attn_concat, w->wo[layer], w->wo_type[layer], q_width, dim, tp);
    }
    tn_vec_add(s->x, s->x, s->xb, dim);
}
