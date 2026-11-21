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
 * qwen35_attention.c's q35_full_attn_forward() pattern (own correctly-sized
 * KV cache and scratch in the RunState) minus the gate-doubling,
 * partial-rotary, and Q2_0-only quant specifics that don't apply here.
 */

#include "transformer/qwen3moe_attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/clamp.h"
#include "core/lora.h"
#include "core/platform.h"
#include "core/step_timing.h"
#include "math/batched_matmul.h"
#include "math/lora_matmul.h"
#include "math/parallel_matmul.h"
#include "math/rope.h"
#include "math/simd_core_ops.h"
#include "math/simd_vec_ops.h"
#include "runtime/spec_scratch.h" /* reads the batch scratch fields */
#include "transformer/dense_matmul_dispatch.h"

#define QWEN3MOE_PREQ_BUF_SIZE 16384

/* Q, K and V of one token (reads s->xb): the packed ternary kernels share one activation
 * quantisation, any other weight type goes through the dense dispatch. */
static void project_qkv(RunState *s, const TransformerWeights *w, int layer, int dim, float *q_buf,
                        float *k_buf, float *v_buf, int q_width, int kv_width, ThreadPool *tp) {
    if (w->layers_are_ternary) {
        int8_t preq_buf[QWEN3MOE_PREQ_BUF_SIZE];
        TnPreqActivation preq;
        tn_preq_prepare(&preq, preq_buf, s->xb, dim);
        parallel_ternary_matmul_packed_preq(q_buf, s->xb, (const tn_u8 *)w->attn.wq[layer], dim,
                                            q_width, w->attn.sq[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(k_buf, s->xb, (const tn_u8 *)w->attn.wk[layer], dim,
                                            kv_width, w->attn.sk[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(v_buf, s->xb, (const tn_u8 *)w->attn.wv[layer], dim,
                                            kv_width, w->attn.sv[layer], &preq, tp);
    } else {
        tn_dense_matmul_dispatch(q_buf, s->xb, w->attn.wq[layer], w->attn.wq_type[layer], dim,
                                 q_width, tp);
        tn_dense_matmul_dispatch(k_buf, s->xb, w->attn.wk[layer], w->attn.wk_type[layer], dim,
                                 kv_width, tp);
        tn_dense_matmul_dispatch(v_buf, s->xb, w->attn.wv[layer], w->attn.wv_type[layer], dim,
                                 kv_width, tp);
    }
}

/* This architecture's shapes: head_dim comes from the GGUF (mc->attn_head_dim), so q_width and
 * kv_width are independent of dim. */
typedef struct {
    int dim, n_heads, n_kv_h, head_dim, kv_mul, q_width, kv_width, max_seq;
} Q3Dims;

static Q3Dims q3_dims(const RunState *s, const Config *cfg, const MoEConfig *mc) {
    Q3Dims d;
    d.dim      = cfg->dim;
    d.n_heads  = cfg->n_heads;
    d.n_kv_h   = cfg->n_kv_heads;
    d.head_dim = mc->attn_head_dim;
    d.kv_mul   = d.n_heads / d.n_kv_h;
    d.q_width  = d.n_heads * d.head_dim;
    d.kv_width = d.n_kv_h * d.head_dim;
    d.max_seq  = s->max_seq_len;
    return d;
}

/* Per-head QK-norm of one row, before RoPE. */
static void q3_qk_norm(const TransformerWeights *w, const Config *cfg, const Q3Dims *d, int layer,
                       float *q, float *k) {
    for (int h = 0; h < d->n_heads; h++) {
        float *qh = q + (size_t)h * d->head_dim;
        tn_rmsnorm(qh, qh, w->qwen3moe.attn_q_norm[layer], d->head_dim, cfg->rms_norm_eps);
    }
    for (int kh = 0; kh < d->n_kv_h; kh++) {
        float *khp = k + (size_t)kh * d->head_dim;
        tn_rmsnorm(khp, khp, w->qwen3moe.attn_k_norm[layer], d->head_dim, cfg->rms_norm_eps);
    }
}

/* Full (non-partial) RoPE with YaRN support on one row at `pos`, sized for this arch's real
 * head_dim via s->qwen3moe.rope_freq (qwen3moe_run_state_alloc): the convention of
 * attention_forward()'s generic step 3. */
static void q3_rope(const RunState *s, const Config *cfg, const Q3Dims *d, float *q, float *k,
                    int pos) {
    float corr[2];
    rope_corr_from_config(cfg, d->head_dim, corr);
    apply_rope_ex(q, k, s->qwen3moe.rope_freq, d->head_dim, pos, d->n_heads, d->n_kv_h,
                  cfg->rope_freq_scale, cfg->rope_yarn_ext_factor, cfg->rope_yarn_attn_factor, corr,
                  cfg->rope_layout);
}

/* Writes one row's K/V into this layer's own correctly-sized cache (qwen3moe_run_state_alloc): a
 * flat per-layer buffer, [kv_head][pos][d]. */
static void q3_store_kv(RunState *s, const Q3Dims *d, int layer, int mapped_pos, const float *k,
                        const float *v) {
    for (int kh = 0; kh < d->n_kv_h; kh++) {
        size_t off = ((size_t)kh * d->max_seq + (size_t)mapped_pos) * d->head_dim;
        memcpy(&s->qwen3moe.key_cache[layer][off], &k[(size_t)kh * d->head_dim],
               (size_t)d->head_dim * sizeof(float));
        memcpy(&s->qwen3moe.value_cache[layer][off], &v[(size_t)kh * d->head_dim],
               (size_t)d->head_dim * sizeof(float));
    }
}

/* Scores + softmax + weighted sum for every query head of one row at `pos`: q holds the row's
 * q_width floats, out receives its q_width attention output. */
static void q3_attend_row(const RunState *s, const Q3Dims *d, int layer, int pos, const float *q,
                          float *out) {
    int valid_ctx     = sw_valid_count(&s->sw, pos);
    float inv_sqrt_hd = 1.0f / sqrtf((float)d->head_dim);
    for (int h = 0; h < d->n_heads; h++) {
        const float *q_h = q + (size_t)h * d->head_dim;
        int kh           = h / d->kv_mul;
        float *att       = s->att + (size_t)h * d->max_seq;

        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off   = ((size_t)kh * d->max_seq + (size_t)mapped_t) * d->head_dim;
            att[t] = tn_vec_dot(q_h, &s->qwen3moe.key_cache[layer][off], d->head_dim) * inv_sqrt_hd;
        }
        tn_softmax(att, valid_ctx);

        float *out_h = out + (size_t)h * d->head_dim;
        memset(out_h, 0, (size_t)d->head_dim * sizeof(float));
        for (int t = 0; t < valid_ctx; t++) {
            int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            int mapped_t = sw_map_position(&s->sw, hist);
            size_t off   = ((size_t)kh * d->max_seq + (size_t)mapped_t) * d->head_dim;
            tn_vec_saxpy(out_h, att[t], &s->qwen3moe.value_cache[layer][off], d->head_dim);
        }
    }
}

/* Steps 1-3 (single token): RMSNorm, the Q/K/V projections (3-way dispatch, same convention as
 * attention.c's generic path; this arch is not Q2_0-locked), the LoRA deltas on the wide Q and K/V
 * (Phase 19-B, 2026-10-19: Qwen3 adapters were silently ignored) and per-head QK-norm. */
static void q3_project(RunState *s, const TransformerWeights *w, const Config *cfg, const Q3Dims *d,
                       int layer, ThreadPool *tp, float *q_buf, float *k_buf, float *v_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->attn.norm[layer], d->dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    project_qkv(s, w, layer, d->dim, q_buf, k_buf, v_buf, d->q_width, d->kv_width, tp);
    if (s->active_lora) {
        lora_apply(q_buf, s->xb, lora_mod(s->active_lora->q, layer), d->dim, d->q_width, tp);
        lora_apply(k_buf, s->xb, lora_mod(s->active_lora->k, layer), d->dim, d->kv_width, tp);
        lora_apply(v_buf, s->xb, lora_mod(s->active_lora->v, layer), d->dim, d->kv_width, tp);
    }
    q3_qk_norm(w, cfg, d, layer, q_buf, k_buf);
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
}

/* Step 7 (single token): output projection (q_width -> dim) + residual. wo's input width is
 * q_width (n_heads*head_dim), NOT dim: attn_output.weight is shaped [dim x q_width] for this arch,
 * unlike the generic dense path's [dim x dim] assumption (see weights_from_gguf_qwen3moe). */
static void q3_output(RunState *s, const TransformerWeights *w, const Q3Dims *d, int layer,
                      ThreadPool *tp, const float *attn_concat) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->layers_are_ternary)
        parallel_ternary_matmul_packed(s->xb, attn_concat, (const tn_u8 *)w->attn.wo[layer],
                                       d->q_width, d->dim, w->attn.so[layer], tp);
    else
        tn_dense_matmul_dispatch(s->xb, attn_concat, w->attn.wo[layer], w->attn.wo_type[layer],
                                 d->q_width, d->dim, tp);
    if (s->active_lora)
        lora_apply(s->xb, attn_concat, lora_mod(s->active_lora->o, layer), d->q_width, d->dim, tp);
    tn_vec_add(s->x, s->x, s->xb, d->dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

void qwen3moe_attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                                const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    Q3Dims d = q3_dims(s, cfg, mc);

    /* Per-RunState scratch (qwen3moe_run_state_alloc): no shared static, no width cap. */
    float *q_buf       = s->qwen3moe.scratch;
    float *k_buf       = q_buf + d.q_width;
    float *v_buf       = k_buf + d.kv_width;
    float *attn_concat = v_buf + d.kv_width;

    q3_project(s, w, cfg, &d, layer, tp, q_buf, k_buf, v_buf);

    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    q3_rope(s, cfg, &d, q_buf, k_buf, pos);
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    q3_store_kv(s, &d, layer, sw_map_position(&s->sw, pos), k_buf, v_buf);
    sw_advance(&s->sw);
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    q3_attend_row(s, &d, layer, pos, q_buf, attn_concat);
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);

    q3_output(s, w, &d, layer, tp, attn_concat);
}

/* ── Phase 18 (speculative decoding): batched multi-token Qwen3-MoE attention
 *
 * Same write-phase/read-phase restructuring as attention_forward_batch()
 * (attention.c) -- see that function's header comment for the general
 * design and correctness argument.
 *
 * This arch's q_width/kv_width are independent of dim/hidden_dim (may
 * exceed both -- see this file's top-of-file comment), so it cannot reuse
 * SpecBatchScratch's generic dim/hidden_dim-wide buffers the way the dense
 * and MLA batched paths do. Instead it heap-allocates its own N-wide Q/K/V/
 * attention-output scratch here, sized by this layer's *actual* q_width/
 * kv_width -- freed at the end of this call. */
/* One batch call's scratch: Q/K/V and attention output for all rows (heap, sized by this layer's
 * actual widths). */
typedef struct {
    float *q, *k, *v, *concat;
} Q3Batch;

static int q3_batch_alloc(Q3Batch *b, const Q3Dims *d, int n_tokens) {
    b->q      = (float *)malloc((size_t)n_tokens * d->q_width * sizeof(float));
    b->k      = (float *)malloc((size_t)n_tokens * d->kv_width * sizeof(float));
    b->v      = (float *)malloc((size_t)n_tokens * d->kv_width * sizeof(float));
    b->concat = (float *)malloc((size_t)n_tokens * d->q_width * sizeof(float));
    return b->q && b->k && b->v && b->concat;
}

static void q3_batch_free(Q3Batch *b) {
    free(b->q);
    free(b->k);
    free(b->v);
    free(b->concat);
}

/* Steps 1-2 over all rows: per-token RMSNorm, then the Q/K/V projections (a real batch for
 * ternary/F16, sequential fallback for everything else: documented Phase 18-B scope) and the LoRA
 * deltas. */
static void q3_project_rows(RunState *s, SpecBatchScratch *sb, const TransformerWeights *w,
                            const Config *cfg, const Q3Dims *d, int layer, int n_tokens,
                            ThreadPool *tp, Q3Batch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        tn_rmsnorm(sb->xb + (size_t)k * d->dim, sb->x + (size_t)k * d->dim, w->attn.norm[layer],
                   d->dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->layers_are_ternary) {
        tn_ternary_matmul_packed_batch(b->q, sb->xb, (const tn_u8 *)w->attn.wq[layer], d->dim,
                                       d->q_width, w->attn.sq[layer], n_tokens, tp);
        tn_ternary_matmul_packed_batch(b->k, sb->xb, (const tn_u8 *)w->attn.wk[layer], d->dim,
                                       d->kv_width, w->attn.sk[layer], n_tokens, tp);
        tn_ternary_matmul_packed_batch(b->v, sb->xb, (const tn_u8 *)w->attn.wv[layer], d->dim,
                                       d->kv_width, w->attn.sv[layer], n_tokens, tp);
    } else {
        tn_dense_matmul_dispatch_batch(b->q, sb->xb, w->attn.wq[layer], w->attn.wq_type[layer],
                                       d->dim, d->q_width, n_tokens, tp);
        tn_dense_matmul_dispatch_batch(b->k, sb->xb, w->attn.wk[layer], w->attn.wk_type[layer],
                                       d->dim, d->kv_width, n_tokens, tp);
        tn_dense_matmul_dispatch_batch(b->v, sb->xb, w->attn.wv[layer], w->attn.wv_type[layer],
                                       d->dim, d->kv_width, n_tokens, tp);
    }
    if (s->active_lora) {
        lora_apply_batch(b->q, sb->xb, lora_mod(s->active_lora->q, layer), d->dim, d->q_width,
                         n_tokens, tp);
        lora_apply_batch(b->k, sb->xb, lora_mod(s->active_lora->k, layer), d->dim, d->kv_width,
                         n_tokens, tp);
        lora_apply_batch(b->v, sb->xb, lora_mod(s->active_lora->v, layer), d->dim, d->kv_width,
                         n_tokens, tp);
    }
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
}

/* Steps 3 (QK-norm) and 4 (RoPE) per token, bracketed together as step 9 (see
 * attention_forward_batch()'s comment on why QK-norm's small remaining cost is grouped with RoPE
 * rather than reopening step 5). */
static void q3_norm_rope_rows(const RunState *s, const TransformerWeights *w, const Config *cfg,
                              const Q3Dims *d, int layer, int pos, int n_tokens, Q3Batch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        float *q_k = b->q + (size_t)k * d->q_width;
        float *k_k = b->k + (size_t)k * d->kv_width;
        q3_qk_norm(w, cfg, d, layer, q_k, k_k);
        q3_rope(s, cfg, d, q_k, k_k, pos + k);
    }
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);
}

/* Step 7 over all rows: batched output projection (q_width -> dim) + residual. */
static void q3_output_rows(RunState *s, SpecBatchScratch *sb, const TransformerWeights *w,
                           const Q3Dims *d, int layer, int n_tokens, ThreadPool *tp,
                           const Q3Batch *b) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->layers_are_ternary)
        tn_ternary_matmul_packed_batch(sb->xb, b->concat, (const tn_u8 *)w->attn.wo[layer],
                                       d->q_width, d->dim, w->attn.so[layer], n_tokens, tp);
    else
        tn_dense_matmul_dispatch_batch(sb->xb, b->concat, w->attn.wo[layer], w->attn.wo_type[layer],
                                       d->q_width, d->dim, n_tokens, tp);
    if (s->active_lora)
        lora_apply_batch(sb->xb, b->concat, lora_mod(s->active_lora->o, layer), d->q_width, d->dim,
                         n_tokens, tp);
    tn_vec_add(sb->x, sb->x, sb->xb, n_tokens * d->dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

TernaryError qwen3moe_attention_forward_batch(RunState *s, SpecBatchScratch *sb,
                                              const TransformerWeights *w, const Config *cfg,
                                              const MoEConfig *mc, int layer, int pos, int n_tokens,
                                              ThreadPool *tp) {
    Q3Dims d = q3_dims(s, cfg, mc);
    Q3Batch b;
    if (!q3_batch_alloc(&b, &d, n_tokens)) {
        q3_batch_free(&b);
        return TN_ERR_OOM;
    }

    /* Whole-batch-call step-timing granularity (Stage 5.5); same 4/5/9/10/11/12 mapping as
     * attention_forward_batch() -- see that function's comment. */
    q3_project_rows(s, sb, w, cfg, &d, layer, n_tokens, tp, &b);
    q3_norm_rope_rows(s, w, cfg, &d, layer, pos, n_tokens, &b);

    /* Step 5: write K/V into this layer's own cache, per token (sw_advance kept for parity; see
     * attention_forward_batch()'s comment). */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        q3_store_kv(s, &d, layer, sw_map_position(&s->sw, pos + k), b.k + (size_t)k * d.kv_width,
                    b.v + (size_t)k * d.kv_width);
        sw_advance(&s->sw);
    }
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);

    /* Read phase: attention per token (not batched across k). */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        q3_attend_row(s, &d, layer, pos + k, b.q + (size_t)k * d.q_width,
                      b.concat + (size_t)k * d.q_width);
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);

    q3_output_rows(s, sb, w, &d, layer, n_tokens, tp, &b);
    q3_batch_free(&b);
    return TN_OK;
}
