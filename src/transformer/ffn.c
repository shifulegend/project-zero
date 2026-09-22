#include "transformer/ffn.h"
#include "transformer/moe_ffn.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "math/batched_matmul.h"
#include "core/debug.h"
#include "core/step_timing.h"
#include "core/weights.h"
#include "transformer/dense_matmul_dispatch.h"
#include <string.h>

/* Maximum input dimension supported for layer-level preq stack buffer. */
#define FFN_PREQ_BUF_SIZE 16384

void ffn_forward(RunState *s, const TransformerWeights *w,
                 const Config *cfg, const MoEConfig *mc,
                 int layer, ThreadPool *tp) {
    /* Phase 17: Route MoE layers to their specialist handler.
     * Dense layers (and all BitNet/Llama models) fall through to the
     * standard SwiGLU path below — zero overhead for dense models. */
    if (moe_layer_is_moe(mc, layer)) {
        moe_ffn_forward(s, w, cfg, mc, layer, tp);
        return;
    }

    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;

    /* Step 1: RMSNorm — normalize s->x into s->xb */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->rms_ffn_weight[layer], dim, cfg->rms_norm_eps);
    if (t_step) {
        tn_step_timing_add(TN_STEP_13_PRE_FFN_RMSNORM,
                           tn_step_timing_now_ns() - t_step);
    }
    /* Step 13: pre-FFN RMSNorm output (dense layer) */
    DBG_DUMP(layer, "dense_ffn_norm", s->xb, dim);

    /* Step 2: Gate & Up projections
     * Layer-level preq: quantise s->xb once, reuse for both projections.
     * Saves 1 redundant quantisation per FFN layer (2 calls → 1 quantise). */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->layers_are_ternary) {
        int8_t preq_buf[FFN_PREQ_BUF_SIZE];
        TnPreqActivation preq;
        tn_preq_prepare(&preq, preq_buf, s->xb, dim);
        parallel_ternary_matmul_packed_preq(s->hb,  s->xb, (const tn_u8 *)w->w1[layer], dim, hidden_dim, w->s1[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(s->hb2, s->xb, (const tn_u8 *)w->w3[layer], dim, hidden_dim, w->s3[layer], &preq, tp);
    } else if (tn_is_q4k_family(w->w1_type[layer]) && tn_is_q4k_family(w->w3_type[layer])) {
        /* Gate and up both read s->xb — quantize to Q8K once and reuse.
         * Bug fixed 2026-07-31: this branch previously called
         * tn_dense_matmul_dispatch() twice, each independently re-quantizing
         * s->xb via parallel_matmul_q4k, despite the comment above already
         * claiming the shared-quantize optimization — it had only ever been
         * wired up for the ternary branch, not this (Q4_K dense) one.
         * Extended 2026-07-31 to also cover WEIGHT_TYPE_Q4K_X8 (the repacked
         * multi-row-GEMV variant, matmul_q4k_x8.c) — both consume the same
         * Q8K activation format, so the shared-quantize win applies to either. */
        TnQ8KActBlock acts[FFN_PREQ_BUF_SIZE / TN_Q8K_BLOCK];
        int n_blocks = dim / TN_Q8K_BLOCK;
        tn_quantize_q8k(acts, s->xb, n_blocks);
        tn_dense_matmul_dispatch_preq(s->hb,  acts, w->w1[layer], w->w1_type[layer], dim, hidden_dim, tp);
        tn_dense_matmul_dispatch_preq(s->hb2, acts, w->w3[layer], w->w3_type[layer], dim, hidden_dim, tp);
    } else {
        tn_dense_matmul_dispatch(s->hb,  s->xb, w->w1[layer], w->w1_type[layer], dim, hidden_dim, tp);
        tn_dense_matmul_dispatch(s->hb2, s->xb, w->w3[layer], w->w3_type[layer], dim, hidden_dim, tp);
    }

    /* Apply activation to Gate */
    if (cfg->act_type == 1) {
        tn_relu2(s->hb, hidden_dim);
    } else {
        tn_silu(s->hb, hidden_dim);
    }
    /* Step 14: gate projection after SiLU activation */
    DBG_DUMP(layer, "dense_gate_act", s->hb, hidden_dim);
    /* Step 14: up projection (before multiply) */
    DBG_DUMP(layer, "dense_up", s->hb2, hidden_dim);

    /* Step 4: Element-wise multiply (SwiGLU/ReGLU gate) — s->hb = s->hb * s->hb2 */
    tn_vec_mul(s->hb, s->hb, s->hb2, hidden_dim);
    /* Step 14: gate * up (SwiGLU intermediate) */
    DBG_DUMP(layer, "dense_swiglu", s->hb, hidden_dim);

    /* Step 5: Apply ffn_sub_norm (BitNet) if present */
    if (w->rms_ffn_sub_norm && w->rms_ffn_sub_norm[layer]) {
        tn_rmsnorm(s->hb, s->hb, w->rms_ffn_sub_norm[layer], hidden_dim, cfg->rms_norm_eps);
    }

    /* Step 6: Down projection with Dynamic Dispatch */
    if (w->layers_are_ternary) {
        parallel_ternary_matmul_packed(s->xb, s->hb, (const tn_u8 *)w->w2[layer], hidden_dim, dim, w->s2[layer], tp);
    } else {
        tn_dense_matmul_dispatch(s->xb, s->hb, w->w2[layer], w->w2_type[layer], hidden_dim, dim, tp);
    }
    /* Step 14: dense down projection output (pre-residual) */
    DBG_DUMP(layer, "dense_down", s->xb, dim);

    /* Step 7: Residual connection — s->x += s->xb */
    tn_vec_add(s->x, s->x, s->xb, dim);
    if (t_step) {
        tn_step_timing_add(TN_STEP_14_DENSE_FFN,
                           tn_step_timing_now_ns() - t_step);
    }
}

/* ── Phase 18 (speculative decoding): batched multi-token FFN ──────────────
 * See include/transformer/ffn.h for the MoE-fallback rationale. */
TernaryError ffn_forward_batch(RunState *s, SpecBatchScratch *sb,
                                const TransformerWeights *w, const Config *cfg,
                                const MoEConfig *mc, int layer, int n_tokens, ThreadPool *tp) {
    int dim = cfg->dim;
    int hidden_dim = cfg->hidden_dim;

    if (moe_layer_is_moe(mc, layer)) {
        for (int k = 0; k < n_tokens; k++) {
            memcpy(s->x, sb->x + (size_t)k * dim, (size_t)dim * sizeof(float));
            moe_ffn_forward(s, w, cfg, mc, layer, tp);
            memcpy(sb->x + (size_t)k * dim, s->x, (size_t)dim * sizeof(float));
        }
        return TN_OK;
    }

    /* Dense SwiGLU FFN — fully batched. */
    for (int k = 0; k < n_tokens; k++) {
        tn_rmsnorm(sb->xb + (size_t)k * dim, sb->x + (size_t)k * dim,
                   w->rms_ffn_weight[layer], dim, cfg->rms_norm_eps);
    }

    if (w->layers_are_ternary) {
        tn_ternary_matmul_packed_batch(sb->hb,  sb->xb, (const tn_u8 *)w->w1[layer],
                                        dim, hidden_dim, w->s1[layer], n_tokens, tp);
        tn_ternary_matmul_packed_batch(sb->hb2, sb->xb, (const tn_u8 *)w->w3[layer],
                                        dim, hidden_dim, w->s3[layer], n_tokens, tp);
    } else {
        tn_dense_matmul_dispatch_batch(sb->hb,  sb->xb, w->w1[layer], w->w1_type[layer],
                                        dim, hidden_dim, n_tokens, tp);
        tn_dense_matmul_dispatch_batch(sb->hb2, sb->xb, w->w3[layer], w->w3_type[layer],
                                        dim, hidden_dim, n_tokens, tp);
    }

    /* Activation + SwiGLU gate: purely elementwise, so one call spanning
     * the full n_tokens*hidden_dim block is equivalent to n_tokens
     * per-row calls. */
    if (cfg->act_type == 1) {
        tn_relu2(sb->hb, n_tokens * hidden_dim);
    } else {
        tn_silu(sb->hb, n_tokens * hidden_dim);
    }
    tn_vec_mul(sb->hb, sb->hb, sb->hb2, n_tokens * hidden_dim);

    if (w->rms_ffn_sub_norm && w->rms_ffn_sub_norm[layer]) {
        for (int k = 0; k < n_tokens; k++) {
            float *hb_k = sb->hb + (size_t)k * hidden_dim;
            tn_rmsnorm(hb_k, hb_k, w->rms_ffn_sub_norm[layer], hidden_dim, cfg->rms_norm_eps);
        }
    }

    if (w->layers_are_ternary) {
        tn_ternary_matmul_packed_batch(sb->xb, sb->hb, (const tn_u8 *)w->w2[layer],
                                        hidden_dim, dim, w->s2[layer], n_tokens, tp);
    } else {
        tn_dense_matmul_dispatch_batch(sb->xb, sb->hb, w->w2[layer], w->w2_type[layer],
                                        hidden_dim, dim, n_tokens, tp);
    }

    tn_vec_add(sb->x, sb->x, sb->xb, n_tokens * dim);

    return TN_OK;
}
