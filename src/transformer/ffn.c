#include "transformer/ffn.h"
#include "transformer/moe_ffn.h"
#include "math/parallel_matmul.h"
#include "math/matmul_f16.h"
#include "math/matmul_q2_0.h"
#include "math/simd_dispatch.h"
#include "core/debug.h"
#include "core/step_timing.h"
#include "core/weights.h"
#include <stdlib.h>

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
        int8_t stack_buf[FFN_PREQ_BUF_SIZE];
        int8_t *preq_buf = stack_buf;
        if (dim > FFN_PREQ_BUF_SIZE) {
            preq_buf = (int8_t *)malloc((size_t)dim * sizeof(int8_t));
        }
        TnPreqActivation preq;
        tn_preq_prepare(&preq, preq_buf, s->xb, dim);
        parallel_ternary_matmul_packed_preq(s->hb,  s->xb, (const tn_u8 *)w->w1[layer], dim, hidden_dim, w->s1[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(s->hb2, s->xb, (const tn_u8 *)w->w3[layer], dim, hidden_dim, w->s3[layer], &preq, tp);
        if (preq_buf != stack_buf) {
            free(preq_buf);
        }
    } else if (w->layer_weight_type == WEIGHT_TYPE_F16) {
        parallel_matmul_f16(s->hb,  s->xb, (const tn_u16 *)w->w1[layer], dim, hidden_dim, tp);
        parallel_matmul_f16(s->hb2, s->xb, (const tn_u16 *)w->w3[layer], dim, hidden_dim, tp);
    } else if (w->layer_weight_type == WEIGHT_TYPE_Q2_0) {
        parallel_matmul_q2_0(s->hb,  s->xb, (const uint8_t *)w->w1[layer], dim, hidden_dim, tp);
        parallel_matmul_q2_0(s->hb2, s->xb, (const uint8_t *)w->w3[layer], dim, hidden_dim, tp);
    } else {
        parallel_matmul_float32(s->hb,  s->xb, (const float *)w->w1[layer], dim, hidden_dim, tp);
        parallel_matmul_float32(s->hb2, s->xb, (const float *)w->w3[layer], dim, hidden_dim, tp);
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
    } else if (w->layer_weight_type == WEIGHT_TYPE_F16) {
        parallel_matmul_f16(s->xb, s->hb, (const tn_u16 *)w->w2[layer], hidden_dim, dim, tp);
    } else if (w->layer_weight_type == WEIGHT_TYPE_Q2_0) {
        parallel_matmul_q2_0(s->xb, s->hb, (const uint8_t *)w->w2[layer], hidden_dim, dim, tp);
    } else {
        parallel_matmul_float32(s->xb, s->hb, (const float *)w->w2[layer], hidden_dim, dim, tp);
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
