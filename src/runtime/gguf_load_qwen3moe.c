/* Qwen3-MoE (e.g. Qwen3-30B-A3B) weight loader. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/fp16.h"
#include "core/gguf_i2s.h"
#include "core/gguf_quant.h"
#include "core/log.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include "core/unpack.h"
#include "math/matmul_q4k_x8.h"
#include "math/rope.h"
#include "memory/mapped_file.h"
#include "runtime/gguf_load_util.h"
#include "runtime/gguf_loader.h"
#include "runtime/hardware_profile.h"
#include "runtime/mamba_state.h"

/* ── weights_from_gguf_qwen3moe() — Qwen3-MoE (e.g. Qwen3-30B-A3B) loader ─
 *
 * Plain GQA attention (per-head QK-norm, no MLA compression) + routed MoE
 * FFN (no shared experts, no leading dense layers). head_dim is independent
 * of dim/n_heads (see MoEConfig.attn_head_dim), so wq/wo widths are q_rows-
 * based, NOT dim*dim like the generic dense loader assumes — consumed by
 * qwen3moe_attention.c's dedicated forward function, not attention.c's
 * generic GQA branch (see that file's header comment for why). */

/* Generic projection loader: F16/F32 zero-copy from mmap, other quantized
 * types dequantized to a heap F32 buffer — same 3-way dispatch as the
 * generic dense loader's LOAD_PROJ macro below (this arch is not Q2_0-
 * locked like qwen35, nor Q4_K-heap-copied like deepseek2's MLA path).
 * Sets type_out[l], a per-layer, per-projection array (e.g. w->wq_type) —
 * see weights.h's incident note on why a single shared flag is wrong: the
 * same class of bug (mixed quant types across projections/layers silently
 * misdispatching to the wrong kernel) applies here too if this arch is ever
 * loaded from a mixed-precision GGUF, even though today's F16/F32-only
 * dispatch below doesn't yet have a Q4_K zero-copy branch to trigger it. */
#define Q3MOE_LOAD_PROJ(field, tname, n_elems, type_out)                                           \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        if (_t->type == GGUF_TYPE_F16) {                                                           \
            (field)[l]    = (tn_i8 *)_t->data;                                                     \
            (type_out)[l] = WEIGHT_TYPE_F16;                                                       \
        } else if (_t->type == GGUF_TYPE_F32) {                                                    \
            (field)[l]    = (tn_i8 *)_t->data;                                                     \
            (type_out)[l] = WEIGHT_TYPE_F32;                                                       \
        } else {                                                                                   \
            float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                               \
            if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                \
            (field)[l]    = (tn_i8 *)_f;                                                           \
            (type_out)[l] = WEIGHT_TYPE_F32;                                                       \
        }                                                                                          \
    } while (0)

TernaryError weights_from_gguf_qwen3moe(TransformerWeights *w, const Config *cfg,
                                        const GGUFHeader *hdr, const MoEConfig *mc,
                                        GGUFWeightStore *store) {

    int dim         = cfg->dim;
    int nl          = cfg->n_layers;
    int n_heads     = cfg->n_heads;
    int n_kv_heads  = cfg->n_kv_heads;
    int head_dim    = mc->attn_head_dim; /* NOT dim/n_heads for this arch */
    int q_rows      = n_heads * head_dim;
    int kv_rows     = n_kv_heads * head_dim;
    int num_experts = mc->num_experts;
    int exp_hid     = mc->expert_hidden_dim;
    char name_buf[128];

    w->expert_quant.w2_per_layer  = (int *)calloc((size_t)nl, sizeof(int));
    w->expert_quant.w13_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    if (!w->expert_quant.w2_per_layer || !w->expert_quant.w13_per_layer) return TN_ERR_OOM;

    for (int l = 0; l < nl; l++) {
        /* Attention/FFN norms */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
        w->attn.rms_att_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->attn.rms_att_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
        w->ffn.rms_ffn_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->ffn.rms_ffn_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* Per-head QK-norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q_norm.weight", l);
        w->qwen3moe.attn_q_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)head_dim, store);
        if (!w->qwen3moe.attn_q_norm[l]) return TN_ERR_INVALID_WEIGHTS;

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k_norm.weight", l);
        w->qwen3moe.attn_k_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)head_dim, store);
        if (!w->qwen3moe.attn_k_norm[l]) return TN_ERR_INVALID_WEIGHTS;

        /* GQA attention projections — q_rows/kv_rows widths (NOT dim*dim;
         * head_dim is independent of dim/n_heads for this arch). */
        Q3MOE_LOAD_PROJ(w->attn.wq, "attn_q.weight", (size_t)q_rows * (size_t)dim, w->attn.wq_type);
        Q3MOE_LOAD_PROJ(w->attn.wk, "attn_k.weight", (size_t)kv_rows * (size_t)dim,
                        w->attn.wk_type);
        Q3MOE_LOAD_PROJ(w->attn.wv, "attn_v.weight", (size_t)kv_rows * (size_t)dim,
                        w->attn.wv_type);
        Q3MOE_LOAD_PROJ(w->attn.wo, "attn_output.weight", (size_t)dim * (size_t)q_rows,
                        w->attn.wo_type);
        w->attn.sq[l] = w->attn.sk[l] = w->attn.sv[l] = w->attn.so[l] = 1.0f;

        /* Router (F32, zero-copy) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_inp.weight", l);
        const GGUFTensor *tgate = gguf_find_tensor(hdr, name_buf);
        if (!tgate) {
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);
            return TN_ERR_INVALID_WEIGHTS;
        }
        if (tgate->type != GGUF_TYPE_F32) {
            TN_LOG_ERR("gguf_loader", "[gguf_loader] ffn_gate_inp.weight must be F32 (got %d)\n",
                       (int)tgate->type);
            return TN_ERR_INVALID_WEIGHTS;
        }
        w->moe.gate_w[l] = (tn_i8 *)tgate->data;
        w->moe.gate_s[l] = 1.0f;

        /* Stacked routed expert weights — kept quantized in mmap, same
         * stride-slicing scheme as weights_from_gguf_deepseek2 (no shared
         * experts for this arch). */
        size_t expert_elems = (size_t)exp_hid * (size_t)dim;

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_exps.weight", l);
        const GGUFTensor *tg_exps = gguf_find_tensor(hdr, name_buf);
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up_exps.weight", l);
        const GGUFTensor *tu_exps = gguf_find_tensor(hdr, name_buf);
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down_exps.weight", l);
        const GGUFTensor *td_exps = gguf_find_tensor(hdr, name_buf);

        if (!tg_exps || !tu_exps || !td_exps) {
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing expert tensors for layer %d\n", l);
            return TN_ERR_INVALID_WEIGHTS;
        }

        size_t g_stride = gguf_ld_quant_bytes(tg_exps->type, expert_elems);
        size_t u_stride = gguf_ld_quant_bytes(tu_exps->type, expert_elems);
        size_t d_stride = gguf_ld_quant_bytes(td_exps->type, expert_elems);

        if (g_stride == 0 || u_stride == 0 || d_stride == 0) {
            TN_LOG_ERR("gguf_loader",
                       "[gguf_loader] unknown quant type for expert tensors layer %d\n", l);
            return TN_ERR_INVALID_WEIGHTS;
        }

        for (int e = 0; e < num_experts; e++) {
            w->moe.w1[l][e] = (tn_i8 *)tg_exps->data + (size_t)e * g_stride;
            w->moe.w3[l][e] = (tn_i8 *)tu_exps->data + (size_t)e * u_stride;
            w->moe.w2[l][e] = (tn_i8 *)td_exps->data + (size_t)e * d_stride;
            w->moe.s1[l][e] = w->moe.s2[l][e] = w->moe.s3[l][e] = 0.0f;
        }
        w->expert_quant.w2_per_layer[l]  = (int)td_exps->type;
        w->expert_quant.w13_per_layer[l] = (int)tg_exps->type;
    }

    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* Set expert quant flags from layer 0 (first_k_dense_replace=0 means
     * every layer is MoE for this arch, unlike deepseek2's leading-dense
     * layers). */
    {
        const GGUFTensor *tg     = gguf_find_tensor(hdr, "blk.0.ffn_gate_exps.weight");
        const GGUFTensor *td     = gguf_find_tensor(hdr, "blk.0.ffn_down_exps.weight");
        w->expert_quant.w13_type = tg ? (int)tg->type : GGUF_TYPE_Q4_K;
        w->expert_quant.w2_type  = td ? (int)td->type : GGUF_TYPE_Q4_K;
        w->expert_quant.present  = true;
    }

    w->layers_are_ternary = false;
    w->cls.is_ternary     = false;
    w->cls.scale          = 1.0f;
    /* wq_type/wk_type/wv_type/wo_type were set per-layer by Q3MOE_LOAD_PROJ
     * above; w1_type/w2_type/w3_type stay at their zero-init default
     * (WEIGHT_TYPE_F32) since this arch has no dense FFN layers to read
     * them — every layer routes through moe_ffn_forward instead. */

    TN_OUT_INF("gguf_q3moe",
               "[GGUF-Q3MOE] Weights loaded (%d layers, %d experts/layer, top-%d, "
               "head_dim=%d)\n",
               nl, num_experts, mc->num_experts_per_tok, head_dim);
    return TN_OK;
}

#undef Q3MOE_LOAD_PROJ
