/* Qwen3.5 / 3.6 hybrid (Gated-DeltaNet + gated attention) weight loader. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/fp16.h"
#include "core/gguf_i2s.h"
#include "core/gguf_quant.h"
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

/* ── weights_from_gguf_qwen35() — Qwen3.5/3.6 hybrid attention loader ───── */

/* Zero-copy loader for a Q2_0 projection weight: the whole model is Q2_0
 * (per README: "true 1.71 bits/weight ... no high-precision escape hatches"),
 * so unlike LOAD_PROJ above this does not fall back to F32 dequant — a 27B
 * model fully dequantized to F32 would need ~109 GB, impossible on a normal
 * host. Errors loudly if a tensor turns out not to be Q2_0. */
#define Q35_LOAD_Q2_0(field, tname)                                                                \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);                      \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        if (_t->type != GGUF_TYPE_Q2_0) {                                                          \
            fprintf(stderr, "[gguf_loader] qwen35: '%s' must be Q2_0 (got %d '%s')\n", name_buf,   \
                    (int)_t->type, gguf_type_name(_t->type));                                      \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        (field)[l] = (tn_i8 *)_t->data; /* zero-copy raw Q2_0 bytes */                             \
    } while (0)

#define Q35_LOAD_NORM(field, tname, n_elems)                                                       \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        (field)[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)(n_elems), store);                 \
        if (!(field)[l]) return TN_ERR_INVALID_WEIGHTS;                                            \
    } while (0)

TernaryError weights_from_gguf_qwen35(TransformerWeights *w, const Config *cfg,
                                      const GGUFHeader *hdr, const MoEConfig *mc,
                                      GGUFWeightStore *store) {

    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int nl         = cfg->n_layers;
    int n_head     = cfg->n_heads;
    int head_dim   = mc->attn_head_dim; /* NOT dim/n_heads for this arch */
    int key_dim    = mc->ssm_group_count * mc->ssm_state_size;
    int value_dim  = mc->ssm_inner_size;
    int conv_dim   = key_dim * 2 + value_dim;
    int n_v_heads  = mc->ssm_time_step_rank;
    char name_buf[128];

    for (int l = 0; l < nl; l++) {
        Q35_LOAD_NORM(w->rms_att_weight, "attn_norm.weight", dim);
        Q35_LOAD_NORM(w->rms_ffn_weight, "post_attention_norm.weight", dim);

        /* Dense SwiGLU FFN — shared by both layer types, same as generic path */
        Q35_LOAD_Q2_0(w->w1, "ffn_gate.weight");
        Q35_LOAD_Q2_0(w->w3, "ffn_up.weight");
        Q35_LOAD_Q2_0(w->w2, "ffn_down.weight");
        w->s1[l] = w->s2[l] = w->s3[l] = 1.0f;

        if (q35_layer_is_full_attn(mc, l)) {
            /* Full attention: GQA + QK-norm + partial rotary + gated output.
             * wq holds the combined [Q | gate] projection (width 2*n_head*head_dim). */
            Q35_LOAD_Q2_0(w->wq, "attn_q.weight");
            Q35_LOAD_Q2_0(w->wk, "attn_k.weight");
            Q35_LOAD_Q2_0(w->wv, "attn_v.weight");
            Q35_LOAD_Q2_0(w->wo, "attn_output.weight");
            Q35_LOAD_NORM(w->q35_attn_q_norm, "attn_q_norm.weight", head_dim);
            Q35_LOAD_NORM(w->q35_attn_k_norm, "attn_k_norm.weight", head_dim);
            w->sq[l] = w->sk[l] = w->sv[l] = w->so[l] = 1.0f;
        } else {
            /* Linear attention (Gated DeltaNet) */
            Q35_LOAD_Q2_0(w->q35_ssm_qkv, "attn_qkv.weight");
            Q35_LOAD_Q2_0(w->q35_ssm_gate, "attn_gate.weight");
            Q35_LOAD_NORM(w->q35_ssm_conv1d, "ssm_conv1d.weight",
                          (size_t)mc->ssm_conv_kernel * conv_dim);
            Q35_LOAD_NORM(w->q35_ssm_dt_bias, "ssm_dt.bias", n_v_heads);
            Q35_LOAD_NORM(w->q35_ssm_a, "ssm_a", n_v_heads);
            Q35_LOAD_Q2_0(w->q35_ssm_alpha, "ssm_alpha.weight");
            Q35_LOAD_Q2_0(w->q35_ssm_beta, "ssm_beta.weight");
            Q35_LOAD_NORM(w->q35_ssm_norm, "ssm_norm.weight", mc->ssm_state_size);
            Q35_LOAD_Q2_0(w->q35_ssm_out, "ssm_out.weight");
        }
    }

    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* Token embedding: always kept as zero-copy raw Q2_0 — embedding lookup
     * only ever touches one row per token, so there's no bandwidth reason to
     * materialize the whole table regardless of classifier format. */
    {
        const GGUFTensor *temb = gguf_find_tensor(hdr, "token_embd.weight");
        if (!temb || temb->type != GGUF_TYPE_Q2_0) {
            fprintf(stderr, "[gguf_loader] qwen35: 'token_embd.weight' must be Q2_0\n");
            return TN_ERR_INVALID_WEIGHTS;
        }
        w->q35_token_embd_raw = temb->data;

        const GGUFTensor *tout = gguf_find_tensor(hdr, "output.weight");
        if (!tout || tout->type != GGUF_TYPE_Q2_0) {
            fprintf(stderr, "[gguf_loader] qwen35: 'output.weight' must be Q2_0\n");
            return TN_ERR_INVALID_WEIGHTS;
        }
        w->q35_output_raw    = tout->data;
        w->q35_is_q2_0_model = true;

        /* LM head: kept as zero-copy raw Q2_0 by default (materializing a
         * dense copy of the 248320x5120 vocab would cost ~2.5 GB for BF16,
         * ~1.3 GB for INT8, ~0.6 GB for INT4). Only pay that cost when the
         * user explicitly asked for a specific --classifier format —
         * forward.c's dispatch then uses the materialized copy instead of
         * always falling back to raw Q2_0 regardless of the flag (the bug
         * this replaces — see docs/ai/mistakes.md). */
        const TnHardwareProfile *hw_cls = tn_hardware_profile_get();
        if (hw_cls && hw_cls->classifier_explicit) {
            size_t vocab     = (size_t)cfg->vocab_size;
            size_t d         = (size_t)dim;
            float *tmp_f32   = (float *)malloc(vocab * d * sizeof(float));
            tn_u16 *bf16_buf = (tn_u16 *)malloc(vocab * d * sizeof(tn_u16));
            if (tmp_f32 && bf16_buf) {
                gguf_dequant_q2_0(tmp_f32, tout->data, vocab * d);
                for (size_t i = 0; i < vocab * d; i++) {
                    bf16_buf[i] = gguf_ld_f32_to_bf16_round(tmp_f32[i]);
                }
                if (gguf_ld_store_add(store, bf16_buf) == 0) {
                    w->wcls = bf16_buf;
                    weights_build_classifier_quant(w, cfg);
                } else {
                    free(bf16_buf);
                    bf16_buf = NULL;
                }
            } else {
                free(bf16_buf);
            }
            free(tmp_f32);
        }
    }

    w->layers_are_ternary = false;
    w->wcls_is_ternary    = false;
    w->wcls_scale         = 1.0f;
    /* Uniform Q2_0 for every layer/projection — unlike Q4_K_M, this
     * architecture's quantizer applies one type to all tensors, all layers.
     * (attention routes to qwen35_attention_forward and never reads these;
     * ffn_forward's dense SwiGLU path does, for the shared-FFN portion.) */
    for (int l = 0; l < nl; l++) {
        w->wq_type[l] = w->wk_type[l] = w->wv_type[l] = w->wo_type[l] = WEIGHT_TYPE_Q2_0;
        w->w1_type[l] = w->w2_type[l] = w->w3_type[l] = WEIGHT_TYPE_Q2_0;
    }
    (void)hidden_dim;
    (void)n_head;

    printf("[GGUF-Q35] Weights loaded (%d layers, full_attention_interval=%d, "
           "%d Q2_0 quantized)\n",
           nl, mc->full_attention_interval, nl);
    return TN_OK;
}

#undef Q35_LOAD_Q2_0
#undef Q35_LOAD_NORM
