#include "core/weights.h"
#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/unpack.h"
#include "core/hardware_profile.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

TernaryError weights_alloc_pointers(TransformerWeights *w, const Config *cfg) {
    int nl = cfg->n_layers;
    /* Callers are responsible for zeroing the struct (memset(w, 0, sizeof(*w)))
     * before calling this function.  Doing it here would clobber fields the
     * caller may have set (e.g. layers_are_ternary) — violating modularity. */

    w->wq = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->wk = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->wv = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->wo = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->sq = (float *)calloc(nl, sizeof(float));
    w->sk = (float *)calloc(nl, sizeof(float));
    w->sv = (float *)calloc(nl, sizeof(float));
    w->so = (float *)calloc(nl, sizeof(float));
    w->w1 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->w2 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->w3 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->s1 = (float *)calloc(nl, sizeof(float));
    w->s2 = (float *)calloc(nl, sizeof(float));
    w->s3 = (float *)calloc(nl, sizeof(float));
    w->rms_att_weight = (float **)calloc(nl, sizeof(float *));
    w->rms_ffn_weight = (float **)calloc(nl, sizeof(float *));
    w->rms_attn_sub_norm = (float **)calloc(nl, sizeof(float *));
    w->rms_ffn_sub_norm = (float **)calloc(nl, sizeof(float *));

    if (!w->wq || !w->wk || !w->wv || !w->wo ||
        !w->sq || !w->sk || !w->sv || !w->so ||
        !w->w1 || !w->w2 || !w->w3 ||
        !w->s1 || !w->s2 || !w->s3 ||
        !w->rms_att_weight || !w->rms_ffn_weight ||
        !w->rms_attn_sub_norm || !w->rms_ffn_sub_norm) {
        weights_free_pointers(w);
        return TN_ERR_OOM;
    }
    return TN_OK;
}

void weights_free_pointers(TransformerWeights *w) {
    if (w->wq) free(w->wq); if (w->wk) free(w->wk); if (w->wv) free(w->wv); if (w->wo) free(w->wo);
    if (w->sq) free(w->sq); if (w->sk) free(w->sk); if (w->sv) free(w->sv); if (w->so) free(w->so);
    if (w->w1) free(w->w1); if (w->w2) free(w->w2); if (w->w3) free(w->w3);
    if (w->s1) free(w->s1); if (w->s2) free(w->s2); if (w->s3) free(w->s3);
    if (w->rms_att_weight) free(w->rms_att_weight); if (w->rms_ffn_weight) free(w->rms_ffn_weight);
    if (w->rms_attn_sub_norm) free(w->rms_attn_sub_norm); if (w->rms_ffn_sub_norm) free(w->rms_ffn_sub_norm);
    if (w->wcls_i8) free(w->wcls_i8);
    if (w->wcls_i8_scales) free(w->wcls_i8_scales);
    if (w->wcls_i4) free(w->wcls_i4);
    if (w->wcls_i4_scales) free(w->wcls_i4_scales);
    if (w->expert_w2_quant_per_layer)  free(w->expert_w2_quant_per_layer);
    if (w->expert_w13_quant_per_layer) free(w->expert_w13_quant_per_layer);
    if (w->shared_w1_type_per_layer)  free(w->shared_w1_type_per_layer);
    if (w->shared_w2_type_per_layer)  free(w->shared_w2_type_per_layer);
    if (w->shared_w3_type_per_layer)  free(w->shared_w3_type_per_layer);
    /* Phase 17: MoE pointer arrays must be freed explicitly via moe_weights_free()
     * by the owning component before this call, as that requires the MoEConfig.
     * We just zero the entire struct at the end anyway. */
    memset(w, 0, sizeof(*w));
}

TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                         const MoEConfig *mc,
                         tn_i8 *data, size_t data_size) {
    tn_i8 *ptr = data;
    int dim = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int kv_dim = config_kv_dim(cfg);
    int nl = cfg->n_layers;

    /* scale_mode=0: ternary + sub_norms (BitNet)
     * scale_mode=2: ternary + MoE, NO sub_norms (DeepSeek) */
    bool has_sub_norms = (cfg->scale_mode == 0);

    /* Absolute alignment base is 64 */
    const size_t offset_base = 64;

    #define MAP_PTR(dest, bytes) do {                       \
        if ((size_t)(ptr - data + (bytes)) > data_size) return TN_ERR_INVALID_WEIGHTS; \
        (dest) = (void *)ptr;                               \
        ptr += (bytes);                                     \
    } while(0)

    #define ALIGN64() do {                                              \
        size_t abs_offset = (size_t)(ptr - data) + offset_base;         \
        size_t padding = (64 - (abs_offset % 64)) % 64;                 \
        if (padding > 0) {                                              \
            if ((size_t)(ptr - data + padding) > data_size) return TN_ERR_INVALID_WEIGHTS; \
            ptr += padding;                                             \
        }                                                               \
    } while(0)

    #define COPY_VAL_ALIGN(dest, size) do {                       \
        if ((size_t)(ptr - data + (size)) > data_size) return TN_ERR_INVALID_WEIGHTS; \
        memcpy((dest), ptr, (size));                        \
        ptr += (size);                                      \
        ALIGN64();                                          \
    } while(0)

    /* 1. Token embedding table — bfloat16, 2 bytes per value.
     * Saves ~650 MB vs float32 with zero precision loss (native HF format). */
    MAP_PTR(w->token_embedding_table, (size_t)cfg->vocab_size * dim * sizeof(tn_u16));
    ALIGN64();

    /* scale_mode 0 = ternary dense (BitNet), scale_mode 2 = ternary MoE (DeepSeek) */
    w->layers_are_ternary = (cfg->scale_mode == 0 || cfg->scale_mode == 2);

    /* 2. Per-layer weights (DEFINITELY CONFIRMED Microsoft BitNet-b1.58-2B-4T Sequence) */
    for (int l = 0; l < nl; l++) {
        /* Input Norms (Both present) */
        MAP_PTR(w->rms_att_weight[l], (size_t)dim * sizeof(float)); ALIGN64();
        MAP_PTR(w->rms_ffn_weight[l], (size_t)dim * sizeof(float)); ALIGN64();

        size_t size_q = (size_t)dim * dim;
        size_t size_kv = (size_t)dim * kv_dim;
        size_t size_ffn = (size_t)dim * hidden_dim;

        if (w->layers_are_ternary) {
            /* Attention: Q, K, V */
            if (mc && mc->has_mla) {
                /* MLA path (Phase 17.6): replaces standard wq/wk/wv.
                 * Sizes (DeepSeek-V2-Lite example):
                 *   mla_wq    : n_heads*(nope+rope) × dim  → 3072 × 2048
                 *   mla_wkv_a : (kv_lora+rope) × dim       → 576  × 2048
                 *   mla_wkv_b : n_kv_heads*(nope+v) × kv_lora → 4096 × 512
                 * wq[l] and wk[l]/wv[l] are left NULL for MLA layers. */
                size_t wq_r  = (size_t)cfg->n_heads *
                               (mc->qk_nope_head_dim + mc->qk_rope_head_dim);
                size_t wkva_r = (size_t)(mc->kv_lora_rank + mc->qk_rope_head_dim);
                size_t wkvb_r = (size_t)cfg->n_kv_heads *
                               (mc->qk_nope_head_dim + mc->v_head_dim);

                MAP_PTR(w->mla_wq[l],    packed_bytes(wq_r   * dim));
                COPY_VAL_ALIGN(&w->mla_sq[l],    4);
                MAP_PTR(w->mla_wkv_a[l], packed_bytes(wkva_r * dim));
                COPY_VAL_ALIGN(&w->mla_skv_a[l], 4);
                MAP_PTR(w->mla_wkv_b[l], packed_bytes(wkvb_r * (size_t)mc->kv_lora_rank));
                COPY_VAL_ALIGN(&w->mla_skv_b[l], 4);
            } else {
                MAP_PTR(w->wq[l], packed_bytes(size_q));  COPY_VAL_ALIGN(&w->sq[l], 4);
                MAP_PTR(w->wk[l], packed_bytes(size_kv)); COPY_VAL_ALIGN(&w->sk[l], 4);
                MAP_PTR(w->wv[l], packed_bytes(size_kv)); COPY_VAL_ALIGN(&w->sv[l], 4);
            }

            /* attn_sub_norm (BitNet only) */
            if (has_sub_norms) {
                MAP_PTR(w->rms_attn_sub_norm[l], (size_t)dim * sizeof(float)); ALIGN64();
            }

            /* Output projection */
            MAP_PTR(w->wo[l], packed_bytes(size_q));  COPY_VAL_ALIGN(&w->so[l], 4);

            /* Dense FFN: skip for MoE layers */
            if (!moe_layer_is_moe(mc, l)) {
                /* FFN: Gate, Up */
                MAP_PTR(w->w1[l], packed_bytes(size_ffn)); COPY_VAL_ALIGN(&w->s1[l], 4);
                MAP_PTR(w->w3[l], packed_bytes(size_ffn)); COPY_VAL_ALIGN(&w->s3[l], 4);

                /* ffn_sub_norm (BitNet only) */
                if (has_sub_norms) {
                    MAP_PTR(w->rms_ffn_sub_norm[l], (size_t)hidden_dim * sizeof(float)); ALIGN64();
                }

                /* Down projection */
                MAP_PTR(w->w2[l], packed_bytes(size_ffn)); COPY_VAL_ALIGN(&w->s2[l], 4);
            }
        } else {
            return TN_ERR_INVALID_WEIGHTS;
        }
    }

    /* 3. Final norm */
    MAP_PTR(w->rms_final_weight, (size_t)dim * sizeof(float)); ALIGN64();

    /* 4. Output classifier — weight-tied to bf16 embedding table. */
    w->wcls = w->token_embedding_table;
    w->wcls_scale = 1.0f;
    w->wcls_is_ternary = false;

    /* 5 & 6. Build INT8 / INT4 quantized classifier variants. */
    weights_build_classifier_quant(w, cfg);

    /* 7. MoE expert weights (immediately follows the regular weight section) */
    if (mc && mc->is_moe) {
        TernaryError moe_err = moe_weights_map(w, cfg, mc, data, data_size, &ptr);
        if (moe_err != TN_OK) return moe_err;
    }

    #undef MAP_PTR
    #undef ALIGN64
    #undef COPY_VAL_ALIGN
    return TN_OK;
}

void weights_build_classifier_quant(TransformerWeights *w, const Config *cfg) {
    int dim = cfg->dim;

    /* Build INT8 quantized classifier for faster LM head inference.
     * Per-row symmetric quantization: scale = max_abs / 127.
     * Halves bandwidth (BF16 → INT8). Stored as unsigned (q + 128) for VNNI.
     * Hardware-aware: only build if hardware profile recommends INT8 or INT4. */
    w->wcls_i8 = NULL;
    w->wcls_i8_scales = NULL;
    const TnHardwareProfile *hw = tn_hardware_profile_get();
    bool need_i8 = !hw || hw->classifier_fmt >= TN_CLS_INT8;
    if (!w->wcls_is_ternary && need_i8) {
        size_t vocab = (size_t)cfg->vocab_size;
        size_t d     = (size_t)dim;
        w->wcls_i8 = (tn_u8 *)malloc(vocab * d);
        w->wcls_i8_scales = (float *)malloc(vocab * sizeof(float));
        if (w->wcls_i8 && w->wcls_i8_scales) {
            for (size_t i = 0; i < vocab; i++) {
                const tn_u16 *row = w->wcls + i * d;
                tn_u8 *dst = w->wcls_i8 + i * d;
                /* Pass 1: find max absolute BF16 value in this row */
                float max_abs = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    tn_u32 bits = (tn_u32)row[j] << 16;
                    float fval;
                    memcpy(&fval, &bits, sizeof(float));
                    float a = fval < 0.0f ? -fval : fval;
                    if (a > max_abs) max_abs = a;
                }
                /* Pass 2: quantize to INT8, store as unsigned (+ 128 bias) */
                float scale = max_abs / 127.0f;
                float inv_scale = (max_abs > 0.0f) ? 127.0f / max_abs : 0.0f;
                w->wcls_i8_scales[i] = scale;
                for (size_t j = 0; j < d; j++) {
                    tn_u32 bits = (tn_u32)row[j] << 16;
                    float fval;
                    memcpy(&fval, &bits, sizeof(float));
                    int q = (int)(fval * inv_scale + (fval >= 0 ? 0.5f : -0.5f));
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    dst[j] = (tn_u8)(q + 128);
                }
            }
        }
    }

    /* Build INT4 quantized classifier for minimum-bandwidth LM head.
     * Per-row symmetric quantization: scale = max_abs / 7.
     * Packing: byte[k] = (w[2k+1] << 4) | w[2k], unsigned (+8 bias). */
    w->wcls_i4 = NULL;
    w->wcls_i4_scales = NULL;
    bool need_i4 = !hw || hw->classifier_fmt >= TN_CLS_INT4;
    if (!w->wcls_is_ternary && need_i4) {
        size_t vocab = (size_t)cfg->vocab_size;
        size_t d     = (size_t)dim;
        size_t row_bytes_i4 = (d + 1) / 2;
        w->wcls_i4 = (tn_u8 *)malloc(vocab * row_bytes_i4);
        w->wcls_i4_scales = (float *)malloc(vocab * sizeof(float));
        if (w->wcls_i4 && w->wcls_i4_scales) {
            for (size_t i = 0; i < vocab; i++) {
                const tn_u16 *row = w->wcls + i * d;
                tn_u8 *dst = w->wcls_i4 + i * row_bytes_i4;
                float max_abs_i4 = w->wcls_i8_scales[i] * 127.0f;
                float scale_i4 = max_abs_i4 / 7.0f;
                float inv_scale_i4 = (max_abs_i4 > 0.0f) ? 7.0f / max_abs_i4 : 0.0f;
                w->wcls_i4_scales[i] = scale_i4;
                size_t j = 0;
                for (; j + 1 < d; j += 2) {
                    tn_u32 bits0 = (tn_u32)row[j] << 16;
                    float f0; memcpy(&f0, &bits0, sizeof(float));
                    int q0 = (int)(f0 * inv_scale_i4 + (f0 >= 0 ? 0.5f : -0.5f));
                    if (q0 > 7) q0 = 7; if (q0 < -7) q0 = -7;

                    tn_u32 bits1 = (tn_u32)row[j+1] << 16;
                    float f1; memcpy(&f1, &bits1, sizeof(float));
                    int q1 = (int)(f1 * inv_scale_i4 + (f1 >= 0 ? 0.5f : -0.5f));
                    if (q1 > 7) q1 = 7; if (q1 < -7) q1 = -7;

                    dst[j/2] = (tn_u8)(((q1 + 8) << 4) | (q0 + 8));
                }
                if (j < d) {
                    tn_u32 bits0 = (tn_u32)row[j] << 16;
                    float f0; memcpy(&f0, &bits0, sizeof(float));
                    int q0 = (int)(f0 * inv_scale_i4 + (f0 >= 0 ? 0.5f : -0.5f));
                    if (q0 > 7) q0 = 7; if (q0 < -7) q0 = -7;
                    dst[j/2] = (tn_u8)(q0 + 8);
                }
            }
        }
    }
}
