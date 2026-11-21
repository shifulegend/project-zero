#include "core/weights.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/unpack.h"
#include "runtime/hardware_profile.h"
#include "runtime/mamba_state.h"

/* 1 when none of the n pointers is NULL. */
static int all_allocated(const void *const *ptrs, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!ptrs[i]) return 0;
    return 1;
}

/* The dense attention and FFN pointer arrays (one entry per layer, NULL contents). Returns 1 when
 * every allocation succeeded. */
static int alloc_dense_arrays(TransformerWeights *w, int n_layers) {
    w->attn.wq       = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->attn.wk       = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->attn.wv       = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->attn.wo       = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->attn.sq       = (float *)calloc(n_layers, sizeof(float));
    w->attn.sk       = (float *)calloc(n_layers, sizeof(float));
    w->attn.sv       = (float *)calloc(n_layers, sizeof(float));
    w->attn.so       = (float *)calloc(n_layers, sizeof(float));
    w->ffn.w1        = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->ffn.w2        = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->ffn.w3        = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->ffn.s1        = (float *)calloc(n_layers, sizeof(float));
    w->ffn.s2        = (float *)calloc(n_layers, sizeof(float));
    w->ffn.s3        = (float *)calloc(n_layers, sizeof(float));
    w->attn.norm     = (float **)calloc(n_layers, sizeof(float *));
    w->ffn.norm      = (float **)calloc(n_layers, sizeof(float *));
    w->attn.sub_norm = (float **)calloc(n_layers, sizeof(float *));
    w->ffn.sub_norm  = (float **)calloc(n_layers, sizeof(float *));

    /* Per-layer, per-projection weight type (WEIGHT_TYPE_*) — see weights.h
     * for why this can't be a single shared flag. */
    w->attn.wq_type            = (int *)calloc(n_layers, sizeof(int));
    w->attn.wk_type            = (int *)calloc(n_layers, sizeof(int));
    w->attn.wv_type            = (int *)calloc(n_layers, sizeof(int));
    w->attn.wo_type            = (int *)calloc(n_layers, sizeof(int));
    w->ffn.w1_type             = (int *)calloc(n_layers, sizeof(int));
    w->ffn.w2_type             = (int *)calloc(n_layers, sizeof(int));
    w->ffn.w3_type             = (int *)calloc(n_layers, sizeof(int));
    const void *const arrays[] = {
        w->attn.wq,      w->attn.wk,       w->attn.wv,      w->attn.wo,      w->attn.sq,
        w->attn.sk,      w->attn.sv,       w->attn.so,      w->ffn.w1,       w->ffn.w2,
        w->ffn.w3,       w->ffn.s1,        w->ffn.s2,       w->ffn.s3,       w->attn.norm,
        w->ffn.norm,     w->attn.sub_norm, w->ffn.sub_norm, w->attn.wq_type, w->attn.wk_type,
        w->attn.wv_type, w->attn.wo_type,  w->ffn.w1_type,  w->ffn.w2_type,  w->ffn.w3_type};
    return all_allocated(arrays, sizeof(arrays) / sizeof(arrays[0]));
}

/* The Qwen3.5 hybrid-attention, Qwen3-MoE QK-norm and bias pointer arrays. Returns 1 when every
 * allocation succeeded. */
static int alloc_arch_arrays(TransformerWeights *w, int n_layers) {
    /* Qwen3.5/3.6 hybrid attention — cheap per-layer pointer arrays,
     * always allocated (NULL contents) like attn.sub_norm above;
     * populated only when MoEConfig.has_linear_attn is set. */
    w->q35.attn_q_norm = (float **)calloc(n_layers, sizeof(float *));
    w->q35.attn_k_norm = (float **)calloc(n_layers, sizeof(float *));
    w->q35.ssm_qkv     = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->q35.ssm_gate    = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->q35.ssm_conv1d  = (float **)calloc(n_layers, sizeof(float *));
    w->q35.ssm_dt_bias = (float **)calloc(n_layers, sizeof(float *));
    w->q35.ssm_a       = (float **)calloc(n_layers, sizeof(float *));
    w->q35.ssm_alpha   = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->q35.ssm_beta    = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));
    w->q35.ssm_norm    = (float **)calloc(n_layers, sizeof(float *));
    w->q35.ssm_out     = (tn_i8 **)calloc(n_layers, sizeof(tn_i8 *));

    /* Qwen3-MoE QK-norm — cheap per-layer pointer arrays, always allocated
     * (NULL contents) like q35_attn_q_norm above; populated only when
     * MoEConfig.has_qk_norm is set. */
    w->qwen3moe.attn_q_norm    = (float **)calloc(n_layers, sizeof(float *));
    w->qwen3moe.attn_k_norm    = (float **)calloc(n_layers, sizeof(float *));
    w->attn.bq                 = (float **)calloc(n_layers, sizeof(float *));
    w->attn.bk                 = (float **)calloc(n_layers, sizeof(float *));
    w->attn.bv                 = (float **)calloc(n_layers, sizeof(float *));
    const void *const arrays[] = {
        w->q35.attn_q_norm,      w->q35.attn_k_norm, w->q35.ssm_qkv, w->q35.ssm_gate,
        w->q35.ssm_conv1d,       w->q35.ssm_dt_bias, w->q35.ssm_a,   w->q35.ssm_alpha,
        w->q35.ssm_beta,         w->q35.ssm_norm,    w->q35.ssm_out, w->qwen3moe.attn_q_norm,
        w->qwen3moe.attn_k_norm, w->attn.bq,         w->attn.bk,     w->attn.bv};
    return all_allocated(arrays, sizeof(arrays) / sizeof(arrays[0]));
}

TernaryError weights_alloc_pointers(TransformerWeights *w, const Config *cfg) {
    int n_layers = cfg->n_layers;
    /* Callers are responsible for zeroing the struct (memset(w, 0, sizeof(*w)))
     * before calling this function.  Doing it here would clobber fields the
     * caller may have set (e.g. layers_are_ternary) — violating modularity. */

    if (!alloc_dense_arrays(w, n_layers) || !alloc_arch_arrays(w, n_layers)) {
        weights_free_pointers(w);
        return TN_ERR_OOM;
    }
    return TN_OK;
}

/* Every heap array a TransformerWeights owns directly (zero-copy views into the model file are not
 * here). Freed in this order; free(NULL) is a no-op. */
static const size_t k_owned_arrays[] = {
    offsetof(TransformerWeights, attn.wq),
    offsetof(TransformerWeights, attn.wk),
    offsetof(TransformerWeights, attn.wv),
    offsetof(TransformerWeights, attn.wo),
    offsetof(TransformerWeights, attn.sq),
    offsetof(TransformerWeights, attn.sk),
    offsetof(TransformerWeights, attn.sv),
    offsetof(TransformerWeights, attn.so),
    offsetof(TransformerWeights, ffn.w1),
    offsetof(TransformerWeights, ffn.w2),
    offsetof(TransformerWeights, ffn.w3),
    offsetof(TransformerWeights, ffn.s1),
    offsetof(TransformerWeights, ffn.s2),
    offsetof(TransformerWeights, ffn.s3),
    offsetof(TransformerWeights, attn.norm),
    offsetof(TransformerWeights, ffn.norm),
    offsetof(TransformerWeights, attn.sub_norm),
    offsetof(TransformerWeights, ffn.sub_norm),
    offsetof(TransformerWeights, attn.wq_type),
    offsetof(TransformerWeights, attn.wk_type),
    offsetof(TransformerWeights, attn.wv_type),
    offsetof(TransformerWeights, attn.wo_type),
    offsetof(TransformerWeights, ffn.w1_type),
    offsetof(TransformerWeights, ffn.w2_type),
    offsetof(TransformerWeights, ffn.w3_type),
    offsetof(TransformerWeights, q35.attn_q_norm),
    offsetof(TransformerWeights, q35.attn_k_norm),
    offsetof(TransformerWeights, q35.ssm_qkv),
    offsetof(TransformerWeights, q35.ssm_gate),
    offsetof(TransformerWeights, q35.ssm_conv1d),
    offsetof(TransformerWeights, q35.ssm_dt_bias),
    offsetof(TransformerWeights, q35.ssm_a),
    offsetof(TransformerWeights, q35.ssm_alpha),
    offsetof(TransformerWeights, q35.ssm_beta),
    offsetof(TransformerWeights, q35.ssm_norm),
    offsetof(TransformerWeights, q35.ssm_out),
    offsetof(TransformerWeights, qwen3moe.attn_q_norm),
    offsetof(TransformerWeights, qwen3moe.attn_k_norm),
    offsetof(TransformerWeights, attn.bq),
    offsetof(TransformerWeights, attn.bk),
    offsetof(TransformerWeights, attn.bv),
    offsetof(TransformerWeights, cls.i8),
    offsetof(TransformerWeights, cls.i8_scales),
    offsetof(TransformerWeights, cls.i4),
    offsetof(TransformerWeights, cls.i4_scales),
    offsetof(TransformerWeights, expert_quant.w2_per_layer),
    offsetof(TransformerWeights, expert_quant.w13_per_layer),
    offsetof(TransformerWeights, shared.w1_type_per_layer),
    offsetof(TransformerWeights, shared.w2_type_per_layer),
    offsetof(TransformerWeights, shared.w3_type_per_layer),
};

void weights_free_pointers(TransformerWeights *w) {
    if (w->mamba) {
        mamba_weights_free(w->mamba);
        w->mamba = NULL;
    }
    for (size_t k = 0; k < sizeof(k_owned_arrays) / sizeof(k_owned_arrays[0]); k++) {
        void *p;
        memcpy(&p, (const char *)w + k_owned_arrays[k], sizeof p);
        free(p);
    }
    /* Phase 17: MoE pointer arrays must be freed explicitly via moe_weights_free()
     * by the owning component before this call, as that requires the MoEConfig.
     * We just zero the entire struct at the end anyway. */
    memset(w, 0, sizeof(*w));
}

TernaryError weights_map(TransformerWeights *w, const Config *cfg, const MoEConfig *mc, tn_i8 *data,
                         size_t data_size) {
    tn_i8 *ptr     = data;
    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int kv_dim     = config_kv_dim(cfg);
    int n_layers   = cfg->n_layers;

    /* scale_mode=0: ternary + sub_norms (BitNet)
     * scale_mode=2: ternary + MoE, NO sub_norms (DeepSeek) */
    bool has_sub_norms = (cfg->scale_mode == 0);

    /* Absolute alignment base is 64 */
    const size_t offset_base = 64;

#define MAP_PTR(dest, bytes)                                                                       \
    do {                                                                                           \
        if ((size_t)(ptr - data + (bytes)) > data_size) return TN_ERR_INVALID_WEIGHTS;             \
        (dest) = (void *)ptr;                                                                      \
        ptr += (bytes);                                                                            \
    } while (0)

#define ALIGN64()                                                                                  \
    do {                                                                                           \
        size_t abs_offset = (size_t)(ptr - data) + offset_base;                                    \
        size_t padding    = (64 - (abs_offset % 64)) % 64;                                         \
        if (padding > 0) {                                                                         \
            if ((size_t)(ptr - data + padding) > data_size) return TN_ERR_INVALID_WEIGHTS;         \
            ptr += padding;                                                                        \
        }                                                                                          \
    } while (0)

#define COPY_VAL_ALIGN(dest, size)                                                                 \
    do {                                                                                           \
        if ((size_t)(ptr - data + (size)) > data_size) return TN_ERR_INVALID_WEIGHTS;              \
        memcpy((dest), ptr, (size));                                                               \
        ptr += (size);                                                                             \
        ALIGN64();                                                                                 \
    } while (0)

    /* 1. Token embedding table — bfloat16, 2 bytes per value.
     * Saves ~650 MB vs float32 with zero precision loss (native HF format). */
    MAP_PTR(w->token_embedding_table, (size_t)cfg->vocab_size * dim * sizeof(tn_u16));
    ALIGN64();

    /* scale_mode 0 = ternary dense (BitNet), scale_mode 2 = ternary MoE (DeepSeek) */
    w->layers_are_ternary = (cfg->scale_mode == 0 || cfg->scale_mode == 2);

    /* 2. Per-layer weights (DEFINITELY CONFIRMED Microsoft BitNet-b1.58-2B-4T Sequence) */
    for (int l = 0; l < n_layers; l++) {
        /* Input Norms (Both present) */
        MAP_PTR(w->attn.norm[l], (size_t)dim * sizeof(float));
        ALIGN64();
        MAP_PTR(w->ffn.norm[l], (size_t)dim * sizeof(float));
        ALIGN64();

        size_t size_q   = (size_t)dim * dim;
        size_t size_kv  = (size_t)dim * kv_dim;
        size_t size_ffn = (size_t)dim * hidden_dim;

        if (w->layers_are_ternary) {
            /* Attention: Q, K, V */
            if (mc && mc->mla.enabled) {
                /* MLA path (Phase 17.6): replaces standard wq/wk/wv.
                 * Sizes (DeepSeek-V2-Lite example):
                 *   mla_wq    : n_heads*(nope+rope) × dim  → 3072 × 2048
                 *   mla_wkv_a : (kv_lora+rope) × dim       → 576  × 2048
                 *   mla_wkv_b : n_kv_heads*(nope+v) × kv_lora → 4096 × 512
                 * wq[l] and wk[l]/wv[l] are left NULL for MLA layers. */
                const MlaDims *md = &mc->mla;
                size_t wq_r = (size_t)cfg->n_heads * (md->qk_nope_head_dim + md->qk_rope_head_dim);
                size_t wkva_r = (size_t)(md->kv_lora_rank + md->qk_rope_head_dim);
                size_t wkvb_r = (size_t)cfg->n_kv_heads * (md->qk_nope_head_dim + md->v_head_dim);

                MAP_PTR(w->mla.wq[l], packed_bytes(wq_r * dim));
                COPY_VAL_ALIGN(&w->mla.sq[l], 4);
                MAP_PTR(w->mla.wkv_a[l], packed_bytes(wkva_r * dim));
                COPY_VAL_ALIGN(&w->mla.skv_a[l], 4);
                MAP_PTR(w->mla.wkv_b[l], packed_bytes(wkvb_r * (size_t)md->kv_lora_rank));
                COPY_VAL_ALIGN(&w->mla.skv_b[l], 4);
            } else {
                MAP_PTR(w->attn.wq[l], packed_bytes(size_q));
                COPY_VAL_ALIGN(&w->attn.sq[l], 4);
                MAP_PTR(w->attn.wk[l], packed_bytes(size_kv));
                COPY_VAL_ALIGN(&w->attn.sk[l], 4);
                MAP_PTR(w->attn.wv[l], packed_bytes(size_kv));
                COPY_VAL_ALIGN(&w->attn.sv[l], 4);
            }

            /* attn_sub_norm (BitNet only) */
            if (has_sub_norms) {
                MAP_PTR(w->attn.sub_norm[l], (size_t)dim * sizeof(float));
                ALIGN64();
            }

            /* Output projection */
            MAP_PTR(w->attn.wo[l], packed_bytes(size_q));
            COPY_VAL_ALIGN(&w->attn.so[l], 4);

            /* Dense FFN: skip for MoE layers */
            if (!moe_layer_is_moe(mc, l)) {
                /* FFN: Gate, Up */
                MAP_PTR(w->ffn.w1[l], packed_bytes(size_ffn));
                COPY_VAL_ALIGN(&w->ffn.s1[l], 4);
                MAP_PTR(w->ffn.w3[l], packed_bytes(size_ffn));
                COPY_VAL_ALIGN(&w->ffn.s3[l], 4);

                /* ffn_sub_norm (BitNet only) */
                if (has_sub_norms) {
                    MAP_PTR(w->ffn.sub_norm[l], (size_t)hidden_dim * sizeof(float));
                    ALIGN64();
                }

                /* Down projection */
                MAP_PTR(w->ffn.w2[l], packed_bytes(size_ffn));
                COPY_VAL_ALIGN(&w->ffn.s2[l], 4);
            }
        } else {
            return TN_ERR_INVALID_WEIGHTS;
        }
    }

    /* 3. Final norm */
    MAP_PTR(w->rms_final_weight, (size_t)dim * sizeof(float));
    ALIGN64();

    /* 4. Output classifier — weight-tied to bf16 embedding table. */
    w->cls.w          = w->token_embedding_table;
    w->cls.scale      = 1.0f;
    w->cls.is_ternary = false;

    /* 5 & 6. Build INT8 / INT4 quantized classifier variants. */
    weights_build_classifier_quant(w, cfg);

    /* 7. MoE expert weights (immediately follows the regular weight section) */
    if (mc && mc->is_moe) return moe_weights_map(w, cfg, mc, data, data_size, &ptr);

#undef MAP_PTR
#undef ALIGN64
#undef COPY_VAL_ALIGN
    return TN_OK;
}

/* Quantize block[0..count) to signed [-max_level, max_level] using scale
 * `inv_scale` (= max_level / scale), and return the sum of squared
 * reconstruction error. Used by find_best_block_scale's candidate search
 * and (implicitly) by the final quantize pass. */
static float block_quant_sqerr(const float *block, int count, float inv_scale, float scale,
                               int max_level) {
    float err = 0.0f;
    for (int k = 0; k < count; k++) {
        int q = (int)(block[k] * inv_scale + (block[k] >= 0 ? 0.5f : -0.5f));
        if (q > max_level) q = max_level;
        if (q < -max_level) q = -max_level;
        float recon = (float)q * scale;
        float d     = block[k] - recon;
        err += d * d;
    }
    return err;
}

/* find_best_block_scale — real per-block scale search minimizing squared
 * reconstruction error, added 2026-07-31 in response to GitHub issue #27:
 * the previous classifier INT8/INT4 quantizer used one scale for an entire
 * row (dim elements, e.g. 4096) and picked it via pure max-abs/level — "a
 * mere casting" that "completely ignores the whole issue of perplexity."
 *
 * The max-abs scale (scale0 = max_abs/max_level) is a reasonable starting
 * point but not error-optimal: it's set entirely by whichever single value
 * in the block is largest, even though minimizing total squared error is a
 * function of the whole block's distribution. This does a small local
 * search around scale0 (9 candidates, ±10%) and keeps whichever minimizes
 * squared error for THIS block — the same principle k-quant formats like
 * Q4_K use (see gguf_quant.c's decode side), just for a much smaller,
 * cheaper local search rather than a full joint scale+min optimization
 * (which would need a genuinely different, asymmetric packing format —
 * flagged as a natural follow-up, out of scope for this pass). */
static float find_best_block_scale(const float *block, int count, int max_level) {
    float max_abs = 0.0f;
    for (int k = 0; k < count; k++) {
        float a = block[k] < 0.0f ? -block[k] : block[k];
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 0.0f;

    float scale0     = max_abs / (float)max_level;
    float best_scale = scale0;
    float best_err = block_quant_sqerr(block, count, (float)max_level / max_abs, scale0, max_level);

    for (int step = -4; step <= 4; step++) {
        if (step == 0) continue;
        float mult       = 1.0f + 0.025f * (float)step; /* 0.90 .. 1.10 in 9 steps */
        float cand_scale = scale0 * mult;
        float cand_inv   = (cand_scale > 0.0f) ? 1.0f / cand_scale : 0.0f;
        float err        = block_quant_sqerr(block, count, cand_inv, cand_scale, max_level);
        if (err < best_err) {
            best_err   = err;
            best_scale = cand_scale;
        }
    }
    return best_scale;
}

/* Quantises one BF16 row block by block (TN_CLS_QUANT_BLOCK elements per block, scale chosen to
 * minimise squared error, see find_best_block_scale): the per-block scales go to row_scales, and
 * the values, rounded to nearest and clamped to +-qmax, to q_row[0..d). */
static void quantize_cls_row(const tn_u16 *row, size_t d, int n_blocks, int qmax, float *row_scales,
                             int8_t *q_row) {
    float block_f32[TN_CLS_QUANT_BLOCK];
    for (int b = 0; b < n_blocks; b++) {
        size_t blk_start = (size_t)b * TN_CLS_QUANT_BLOCK;
        int blk_len      = (int)(d - blk_start);
        if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;

        for (int k = 0; k < blk_len; k++) {
            tn_u32 bits = (tn_u32)row[blk_start + (size_t)k] << 16;
            memcpy(&block_f32[k], &bits, sizeof(float));
        }

        float scale     = find_best_block_scale(block_f32, blk_len, qmax);
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        row_scales[b]   = scale;

        for (int k = 0; k < blk_len; k++) {
            float fval = block_f32[k];
            int q      = (int)(fval * inv_scale + (fval >= 0 ? 0.5f : -0.5f));
            if (q > qmax) q = qmax;
            if (q < -qmax) q = -qmax;
            q_row[blk_start + (size_t)k] = (int8_t)q;
        }
    }
}

/* INT8 classifier for faster LM head inference: block-wise quantization, halving bandwidth
 * (BF16 -> INT8). Stored as unsigned (q + 128). On an allocation failure nothing is kept. */
static void build_cls_i8(TransformerWeights *w, const Config *cfg, int n_blocks) {
    size_t vocab = (size_t)cfg->vocab_size, d = (size_t)cfg->dim;
    w->cls.i8        = (tn_u8 *)malloc(vocab * d);
    w->cls.i8_scales = (float *)malloc(vocab * (size_t)n_blocks * sizeof(float));
    int8_t *q_row    = (int8_t *)malloc(d);
    if (!w->cls.i8 || !w->cls.i8_scales || !q_row) {
        free(w->cls.i8);
        free(w->cls.i8_scales);
        w->cls.i8        = NULL;
        w->cls.i8_scales = NULL;
        free(q_row);
        return;
    }
    for (size_t i = 0; i < vocab; i++) {
        quantize_cls_row(w->cls.w + i * d, d, n_blocks, 127,
                         w->cls.i8_scales + i * (size_t)n_blocks, q_row);
        tn_u8 *dst = w->cls.i8 + i * d;
        for (size_t k = 0; k < d; k++) dst[k] = (tn_u8)(q_row[k] + 128);
    }
    free(q_row);
}

/* INT4 classifier for minimum-bandwidth LM head: the same block-wise, error-minimizing scale
 * search as INT8. Packing: byte[k] = (w[2k+1] << 4) | w[2k], unsigned (+8 bias), at the row's
 * absolute element index (blocks are a multiple of 2 elements, TN_CLS_QUANT_BLOCK=32, so no block
 * ever splits a byte). On an allocation failure nothing is kept. */
static void build_cls_i4(TransformerWeights *w, const Config *cfg, int n_blocks) {
    size_t vocab = (size_t)cfg->vocab_size, d = (size_t)cfg->dim;
    size_t row_bytes = (d + 1) / 2;
    w->cls.i4        = (tn_u8 *)malloc(vocab * row_bytes);
    w->cls.i4_scales = (float *)malloc(vocab * (size_t)n_blocks * sizeof(float));
    int8_t *q_row    = (int8_t *)malloc(d);
    if (!w->cls.i4 || !w->cls.i4_scales || !q_row) {
        free(w->cls.i4);
        free(w->cls.i4_scales);
        w->cls.i4        = NULL;
        w->cls.i4_scales = NULL;
        free(q_row);
        return;
    }
    for (size_t i = 0; i < vocab; i++) {
        quantize_cls_row(w->cls.w + i * d, d, n_blocks, 7, w->cls.i4_scales + i * (size_t)n_blocks,
                         q_row);
        tn_u8 *dst = w->cls.i4 + i * row_bytes;
        for (size_t k = 0; k + 1 < d; k += 2)
            dst[k / 2] = (tn_u8)(((q_row[k + 1] + 8) << 4) | (q_row[k] + 8));
        if (d & 1) dst[(d - 1) / 2] = (tn_u8)(q_row[d - 1] + 8);
    }
    free(q_row);
}

void weights_build_classifier_quant(TransformerWeights *w, const Config *cfg) {
    int n_blocks = (cfg->dim + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;

    w->cls.i8        = NULL;
    w->cls.i8_scales = NULL;
    w->cls.i4        = NULL;
    w->cls.i4_scales = NULL;
    /* Raw quantized LM head (wcls_raw) in use: there is no BF16 table to derive INT8/INT4 copies
     * from, and none is needed (2026-10-21). */
    if (!w->cls.w || w->cls.is_ternary) return;

    /* Hardware-aware: only build what the hardware profile recommends (INT8 or INT4). */
    const TnHardwareProfile *hw = tn_hardware_profile_get();
    if (!hw || hw->classifier_fmt >= TN_CLS_INT8) build_cls_i8(w, cfg, n_blocks);
    if (!hw || hw->classifier_fmt >= TN_CLS_INT4) build_cls_i4(w, cfg, n_blocks);
}
