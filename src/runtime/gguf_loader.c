#include "runtime/gguf_loader.h"

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
#include "runtime/hardware_profile.h"
#include "runtime/mamba_state.h"


/* Shape of a dense (llama-like) model. A dense Qwen3 can have an independent head width
 * (key_length != dim / n_heads): Q/K/V/O widths and the per-head norms then follow it, and the
 * forward pass routes to the QK-norm attention (MoEConfig.has_qk_norm). */
typedef struct {
    int dim, hidden_dim, kv_dim, head_dim, n_layers, q_rows;
    int wide_heads;
} DenseShape;

static void dense_shape(DenseShape *sh, const Config *cfg, const GGUFHeader *hdr) {
    sh->dim        = cfg->dim;
    sh->hidden_dim = cfg->hidden_dim;
    sh->kv_dim     = config_kv_dim(cfg);
    sh->head_dim   = config_head_dim(cfg);
    sh->n_layers   = cfg->n_layers;
    sh->q_rows     = sh->dim;
    sh->wide_heads = 0;
    MoEConfig gmc;
    memset(&gmc, 0, sizeof(gmc));
    if (moe_config_from_gguf(&gmc, hdr) == TN_OK && gmc.has_qk_norm && !gmc.is_moe &&
        gmc.attn_head_dim > 0) {
        sh->wide_heads = 1;
        sh->head_dim   = gmc.attn_head_dim;
        sh->q_rows     = cfg->n_heads * sh->head_dim;
        sh->kv_dim     = cfg->n_kv_heads * sh->head_dim;
    }
}

/* Token embedding table (vocab_size × dim).
 *
 * BF16/F16/F32 on-disk → stored as BF16 (tn_u16) for classifier (wcls),
 * embd_f32 stays NULL → embed_token uses BF16 bit-shift path.
 *
 * Quantized on-disk (Q4_K etc.) → dequant to F32 kept in embd_f32 so that
 * embed_token does a direct memcpy — exactly llama.cpp's ggml_get_rows path
 * (Q4_K → F32, no BF16 intermediate, no precision loss from truncation).
 * A separate BF16 copy is still built and kept in token_embedding_table for
 * the classifier (wcls weight-tied path). */
static TernaryError load_dense_embedding(TransformerWeights *w, const Config *cfg,
                                         const GGUFHeader *hdr, GGUFWeightStore *store) {
    const int dim       = cfg->dim;
    const GGUFTensor *t = gguf_find_tensor(hdr, "token_embd.weight");
    if (!t) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing 'token_embd.weight'\n");
        return TN_ERR_INVALID_WEIGHTS;
    }
    size_t n = (size_t)cfg->vocab_size * (size_t)dim;
    if (t->type == GGUF_TYPE_BF16) {
        /* Zero-copy: BF16 is already uint16 layout, identical to tn_u16 */
        w->token_embedding_table = (tn_u16 *)t->data;
    } else if (t->type == GGUF_TYPE_F32) {
        /* Convert F32 → BF16: keep upper 16 bits of IEEE 754 representation */
        tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
        if (!buf) return TN_ERR_OOM;
        const float *src = (const float *)t->data;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits;
            memcpy(&bits, &src[i], 4);
            buf[i] = (tn_u16)(bits >> 16);
        }
        if (gguf_ld_store_add(store, buf) != 0) {
            free(buf);
            return TN_ERR_OOM;
        }
        w->token_embedding_table = buf;
        mapped_file_release(t->data, t->size_bytes);
    } else if (t->type == GGUF_TYPE_F16) {
        /* Dequant F16 → F32 → BF16 */
        tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
        if (!buf) return TN_ERR_OOM;
        const uint16_t *src = (const uint16_t *)t->data;
        for (size_t i = 0; i < n; i++) {
            float f = tn_fp16_to_f32(src[i]);
            uint32_t bits;
            memcpy(&bits, &f, 4);
            buf[i] = (tn_u16)(bits >> 16);
        }
        if (gguf_ld_store_add(store, buf) != 0) {
            free(buf);
            return TN_ERR_OOM;
        }
        w->token_embedding_table = buf;
        mapped_file_release(t->data, t->size_bytes);
    } else if (t->type == GGUF_TYPE_Q4_K || t->type == GGUF_TYPE_Q8_0 ||
               t->type == GGUF_TYPE_Q4_0 || t->type == GGUF_TYPE_Q5_0 ||
               t->type == GGUF_TYPE_Q5_1 || t->type == GGUF_TYPE_Q5_K ||
               t->type == GGUF_TYPE_Q6_K) {
        /* Dequantise → F32 kept as embd_f32 (exact llama.cpp path).
         * Also build BF16 copy for the classifier (wcls weight-tied). */
        float *ftmp = (float *)malloc(n * sizeof(float));
        if (!ftmp) return TN_ERR_OOM;
        switch (t->type) {
            case GGUF_TYPE_Q4_K:
                gguf_dequant_q4_k(ftmp, t->data, n);
                break;
            case GGUF_TYPE_Q8_0:
                gguf_dequant_q8_0(ftmp, t->data, n);
                break;
            case GGUF_TYPE_Q5_0:
                gguf_dequant_q5_0(ftmp, t->data, n);
                break;
            case GGUF_TYPE_Q5_1:
                gguf_dequant_q5_1(ftmp, t->data, n);
                break;
            case GGUF_TYPE_Q5_K:
                gguf_dequant_q5_k(ftmp, t->data, n);
                break;
            case GGUF_TYPE_Q6_K:
                gguf_dequant_q6_k(ftmp, t->data, n);
                break;
            default:
                gguf_dequant_q4_0(ftmp, t->data, n);
                break;
        }
        /* Keep F32 buffer for embedding lookup — registered in store for cleanup */
        if (gguf_ld_store_add(store, ftmp) != 0) {
            free(ftmp);
            return TN_ERR_OOM;
        }
        w->embd_f32 = ftmp;

        /* BF16 copy for classifier (wcls weight-tied) */
        tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
        if (!buf) return TN_ERR_OOM;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits;
            memcpy(&bits, &ftmp[i], 4);
            buf[i] = (tn_u16)(bits >> 16);
        }
        if (gguf_ld_store_add(store, buf) != 0) {
            free(buf);
            return TN_ERR_OOM;
        }
        w->token_embedding_table = buf;
        mapped_file_release(t->data, t->size_bytes);
    } else {
        TN_LOG_ERR("gguf_loader",
                   "[gguf_loader] unsupported embedding type %d ('%s')\n"
                   "[gguf_loader] Supported embedding types: F32, F16, BF16\n",
                   (int)t->type, gguf_type_name(t->type));
        return TN_ERR_INVALID_WEIGHTS;
    }
    return TN_OK;
}

/* One layer of a dense model: norms, optional QK-norm, the seven projections, optional biases and
 * BitNet sub-norms. */
static TernaryError load_dense_layer(TransformerWeights *w, const GGUFHeader *hdr,
                                     GGUFWeightStore *store, const DenseShape *sh, int l) {
    char name_buf[128];
    const int dim        = sh->dim;
    const int hidden_dim = sh->hidden_dim;
    const int kv_dim     = sh->kv_dim;
    const int head_dim   = sh->head_dim;
    const int q_rows     = sh->q_rows;
    const int wide_heads = sh->wide_heads;
    /* Attention norm (rms_att_weight): F32 zero-copy or dequant */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
    w->rms_att_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
    if (!w->rms_att_weight[l]) return TN_ERR_INVALID_WEIGHTS;

    /* FFN norm (rms_ffn_weight) */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
    w->rms_ffn_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
    if (!w->rms_ffn_weight[l]) return TN_ERR_INVALID_WEIGHTS;

    /* Qwen3 (dense) per-head QK-norm — optional: present for "qwen3" arch,
     * absent for llama/mistral/gemma/phi. Reuses the q35_attn_{q,k}_norm
     * fields the Qwen3.5/3.6 hybrid loader already populates; attention.c's
     * generic forward path applies them the same way when non-NULL. */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q_norm.weight", l);
    if (gguf_find_tensor(hdr, name_buf)) {
        float **qn = wide_heads ? w->qwen3moe_attn_q_norm : w->q35_attn_q_norm;
        float **kn = wide_heads ? w->qwen3moe_attn_k_norm : w->q35_attn_k_norm;
        qn[l]      = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)head_dim, store);
        if (!qn[l]) return TN_ERR_INVALID_WEIGHTS;
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k_norm.weight", l);
        kn[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)head_dim, store);
        if (!kn[l]) return TN_ERR_INVALID_WEIGHTS;
    }

    TernaryError e = gguf_ld_dense_projection(hdr, store, l, "attn_q.weight", sh->q_rows, sh->dim,
                                              w->wq, w->wq_type, w->sq);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "attn_k.weight", sh->kv_dim, sh->dim, w->wk,
                                     w->wk_type, w->sk);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "attn_v.weight", sh->kv_dim, sh->dim, w->wv,
                                     w->wv_type, w->sv);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "attn_output.weight", sh->dim, sh->q_rows,
                                     w->wo, w->wo_type, w->so);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "ffn_gate.weight", sh->hidden_dim, sh->dim,
                                     w->w1, w->w1_type, w->s1);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "ffn_down.weight", sh->dim, sh->hidden_dim,
                                     w->w2, w->w2_type, w->s2);
    if (e == TN_OK)
        e = gguf_ld_dense_projection(hdr, store, l, "ffn_up.weight", sh->hidden_dim, sh->dim, w->w3,
                                     w->w3_type, w->s3);
    if (e != TN_OK) return e;

    /* Q/K/V biases (Qwen2/2.5) — optional; 2026-10-19: previously never
     * loaded, so such models produced garbage instead of being refused. */
    {
        static const char *const bias_names[3] = {"attn_q.bias", "attn_k.bias", "attn_v.bias"};
        float **dst[3]                         = {w->bq, w->bk, w->bv};
        const int lens[3]                      = {q_rows, kv_dim, kv_dim};
        for (int b = 0; b < 3; b++) {
            snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", l, bias_names[b]);
            if (!gguf_find_tensor(hdr, name_buf)) continue;
            dst[b][l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)lens[b], store);
            if (!dst[b][l]) return TN_ERR_INVALID_WEIGHTS;
        }
    }

    /* BitNet sub-norms (before attn_output / ffn_down) — optional. */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_sub_norm.weight", l);
    if (gguf_find_tensor(hdr, name_buf)) {
        w->rms_attn_sub_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->rms_attn_sub_norm[l]) return TN_ERR_INVALID_WEIGHTS;
    }
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_sub_norm.weight", l);
    if (gguf_find_tensor(hdr, name_buf)) {
        w->rms_ffn_sub_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)hidden_dim, store);
        if (!w->rms_ffn_sub_norm[l]) return TN_ERR_INVALID_WEIGHTS;
    }
    return TN_OK;
}

/* Final RMS norm and the output classifier: output.weight when present, else weight-tied. */
static TernaryError load_dense_output(TransformerWeights *w, const Config *cfg,
                                      const GGUFHeader *hdr, GGUFWeightStore *store) {
    const int dim = cfg->dim;
    /* 3. Final RMS norm */
    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* 4. Output classifier — use output.weight if present; else weight-tie. */
    {
        const GGUFTensor *t = gguf_find_tensor(hdr, "output.weight");
        if (t) {
            size_t n = (size_t)cfg->vocab_size * (size_t)dim;
            if (gguf_ld_classifier_keep_raw(w, cfg, t)) {
                w->wcls = NULL; /* forward.c dispatches on wcls_raw */
            } else if (t->type == GGUF_TYPE_BF16) {
                w->wcls = (tn_u16 *)t->data; /* zero-copy */
            } else if (t->type == GGUF_TYPE_F32) {
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                const float *src = (const float *)t->data;
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits;
                    memcpy(&bits, &src[i], 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (gguf_ld_store_add(store, buf) != 0) {
                    free(buf);
                    return TN_ERR_OOM;
                }
                w->wcls = buf;
                mapped_file_release(t->data, t->size_bytes);
            } else if (t->type == GGUF_TYPE_F16) {
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                const uint16_t *src = (const uint16_t *)t->data;
                for (size_t i = 0; i < n; i++) {
                    float f = tn_fp16_to_f32(src[i]);
                    uint32_t bits;
                    memcpy(&bits, &f, 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (gguf_ld_store_add(store, buf) != 0) {
                    free(buf);
                    return TN_ERR_OOM;
                }
                w->wcls = buf;
                mapped_file_release(t->data, t->size_bytes);
            } else {
                /* Quantized types → F32 → BF16 */
                float *ftmp = gguf_ld_tensor_to_f32(t, n, store);
                if (!ftmp) {
                    w->wcls = w->token_embedding_table;
                    goto wcls_done;
                }
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits;
                    memcpy(&bits, &ftmp[i], 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (gguf_ld_store_add(store, buf) != 0) {
                    free(buf);
                    return TN_ERR_OOM;
                }
                w->wcls = buf;
                mapped_file_release(t->data, t->size_bytes);
            }
        wcls_done:;
        } else {
            w->wcls = w->token_embedding_table; /* weight-tied */
        }
    }
    return TN_OK;
}

/* All-I2_S projections (BitNet GGUF) run on the ternary path; a mix of ternary and float
 * projections has no kernel path, so it is refused rather than misdispatched. */
static TernaryError set_inference_flags(TransformerWeights *w, int nl) {
    {
        int n_tern = 0, n_proj = 0;
        for (int l = 0; l < nl; l++) {
            const int t7[7] = {w->wq_type[l], w->wk_type[l], w->wv_type[l], w->wo_type[l],
                               w->w1_type[l], w->w2_type[l], w->w3_type[l]};
            for (int k = 0; k < 7; k++) {
                n_proj++;
                n_tern += (t7[k] == WEIGHT_TYPE_TERNARY);
            }
        }
        if (n_tern && n_tern != n_proj) {
            TN_LOG_ERR("gguf_loader",
                       "[gguf_loader] mixed I2_S and non-ternary projections are not supported\n");
            return TN_ERR_UNSUPPORTED;
        }
        w->layers_are_ternary = (n_tern == n_proj && n_proj > 0);
    }
    w->wcls_is_ternary = false;
    w->wcls_scale      = 1.0f;
    /* wq_type/wk_type/.../w3_type were set per-projection by LOAD_PROJ above;
 * sq/sk/.../s3 hold the I2_S tensor scales on the ternary path (0 otherwise);
 * rms_*_sub_norm are set only when the file has them (BitNet). */
    return TN_OK;
}

/* Dense (llama / mistral / qwen2 / qwen3 / BitNet) weights. */
static TernaryError weights_from_gguf_dense(TransformerWeights *w, const Config *cfg,
                                            const GGUFHeader *hdr, GGUFWeightStore *store) {
    DenseShape sh;
    dense_shape(&sh, cfg, hdr);
    TernaryError e = load_dense_embedding(w, cfg, hdr, store);
    for (int l = 0; e == TN_OK && l < sh.n_layers; l++) e = load_dense_layer(w, hdr, store, &sh, l);
    if (e == TN_OK) e = load_dense_output(w, cfg, hdr, store);
    if (e == TN_OK) e = set_inference_flags(w, sh.n_layers);
    if (e != TN_OK) return e;

    w->wcls_is_ternary = false;
    w->wcls_scale      = 1.0f;
    /* wq_type/wk_type/.../w3_type were set per-projection by gguf_ld_dense_projection(); sq/sk/.../s3
     * hold the I2_S tensor scales on the ternary path (0 otherwise); rms_*_sub_norm are set only
     * when the file has them (BitNet). */

    /* INT8 / INT4 quantized classifier variants */
    weights_build_classifier_quant(w, cfg);

    TN_OUT_INF("gguf", "[GGUF] Weights loaded (%d layers, arch=%s)\n", sh.n_layers, hdr->arch);
    return TN_OK;
}

/* The MoE-family loaders (deepseek2, qwen3moe, mamba, qwen35) share this shape: read the MoE
 * config, optionally load the dense BF16 embedding / output head, run the architecture loader,
 * and optionally build the quantised classifier (qwen35 keeps its zero-copy Q2_0 head; mamba
 * builds its own). */
typedef TernaryError (*ArchLoader)(TransformerWeights *, const Config *, const GGUFHeader *,
                                   const MoEConfig *, GGUFWeightStore *);

static TernaryError load_moe_family(TransformerWeights *w, const Config *cfg, const GGUFHeader *hdr,
                                    GGUFWeightStore *store, ArchLoader loader, int bf16_head) {
    MoEConfig mc;
    TernaryError e = moe_config_from_gguf(&mc, hdr);
    if (e != TN_OK) return e;
    if (bf16_head) {
        e = gguf_ld_bf16_embedding_and_output(w, cfg, hdr, store);
        if (e != TN_OK) return e;
    }
    e = loader(w, cfg, hdr, &mc, store);
    if (e != TN_OK) return e;
    if (bf16_head) weights_build_classifier_quant(w, cfg);
    return TN_OK;
}

/* One row per architecture with its own loader; adding a family is a row here plus its loader
 * file. Anything not listed takes the dense path. */
static const struct {
    const char *arch;
    ArchLoader loader;
    int bf16_head; /* dense BF16 embedding / head + quantised classifier built here */
} k_arch_loaders[] = {
    {"deepseek2", weights_from_gguf_deepseek2, 1},
    {"qwen3moe", weights_from_gguf_qwen3moe, 1},
    {"mamba", weights_from_gguf_mamba, 0},
    /* Qwen3.5/3.6 hybrid Gated-DeltaNet + Gated-Attention: no classifier-quant build here, the LM
     * head stays zero-copy Q2_0 (see weights_from_gguf_qwen35 / forward.c's q35_is_q2_0_model). */
    {"qwen35", weights_from_gguf_qwen35, 0},
};

/* The reader leaves data NULL for a tensor that does not fit in the file (a truncated download);
 * no loader below can use one, so the whole file is refused up front with the tensor's name. */
static TernaryError check_tensors_present(const GGUFHeader *hdr) {
    for (uint64_t i = 0; i < hdr->n_tensors; i++) {
        if (!hdr->tensors[i].data) {
            TN_LOG_ERR("gguf_loader",
                       "[gguf_loader] tensor '%s' is missing from the file (truncated download?)\n",
                       hdr->tensors[i].name);
            return TN_ERR_INVALID_WEIGHTS;
        }
    }
    return TN_OK;
}

TernaryError weights_from_gguf(TransformerWeights *w, const Config *cfg, const GGUFHeader *hdr,
                               GGUFWeightStore **store_out) {
    TernaryError present = check_tensors_present(hdr);
    if (present != TN_OK) return present;
    GGUFWeightStore *store = gguf_ld_store_alloc();
    if (!store) return TN_ERR_OOM;
    *store_out = store;

    for (size_t i = 0; i < sizeof(k_arch_loaders) / sizeof(k_arch_loaders[0]); i++)
        if (strcmp(hdr->arch, k_arch_loaders[i].arch) == 0)
            return load_moe_family(w, cfg, hdr, store, k_arch_loaders[i].loader,
                                   k_arch_loaders[i].bf16_head);
    return weights_from_gguf_dense(w, cfg, hdr, store);
}
