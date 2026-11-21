/* Shared helpers of the GGUF weight loaders: the buffer store, tensor-to-float conversion, the
 * typed projection policy and the dense BF16 embedding / output head. */

#include "runtime/gguf_load_util.h"

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
#include "runtime/gguf_loader.h"
#include "runtime/hardware_profile.h"
#include "runtime/mamba_state.h"

/* ── GGUFWeightStore — growable list of heap buffers to free on cleanup ──── */

struct GGUFWeightStore {
    void **bufs;
    size_t count;
    size_t cap;
};

GGUFWeightStore *gguf_ld_store_alloc(void) {
    GGUFWeightStore *s = (GGUFWeightStore *)malloc(sizeof(GGUFWeightStore));
    if (!s) return NULL;
    s->cap   = 64;
    s->count = 0;
    s->bufs  = (void **)malloc(s->cap * sizeof(void *));
    if (!s->bufs) {
        free(s);
        return NULL;
    }
    return s;
}

int gguf_ld_store_add(GGUFWeightStore *s, void *ptr) {
    if (s->count == s->cap) {
        size_t new_cap = s->cap * 2;
        void **nb      = (void **)realloc(s->bufs, new_cap * sizeof(void *));
        if (!nb) return -1;
        s->bufs = nb;
        s->cap  = new_cap;
    }
    s->bufs[s->count++] = ptr;
    return 0;
}

void weights_free_gguf(GGUFWeightStore *s) {
    if (!s) return;
    for (size_t i = 0; i < s->count; i++) free(s->bufs[i]);
    free(s->bufs);
    free(s);
}

/* ── Dequantization helpers ───────────────────────────────────────────────── */

typedef void (*DequantFn)(float *out, const void *data, size_t n_elems);

static void dequant_f32(float *out, const void *data, size_t n_elems) {
    memcpy(out, data, n_elems * sizeof(float));
}

static void dequant_f16(float *out, const void *data, size_t n_elems) {
    const uint16_t *s = (const uint16_t *)data;
    for (size_t i = 0; i < n_elems; i++) out[i] = tn_fp16_to_f32(s[i]);
}

/* One row per tensor type this loader can turn into float32. A new quantisation format adds a
 * row here and its block size to the type table in core/gguf_reader.c; nothing else changes. */
static const struct {
    GGUFType type;
    DequantFn fn;
} k_dequant[] = {
    {GGUF_TYPE_F32, dequant_f32},
    {GGUF_TYPE_F16, dequant_f16},
    {GGUF_TYPE_BF16, gguf_dequant_bf16},
    {GGUF_TYPE_Q8_0, gguf_dequant_q8_0},
    {GGUF_TYPE_Q4_K, gguf_dequant_q4_k},
    {GGUF_TYPE_Q4_0, gguf_dequant_q4_0},
    {GGUF_TYPE_Q5_0, gguf_dequant_q5_0},
    {GGUF_TYPE_Q5_1, gguf_dequant_q5_1},
    {GGUF_TYPE_Q5_K, gguf_dequant_q5_k},
    {GGUF_TYPE_Q6_K, gguf_dequant_q6_k},
    {GGUF_TYPE_Q2_K, gguf_dequant_q2_k},
    {GGUF_TYPE_Q3_K, gguf_dequant_q3_k},
    {GGUF_TYPE_IQ4_NL, gguf_dequant_iq4_nl},
    {GGUF_TYPE_Q2_0, gguf_dequant_q2_0},
    {GGUF_TYPE_Q4_1, gguf_dequant_q4_1},
    {GGUF_TYPE_Q8_1, gguf_dequant_q8_1},
    {GGUF_TYPE_Q8_K, gguf_dequant_q8_k},
    {GGUF_TYPE_IQ4_XS, gguf_dequant_iq4_xs},
    {GGUF_TYPE_IQ2_XXS, gguf_dequant_iq2_xxs},
    {GGUF_TYPE_IQ2_XS, gguf_dequant_iq2_xs},
    {GGUF_TYPE_IQ2_S, gguf_dequant_iq2_s},
    {GGUF_TYPE_IQ3_XXS, gguf_dequant_iq3_xxs},
    {GGUF_TYPE_IQ3_S, gguf_dequant_iq3_s},
    {GGUF_TYPE_IQ1_S, gguf_dequant_iq1_s},
    {GGUF_TYPE_IQ1_M, gguf_dequant_iq1_m},
};

static DequantFn dequant_for(GGUFType type) {
    for (size_t i = 0; i < sizeof(k_dequant) / sizeof(k_dequant[0]); i++)
        if (k_dequant[i].type == type) return k_dequant[i].fn;
    return NULL;
}

int gguf_ld_dequant(GGUFType type, float *out, const void *data, size_t n_elems) {
    DequantFn dequant = dequant_for(type);
    if (!dequant) return -1;
    dequant(out, data, n_elems);
    return 0;
}

/* Allocate a float32 buffer (n_elems floats), fill from tensor data, add to store.
 * Returns NULL and prints a message for unsupported types. */
float *gguf_ld_tensor_to_f32(const GGUFTensor *t, size_t n_elems, GGUFWeightStore *store) {
    DequantFn dequant = dequant_for(t->type);
    if (!dequant) {
        TN_LOG_ERR("gguf_loader",
                   "[gguf_loader] unsupported quant type %d ('%s') for tensor '%s'\n"
                   "[gguf_loader] Supported: F32, F16, BF16, Q8_0, Q4_K, Q4_0, Q5_0, Q5_1, Q5_K, "
                   "Q6_K, Q2_K, Q3_K, IQ4_NL\n"
                   "[gguf_loader] Convert with: llama-quantize model.gguf out.gguf Q4_K_S\n",
                   (int)t->type, gguf_type_name(t->type), t->name);
        return NULL;
    }
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) return NULL;
    gguf_ld_dequant(t->type, buf, t->data, n_elems);
    if (gguf_ld_store_add(store, buf) != 0) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* ── Norm-weight helper: F32 → zero-copy; F16/BF16 → dequant into store ─── */

float *gguf_ld_norm_to_f32(const GGUFHeader *hdr, const char *name, size_t n_elems,
                           GGUFWeightStore *store) {
    const GGUFTensor *t = gguf_find_tensor(hdr, name);
    if (!t) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing norm tensor '%s'\n", name);
        return NULL;
    }
    if (t->type == GGUF_TYPE_F32) {
        return (float *)t->data; /* zero-copy from mmap */
    }
    return gguf_ld_tensor_to_f32(t, n_elems, store);
}

/* ── Byte-count helper for quantised blocks ──────────────────────────────── */

size_t gguf_ld_quant_bytes(GGUFType type, size_t n_elems) {
    if (!dequant_for(type)) return 0;
    return (n_elems / gguf_type_block_elems(type)) * gguf_type_block_bytes(type);
}

/* Router and stacked routed-expert tensors of one MoE layer, shared by the DeepSeek-2 and
 * Qwen3-MoE loaders. The router is F32 and zero-copy; the three stacked expert tensors stay
 * quantised in the mapping and each expert's pointer is sliced from them by stride; the layer's
 * gate/up and down quant types are recorded for dispatch. */
TernaryError gguf_ld_moe_layer(const GGUFHeader *hdr, TransformerWeights *w, int l, int num_experts,
                               size_t expert_elems) {
    char name_buf[128];
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
    w->moe.gate_w[l] = (tn_i8 *)tgate->data; /* zero-copy, F32 */
    w->moe.gate_s[l] = 1.0f;

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
        TN_LOG_ERR("gguf_loader", "[gguf_loader] unknown quant type for expert tensors layer %d\n",
                   l);
        return TN_ERR_INVALID_WEIGHTS;
    }
    for (int e = 0; e < num_experts; e++) {
        w->moe.w1[l][e] = (tn_i8 *)tg_exps->data + (size_t)e * g_stride;
        w->moe.w3[l][e] = (tn_i8 *)tu_exps->data + (size_t)e * u_stride;
        w->moe.w2[l][e] = (tn_i8 *)td_exps->data + (size_t)e * d_stride;
        /* the scales are unused: the per-layer quant types below drive the dequant path */
        w->moe.s1[l][e] = w->moe.s2[l][e] = w->moe.s3[l][e] = 0.0f;
    }
    w->expert_quant.w2_per_layer[l]  = (int)td_exps->type;
    w->expert_quant.w13_per_layer[l] = (int)tg_exps->type;
    return TN_OK;
}

/* ── Raw quantized LM head (2026-10-21) ──────────────────────────────────
 * Keep output.weight zero-copy in its own Q4_K/Q6_K format when a fused
 * kernel exists and the user did not explicitly ask for a --classifier
 * format (which needs the BF16 table to derive INT8/INT4 copies). Returns 1
 * when w->wcls_raw was set. Previously every quantized head was widened to
 * BF16: 2.4x the bytes streamed per token for a Q6_K head. */
int gguf_ld_classifier_keep_raw(TransformerWeights *w, const Config *cfg, const GGUFTensor *t) {
    const TnHardwareProfile *hw = tn_hardware_profile_get();
    if (!t || (hw && hw->classifier_explicit)) return 0;
    if (cfg->dim % 256 != 0) return 0;
    if (t->type != GGUF_TYPE_Q6_K && t->type != GGUF_TYPE_Q4_K) return 0;
    w->cls.raw      = t->data;
    w->cls.raw_type = (int)t->type;
    return 1;
}

/* ── Shared dense-BF16 embedding + output.weight loader ──────────────────
 * Common to deepseek2 and qwen3moe (both dequant-then-BF16, unlike qwen35's
 * zero-copy-raw-Q2_0 embedding path). Extracted out of weights_from_gguf()'s
 * former inline deepseek2-only block so qwen3moe can reuse it verbatim
 * rather than duplicating ~35 lines. */
TernaryError gguf_ld_bf16_embedding_and_output(TransformerWeights *w, const Config *cfg,
                                               const GGUFHeader *hdr, GGUFWeightStore *store) {
    const GGUFTensor *temb = gguf_find_tensor(hdr, "token_embd.weight");
    if (!temb) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing 'token_embd.weight'\n");
        return TN_ERR_INVALID_WEIGHTS;
    }
    size_t n_emb = (size_t)cfg->vocab_size * (size_t)cfg->dim;
    float *ftmp  = gguf_ld_tensor_to_f32(temb, n_emb, store);
    if (!ftmp) return TN_ERR_INVALID_WEIGHTS;
    tn_u16 *ebuf = (tn_u16 *)malloc(n_emb * sizeof(tn_u16));
    if (!ebuf) return TN_ERR_OOM;
    for (size_t i = 0; i < n_emb; i++) {
        uint32_t bits;
        memcpy(&bits, &ftmp[i], 4);
        ebuf[i] = (tn_u16)(bits >> 16);
    }
    if (gguf_ld_store_add(store, ebuf) != 0) {
        free(ebuf);
        return TN_ERR_OOM;
    }
    w->token_embedding_table = ebuf;

    const GGUFTensor *toutw = gguf_find_tensor(hdr, "output.weight");
    if (toutw && gguf_ld_classifier_keep_raw(w, cfg, toutw)) {
        w->cls.w = NULL; /* forward.c dispatches on wcls_raw */
    } else if (toutw) {
        float *fout = gguf_ld_tensor_to_f32(toutw, n_emb, store);
        if (!fout) return TN_ERR_INVALID_WEIGHTS;
        tn_u16 *obuf = (tn_u16 *)malloc(n_emb * sizeof(tn_u16));
        if (!obuf) return TN_ERR_OOM;
        for (size_t i = 0; i < n_emb; i++) {
            uint32_t bits;
            memcpy(&bits, &fout[i], 4);
            obuf[i] = (tn_u16)(bits >> 16);
        }
        if (gguf_ld_store_add(store, obuf) != 0) {
            free(obuf);
            return TN_ERR_OOM;
        }
        w->cls.w = obuf;
    } else {
        w->cls.w = w->token_embedding_table;
    }
    return TN_OK;
}

/* ── Typed zero-copy projection loader (shared) ──────────────────────────
 * F16/F32 zero-copy; Q4_K / Q6_K zero-copy when n_inner % 256 == 0;
 * Q5_0 / Q5_1 / Q8_0 zero-copy when n_inner % 32 == 0 (2026-10-21: run by
 * the Q8_0-activation integer kernels, math/q8_0_act.c); anything else is
 * dequantized to F32 once. `d_rows` x `n_inner` (output x input). Used by
 * the mamba loader and DeepSeek-V2's leading dense FFN layers (which were
 * F32-dequantized before: 269 MB/token for DeepSeek-V2-Lite's layer 0
 * instead of 42 MB). */
TernaryError gguf_ld_proj_typed(const GGUFHeader *hdr, GGUFWeightStore *store, const char *name,
                                int d_rows, int n_inner, tn_i8 **out, int *type_out) {
    const GGUFTensor *t = gguf_find_tensor(hdr, name);
    if (!t) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name);
        return TN_ERR_INVALID_WEIGHTS;
    }
    size_t n       = (size_t)d_rows * (size_t)n_inner;
    const int k_ok = n_inner % 256 == 0, l_ok = n_inner % 32 == 0;
    if (t->type == GGUF_TYPE_F16) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_F16;
    } else if (t->type == GGUF_TYPE_F32) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_F32;
    } else if (t->type == GGUF_TYPE_Q4_K && k_ok) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_Q4K;
    } else if (t->type == GGUF_TYPE_Q6_K && k_ok) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_Q6K;
    } else if (t->type == GGUF_TYPE_Q5_0 && l_ok) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_Q5_0;
    } else if (t->type == GGUF_TYPE_Q5_1 && l_ok) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_Q5_1;
    } else if (t->type == GGUF_TYPE_Q8_0 && l_ok) {
        *out      = (tn_i8 *)t->data;
        *type_out = WEIGHT_TYPE_Q8_0;
    } else {
        float *f = gguf_ld_tensor_to_f32(t, n, store);
        if (!f) return TN_ERR_INVALID_WEIGHTS;
        *out      = (tn_i8 *)f;
        *type_out = WEIGHT_TYPE_F32;
    }
    return TN_OK;
}
