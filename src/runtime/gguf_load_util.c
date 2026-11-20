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


/* Allocate a float32 buffer (n_elems floats), fill from tensor data, add to store.
 * Returns NULL and prints a message for unsupported types. */
float *gguf_ld_tensor_to_f32(const GGUFTensor *t, size_t n_elems, GGUFWeightStore *store) {
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) return NULL;

    switch (t->type) {
        case GGUF_TYPE_F32:
            memcpy(buf, t->data, n_elems * sizeof(float));
            break;
        case GGUF_TYPE_F16: {
            const uint16_t *s = (const uint16_t *)t->data;
            for (size_t i = 0; i < n_elems; i++) buf[i] = tn_fp16_to_f32(s[i]);
            break;
        }
        case GGUF_TYPE_BF16:
            gguf_dequant_bf16(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q8_0:
            gguf_dequant_q8_0(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q4_K:
            gguf_dequant_q4_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q4_0:
            gguf_dequant_q4_0(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q5_0:
            gguf_dequant_q5_0(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q5_1:
            gguf_dequant_q5_1(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q5_K:
            gguf_dequant_q5_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q6_K:
            gguf_dequant_q6_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q2_K:
            gguf_dequant_q2_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q3_K:
            gguf_dequant_q3_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ4_NL:
            gguf_dequant_iq4_nl(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q2_0:
            gguf_dequant_q2_0(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q4_1:
            gguf_dequant_q4_1(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q8_1:
            gguf_dequant_q8_1(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_Q8_K:
            gguf_dequant_q8_k(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ4_XS:
            gguf_dequant_iq4_xs(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ2_XXS:
            gguf_dequant_iq2_xxs(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ2_XS:
            gguf_dequant_iq2_xs(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ2_S:
            gguf_dequant_iq2_s(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ3_XXS:
            gguf_dequant_iq3_xxs(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ3_S:
            gguf_dequant_iq3_s(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ1_S:
            gguf_dequant_iq1_s(buf, t->data, n_elems);
            break;
        case GGUF_TYPE_IQ1_M:
            gguf_dequant_iq1_m(buf, t->data, n_elems);
            break;
        default:
            TN_LOG_ERR(
                "gguf_loader",
                "[gguf_loader] unsupported quant type %d ('%s') for tensor '%s'\n"
                "[gguf_loader] Supported: F32, F16, BF16, Q8_0, Q4_K, Q4_0, Q5_0, Q5_1, Q5_K, "
                "Q6_K, Q2_K, Q3_K, IQ4_NL\n"
                "[gguf_loader] Convert with: llama-quantize model.gguf out.gguf Q4_K_S\n",
                (int)t->type, gguf_type_name(t->type), t->name);
            free(buf);
            return NULL;
    }

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
    switch (type) {
        case GGUF_TYPE_F32:
            return n_elems * 4;
        case GGUF_TYPE_F16:
            return n_elems * 2;
        case GGUF_TYPE_BF16:
            return n_elems * 2;
        case GGUF_TYPE_Q8_0:
            return (n_elems / 32) * 34;
        case GGUF_TYPE_Q4_0:
            return (n_elems / 32) * 18;
        case GGUF_TYPE_Q5_0:
            return (n_elems / 32) * 22;
        case GGUF_TYPE_Q5_1:
            return (n_elems / 32) * 24;
        case GGUF_TYPE_Q4_K:
            return (n_elems / 256) * 144;
        case GGUF_TYPE_Q5_K:
            return (n_elems / 256) * 176;
        case GGUF_TYPE_Q6_K:
            return (n_elems / 256) * 210;
        case GGUF_TYPE_Q2_K:
            return (n_elems / 256) * 84;
        case GGUF_TYPE_Q3_K:
            return (n_elems / 256) * 110;
        case GGUF_TYPE_IQ4_NL:
            return (n_elems / 32) * 18;
        case GGUF_TYPE_Q4_1:
            return (n_elems / 32) * 20;
        case GGUF_TYPE_Q8_1:
            return (n_elems / 32) * 36;
        case GGUF_TYPE_Q8_K:
            return (n_elems / 256) * 292;
        case GGUF_TYPE_IQ4_XS:
            return (n_elems / 256) * 136;
        case GGUF_TYPE_IQ2_XXS:
            return (n_elems / 256) * 66;
        case GGUF_TYPE_IQ2_XS:
            return (n_elems / 256) * 74;
        case GGUF_TYPE_IQ2_S:
            return (n_elems / 256) * 82;
        case GGUF_TYPE_IQ3_XXS:
            return (n_elems / 256) * 98;
        case GGUF_TYPE_IQ3_S:
            return (n_elems / 256) * 110;
        case GGUF_TYPE_IQ1_S:
            return (n_elems / 256) * 50;
        case GGUF_TYPE_IQ1_M:
            return (n_elems / 256) * 56;
        default:
            return 0;
    }
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
