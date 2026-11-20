/* DeepSeek-V2 (MLA + MoE) weight loader. */

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

/* ── weights_from_gguf_deepseek2() — DeepSeek-V2 MLA+MoE weight loader ─── */

TernaryError weights_from_gguf_deepseek2(TransformerWeights *w, const Config *cfg,
                                         const GGUFHeader *hdr, const MoEConfig *mc,
                                         GGUFWeightStore *store) {

    int dim         = cfg->dim;
    int hidden_dim  = cfg->hidden_dim;
    int nl          = cfg->n_layers;
    int n_heads     = cfg->n_heads;
    int n_kv_heads  = cfg->n_kv_heads;
    int first_dense = mc->first_k_dense_replace;
    int num_experts = mc->num_experts;
    int exp_hid     = mc->expert_hidden_dim;
    int sh_hid      = mc->shared_expert_hidden_dim;
    int lora        = mc->mla.kv_lora_rank;
    int nope        = mc->mla.qk_nope_head_dim;
    int rope_dim    = mc->mla.qk_rope_head_dim;
    int v_dim       = mc->mla.v_head_dim;
    char name_buf[128];

    /* MLA dimension sizes */
    int q_rows   = n_heads * (nope + rope_dim); /* e.g. 16*192=3072 */
    int kva_rows = lora + rope_dim;             /* e.g. 576          */
    int kvb_rows = n_kv_heads * (nope + v_dim); /* e.g. 16*256=4096  */

#define DS_LOAD_PROJ(field, tname, n_elems)                                                        \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                    \
        (field)[l] = (tn_i8 *)_f;                                                                  \
    } while (0)

/* Zero-copy loader for MLA projection weights (mmap pointer, may page-fault).
 * Falls back to F32 dequant for non-Q4_K types. */
#define DS_LOAD_MLA_PROJ(field, tname, n_elems)                                                    \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        if (_t->type == GGUF_TYPE_Q4_K) {                                                          \
            (field)[l]       = (tn_i8 *)_t->data; /* zero-copy raw Q4_K bytes */                   \
            w->has_mla_quant = true;                                                               \
        } else {                                                                                   \
            float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                               \
            if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                \
            (field)[l] = (tn_i8 *)_f;                                                              \
        }                                                                                          \
    } while (0)

/* Heap-copy loader for shared expert weights.
 * Stores raw Q4_K bytes when tensor is Q4_K (type 12), else dequantizes to F32.
 * Records the actual GGUF type per-layer in type_arr[l] for correct dispatch.
 * This avoids the global-flag pitfall where mixed-type layers cause wrong kernel dispatch
 * (e.g. ffn_down_shexp is Q5_K on layer 1 but Q4_K on all other layers). */
#define DS_LOAD_SHEXP_HEAP(field, tname, n_elems, type_arr)                                        \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        (type_arr)[l] = (int)_t->type;                                                             \
        if (_t->type == GGUF_TYPE_Q4_K) {                                                          \
            size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;                                   \
            uint8_t *_heap    = (uint8_t *)malloc(_q4k_bytes);                                     \
            if (!_heap) return TN_ERR_OOM;                                                         \
            memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);                                  \
            if (gguf_ld_store_add(store, _heap) != 0) {                                            \
                free(_heap);                                                                       \
                return TN_ERR_OOM;                                                                 \
            }                                                                                      \
            (field)[l] = (tn_i8 *)_heap;                                                           \
        } else {                                                                                   \
            float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                               \
            if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                \
            (field)[l] = (tn_i8 *)_f;                                                              \
        }                                                                                          \
    } while (0)

/* Generic heap-copy loader for any Q4_K tensor → flag pointer.
 * Sets *flag_ptr = true when tensor is Q4_K (no-op for other types).
 * Used for MLA projections where all layers have consistent type. */
#define DS_LOAD_Q4K_HEAP(field, tname, n_elems, flag_ptr)                                          \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        if (_t->type == GGUF_TYPE_Q4_K) {                                                          \
            size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;                                   \
            uint8_t *_heap    = (uint8_t *)malloc(_q4k_bytes);                                     \
            if (!_heap) return TN_ERR_OOM;                                                         \
            memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);                                  \
            if (gguf_ld_store_add(store, _heap) != 0) {                                            \
                free(_heap);                                                                       \
                return TN_ERR_OOM;                                                                 \
            }                                                                                      \
            (field)[l]  = (tn_i8 *)_heap;                                                          \
            *(flag_ptr) = true;                                                                    \
        } else {                                                                                   \
            float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                               \
            if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                \
            (field)[l] = (tn_i8 *)_f;                                                              \
        }                                                                                          \
    } while (0)

/* Heap-copy loader for MLA projection weights.
 * Copies Q4_K bytes from mmap → malloc'd heap buffer.
 * Advantages vs zero-copy mmap:
 *   - Guaranteed RAM residency (never evicted under memory pressure)
 *   - 8× smaller than F32 dequant (~200 MB vs ~1.5 GB for all MLA)
 *   - Stays warm in CPU cache across tokens
 * Analogous to llama.cpp CPU_REPACK approach. */
#define DS_LOAD_MLA_PROJ_HEAP(field, tname, n_elems)                                               \
    do {                                                                                           \
        snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);                                  \
        const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                                    \
        if (!_t) {                                                                                 \
            TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);            \
            return TN_ERR_INVALID_WEIGHTS;                                                         \
        }                                                                                          \
        if (_t->type == GGUF_TYPE_Q4_K) {                                                          \
            size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;                                   \
            uint8_t *_heap    = (uint8_t *)malloc(_q4k_bytes);                                     \
            if (!_heap) return TN_ERR_OOM;                                                         \
            memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);                                  \
            if (gguf_ld_store_add(store, _heap) != 0) {                                            \
                free(_heap);                                                                       \
                return TN_ERR_OOM;                                                                 \
            }                                                                                      \
            (field)[l]       = (tn_i8 *)_heap;                                                     \
            w->mla.has_quant = true;                                                               \
        } else {                                                                                   \
            float *_f = gguf_ld_tensor_to_f32(_t, (n_elems), store);                               \
            if (!_f) return TN_ERR_INVALID_WEIGHTS;                                                \
            (field)[l] = (tn_i8 *)_f;                                                              \
        }                                                                                          \
    } while (0)

    /* Allocate per-layer w2 and w13 quant type arrays */
    w->expert_quant.w2_per_layer  = (int *)calloc((size_t)nl, sizeof(int));
    w->expert_quant.w13_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    if (!w->expert_quant.w2_per_layer || !w->expert_quant.w13_per_layer) return TN_ERR_OOM;
    /* Allocate per-layer type arrays for shared expert projections */
    w->shared.w1_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    w->shared.w2_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    w->shared.w3_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    if (!w->shared.w1_type_per_layer || !w->shared.w2_type_per_layer ||
        !w->shared.w3_type_per_layer)
        return TN_ERR_OOM;

    for (int l = 0; l < nl; l++) {
        /* Attention norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
        w->attn.rms_att_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->attn.rms_att_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* FFN norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
        w->ffn.rms_ffn_weight[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->ffn.rms_ffn_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* KV-A sub-norm (applied to kv_latent after KV-A projection) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_kv_a_norm.weight", l);
        w->attn.rms_attn_sub_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)lora, store);
        if (!w->attn.rms_attn_sub_norm[l]) return TN_ERR_INVALID_WEIGHTS;

        /* MLA attention projections — Q4K heap-copy when available, F32 fallback otherwise.
         * DS_LOAD_MLA_PROJ_HEAP sets has_mla_quant=true so MLA_MATMUL dispatches
         * to parallel_matmul_q4k (7.1× less DRAM traffic; fits in L3 per layer). */
        DS_LOAD_MLA_PROJ_HEAP(w->mla.wq, "attn_q.weight", (size_t)q_rows * (size_t)dim);
        DS_LOAD_MLA_PROJ_HEAP(w->mla.wkv_a, "attn_kv_a_mqa.weight", (size_t)kva_rows * (size_t)dim);
        DS_LOAD_MLA_PROJ_HEAP(w->mla.wkv_b, "attn_kv_b.weight", (size_t)kvb_rows * (size_t)lora);
        DS_LOAD_MLA_PROJ_HEAP(w->attn.wo, "attn_output.weight", (size_t)dim * (size_t)dim);
        w->mla.sq[l]    = 1.0f;
        w->mla.skv_a[l] = 1.0f;
        w->mla.skv_b[l] = 1.0f;
        w->attn.so[l]   = 1.0f;

        if (l < first_dense) {
            /* Dense FFN layer — typed zero-copy (ffn.c dispatches on
             * w{1,2,3}_type via tn_dense_matmul_dispatch). */
            TernaryError pe;
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", l);
            pe = gguf_ld_proj_typed(hdr, store, name_buf, hidden_dim, dim, &w->ffn.w1[l],
                                    &w->ffn.w1_type[l]);
            if (pe != TN_OK) return pe;
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", l);
            pe = gguf_ld_proj_typed(hdr, store, name_buf, dim, hidden_dim, &w->ffn.w2[l],
                                    &w->ffn.w2_type[l]);
            if (pe != TN_OK) return pe;
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", l);
            pe = gguf_ld_proj_typed(hdr, store, name_buf, hidden_dim, dim, &w->ffn.w3[l],
                                    &w->ffn.w3_type[l]);
            if (pe != TN_OK) return pe;
            w->ffn.s1[l] = w->ffn.s2[l] = w->ffn.s3[l] = 1.0f;
        } else {
            /* MoE FFN layer */

            /* Router gate matrix (F32, zero-copy from mmap) */
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_inp.weight", l);
            const GGUFTensor *tgate = gguf_find_tensor(hdr, name_buf);
            if (!tgate) {
                TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);
                return TN_ERR_INVALID_WEIGHTS;
            }
            if (tgate->type != GGUF_TYPE_F32) {
                TN_LOG_ERR("gguf_loader",
                           "[gguf_loader] ffn_gate_inp.weight must be F32 (got %d)\n",
                           (int)tgate->type);
                return TN_ERR_INVALID_WEIGHTS;
            }
            w->moe.gate_w[l] = (tn_i8 *)tgate->data; /* zero-copy, F32 */
            w->moe.gate_s[l] = 1.0f;

            /* Stacked routed expert weights — kept quantized in mmap.
             * Per-expert pointers computed from the stacked tensor's base. */
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
                /* moe_s{1,2,3} unused — has_expert_quant flag drives dequant path */
                w->moe.s1[l][e] = w->moe.s2[l][e] = w->moe.s3[l][e] = 0.0f;
            }
            /* Record the actual quant type for this layer's down-proj AND gate/up experts */
            w->expert_quant.w2_per_layer[l]  = (int)td_exps->type;
            w->expert_quant.w13_per_layer[l] = (int)tg_exps->type;

            /* Shared expert weights — heap-copy as Q4_K when available.
             * Uses DS_LOAD_SHEXP_HEAP which records per-layer type for correct dispatch.
             * Different projections can have different quant types (e.g. gate/up=Q4_K, down=Q5_K). */
            DS_LOAD_SHEXP_HEAP(w->shared.w1, "ffn_gate_shexp.weight", (size_t)sh_hid * (size_t)dim,
                               w->shared.w1_type_per_layer);
            DS_LOAD_SHEXP_HEAP(w->shared.w2, "ffn_down_shexp.weight", (size_t)sh_hid * (size_t)dim,
                               w->shared.w2_type_per_layer);
            DS_LOAD_SHEXP_HEAP(w->shared.w3, "ffn_up_shexp.weight", (size_t)sh_hid * (size_t)dim,
                               w->shared.w3_type_per_layer);
            w->shared.s1[l] = w->shared.s2[l] = w->shared.s3[l] = 1.0f;
        }
    }

#undef DS_LOAD_PROJ

    /* Final RMS norm */
    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* Set expert quant flags */
    {
        /* Determine gate/up quant type from blk.first_dense.ffn_gate_exps */
        char tn[128];
        snprintf(tn, sizeof(tn), "blk.%d.ffn_gate_exps.weight", first_dense);
        const GGUFTensor *tg = gguf_find_tensor(hdr, tn);
        snprintf(tn, sizeof(tn), "blk.%d.ffn_down_exps.weight", first_dense);
        const GGUFTensor *td     = gguf_find_tensor(hdr, tn);
        w->expert_quant.w13_type = tg ? (int)tg->type : GGUF_TYPE_Q4_K;
        w->expert_quant.w2_type  = td ? (int)td->type : GGUF_TYPE_Q5_1;
        w->expert_quant.present  = true;

        /* (per-layer quant types stored in expert_w13/w2_quant_per_layer for dispatch) */
    }

    w->layers_are_ternary = false;
    w->cls.is_ternary     = false;
    w->cls.scale          = 1.0f;

    TN_OUT_INF("gguf_ds2",
               "[GGUF-DS2] Weights loaded (%d layers, %d dense + %d MoE, "
               "%d experts/layer, experts=quantized)\n",
               nl, first_dense, nl - first_dense, num_experts);
    return TN_OK;
}
