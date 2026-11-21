/* DeepSeek-V2 (MLA + MoE) weight loader. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/fp16.h"
#include "core/log.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include "math/matmul_q4k_x8.h"
#include "memory/mapped_file.h"
#include "runtime/gguf_load_util.h"
#include "runtime/gguf_loader.h"
#include "runtime/mamba_state.h"

/* ── weights_from_gguf_deepseek2() — DeepSeek-V2 MLA+MoE weight loader ─── */

/* Loads blk.<l>.<tname> into *slot: a Q4_K tensor is copied from the mapping into a heap buffer
 * owned by the store (guaranteed RAM residency, never evicted under memory pressure; about 8x
 * smaller than F32 dequant; warm in cache across tokens: analogous to llama.cpp's CPU_REPACK),
 * anything else is dequantised to F32. Optionally records the tensor's GGUF type in *type_out
 * (per-layer dispatch: different projections can differ, e.g. ffn_down_shexp is Q5_K on layer 1
 * and Q4_K elsewhere, so a global flag would pick the wrong kernel) and sets *q4k_flag when the
 * tensor was Q4_K. */
static TernaryError ds_load_heap_q4k(const GGUFHeader *hdr, GGUFWeightStore *store, int l,
                                     const char *tname, size_t n_elems, tn_i8 **slot, int *type_out,
                                     bool *q4k_flag) {
    char name_buf[128];
    snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", l, tname);
    const GGUFTensor *t = gguf_find_tensor(hdr, name_buf);
    if (!t) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);
        return TN_ERR_INVALID_WEIGHTS;
    }
    if (type_out) *type_out = (int)t->type;
    if (t->type != GGUF_TYPE_Q4_K) {
        float *f = gguf_ld_tensor_to_f32(t, n_elems, store);
        if (!f) return TN_ERR_INVALID_WEIGHTS;
        *slot = (tn_i8 *)f;
        return TN_OK;
    }
    size_t q4k_bytes = (n_elems / 256) * 144;
    uint8_t *heap    = (uint8_t *)malloc(q4k_bytes);
    if (!heap) return TN_ERR_OOM;
    memcpy(heap, (const uint8_t *)t->data, q4k_bytes);
    if (gguf_ld_store_add(store, heap) != 0) {
        free(heap);
        return TN_ERR_OOM;
    }
    *slot = (tn_i8 *)heap;
    if (q4k_flag) *q4k_flag = true;
    return TN_OK;
}


/* MLA attention projections of layer l: Q4_K heap copy when available, F32 fallback otherwise. A
 * Q4_K projection sets has_mla_quant so MLA matmuls dispatch to parallel_matmul_q4k (7.1x less DRAM
 * traffic; fits in L3 per layer). */
static TernaryError ds_load_mla_layer(TransformerWeights *w, const Config *cfg,
                                      const GGUFHeader *hdr, const MoEConfig *mc,
                                      GGUFWeightStore *store, int l) {
    size_t dim  = (size_t)cfg->dim;
    size_t lora = (size_t)mc->mla.kv_lora_rank;
    size_t q_rows =
        (size_t)cfg->n_heads * (size_t)(mc->mla.qk_nope_head_dim + mc->mla.qk_rope_head_dim);
    size_t kva_rows = lora + (size_t)mc->mla.qk_rope_head_dim;
    size_t kvb_rows =
        (size_t)cfg->n_kv_heads * (size_t)(mc->mla.qk_nope_head_dim + mc->mla.v_head_dim);
    TernaryError e = ds_load_heap_q4k(hdr, store, l, "attn_q.weight", q_rows * dim, &w->mla.wq[l],
                                      NULL, &w->mla.has_quant);
    if (e != TN_OK) return e;
    e = ds_load_heap_q4k(hdr, store, l, "attn_kv_a_mqa.weight", kva_rows * dim, &w->mla.wkv_a[l],
                         NULL, &w->mla.has_quant);
    if (e != TN_OK) return e;
    e = ds_load_heap_q4k(hdr, store, l, "attn_kv_b.weight", kvb_rows * lora, &w->mla.wkv_b[l], NULL,
                         &w->mla.has_quant);
    if (e != TN_OK) return e;
    return ds_load_heap_q4k(hdr, store, l, "attn_output.weight", dim * dim, &w->attn.wo[l], NULL,
                            &w->mla.has_quant);
}

/* Shared expert weights of layer l: heap copy as Q4_K when available, recording each projection's
 * own type per layer (gate/up may be Q4_K while down is Q5_K). */
static TernaryError ds_load_shexp_layer(TransformerWeights *w, const Config *cfg,
                                        const GGUFHeader *hdr, const MoEConfig *mc,
                                        GGUFWeightStore *store, int l) {
    const size_t n = (size_t)mc->shared_expert_hidden_dim * (size_t)cfg->dim;
    TernaryError e = ds_load_heap_q4k(hdr, store, l, "ffn_gate_shexp.weight", n, &w->shared.w1[l],
                                      &w->shared.w1_type_per_layer[l], NULL);
    if (e != TN_OK) return e;
    e = ds_load_heap_q4k(hdr, store, l, "ffn_down_shexp.weight", n, &w->shared.w2[l],
                         &w->shared.w2_type_per_layer[l], NULL);
    if (e != TN_OK) return e;
    return ds_load_heap_q4k(hdr, store, l, "ffn_up_shexp.weight", n, &w->shared.w3[l],
                            &w->shared.w3_type_per_layer[l], NULL);
}

/* Layer l's three RMS norms: attention, FFN and the KV-A sub-norm (applied to kv_latent after the
 * KV-A projection). */
static TernaryError ds_load_norms(TransformerWeights *w, const Config *cfg, const GGUFHeader *hdr,
                                  const MoEConfig *mc, GGUFWeightStore *store, int l) {
    char name_buf[128];
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
    w->attn.norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)cfg->dim, store);
    if (!w->attn.norm[l]) return TN_ERR_INVALID_WEIGHTS;

    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
    w->ffn.norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)cfg->dim, store);
    if (!w->ffn.norm[l]) return TN_ERR_INVALID_WEIGHTS;

    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_kv_a_norm.weight", l);
    w->attn.sub_norm[l] = gguf_ld_norm_to_f32(hdr, name_buf, (size_t)mc->mla.kv_lora_rank, store);
    return w->attn.sub_norm[l] ? TN_OK : TN_ERR_INVALID_WEIGHTS;
}

/* Dense FFN layer: typed zero-copy (ffn.c dispatches on w{1,2,3}_type via
 * tn_dense_matmul_dispatch). */
static TernaryError ds_load_dense_ffn(TransformerWeights *w, const Config *cfg,
                                      const GGUFHeader *hdr, GGUFWeightStore *store, int l) {
    int dim = cfg->dim, hidden_dim = cfg->hidden_dim;
    char name_buf[128];
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", l);
    TernaryError e = gguf_ld_proj_typed(hdr, store, name_buf, hidden_dim, dim, &w->ffn.w1[l],
                                        &w->ffn.w1_type[l]);
    if (e != TN_OK) return e;
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", l);
    e = gguf_ld_proj_typed(hdr, store, name_buf, dim, hidden_dim, &w->ffn.w2[l],
                           &w->ffn.w2_type[l]);
    if (e != TN_OK) return e;
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", l);
    e = gguf_ld_proj_typed(hdr, store, name_buf, hidden_dim, dim, &w->ffn.w3[l],
                           &w->ffn.w3_type[l]);
    if (e != TN_OK) return e;
    w->ffn.s1[l] = w->ffn.s2[l] = w->ffn.s3[l] = 1.0f;
    return TN_OK;
}

/* MoE FFN layer: the router and stacked routed experts stay quantized in the mapping; the shared
 * expert is copied to the heap. */
static TernaryError ds_load_moe_ffn(TransformerWeights *w, const Config *cfg, const GGUFHeader *hdr,
                                    const MoEConfig *mc, GGUFWeightStore *store, int l) {
    TernaryError e = gguf_ld_moe_layer(hdr, w, l, mc->num_experts,
                                       (size_t)mc->expert_hidden_dim * (size_t)cfg->dim);
    if (e != TN_OK) return e;
    e = ds_load_shexp_layer(w, cfg, hdr, mc, store, l);
    if (e != TN_OK) return e;
    w->shared.s1[l] = w->shared.s2[l] = w->shared.s3[l] = 1.0f;
    return TN_OK;
}

/* The gate/up and down quant types of the routed experts, read from the first MoE layer (the
 * per-layer types for dispatch are stored in w2/w13_per_layer). */
static void ds_set_expert_quant(TransformerWeights *w, const GGUFHeader *hdr, int first_dense) {
    char tn[128];
    snprintf(tn, sizeof(tn), "blk.%d.ffn_gate_exps.weight", first_dense);
    const GGUFTensor *tg = gguf_find_tensor(hdr, tn);
    snprintf(tn, sizeof(tn), "blk.%d.ffn_down_exps.weight", first_dense);
    const GGUFTensor *td     = gguf_find_tensor(hdr, tn);
    w->expert_quant.w13_type = tg ? (int)tg->type : GGUF_TYPE_Q4_K;
    w->expert_quant.w2_type  = td ? (int)td->type : GGUF_TYPE_Q5_1;
    w->expert_quant.present  = true;
}

TernaryError weights_from_gguf_deepseek2(TransformerWeights *w, const Config *cfg,
                                         const GGUFHeader *hdr, const MoEConfig *mc,
                                         GGUFWeightStore *store) {
    int dim         = cfg->dim;
    int nl          = cfg->n_layers;
    int first_dense = mc->first_k_dense_replace;

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
        TernaryError e = ds_load_norms(w, cfg, hdr, mc, store, l);
        if (e != TN_OK) return e;
        e = ds_load_mla_layer(w, cfg, hdr, mc, store, l);
        if (e != TN_OK) return e;
        w->mla.sq[l]    = 1.0f;
        w->mla.skv_a[l] = 1.0f;
        w->mla.skv_b[l] = 1.0f;
        w->attn.so[l]   = 1.0f;

        e = l < first_dense ? ds_load_dense_ffn(w, cfg, hdr, store, l)
                            : ds_load_moe_ffn(w, cfg, hdr, mc, store, l);
        if (e != TN_OK) return e;
    }

    /* Final RMS norm */
    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    ds_set_expert_quant(w, hdr, first_dense);
    w->layers_are_ternary = false;
    w->cls.is_ternary     = false;
    w->cls.scale          = 1.0f;

    TN_OUT_INF("gguf_ds2",
               "[GGUF-DS2] Weights loaded (%d layers, %d dense + %d MoE, "
               "%d experts/layer, experts=quantized)\n",
               nl, first_dense, nl - first_dense, mc->num_experts);
    return TN_OK;
}
