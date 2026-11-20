#include "core/moe_weights.h"

#include <stdlib.h>
#include <string.h>

#include "core/unpack.h"

/* -----------------------------------------------------------------------
 * Allocation
 * ----------------------------------------------------------------------- */

TernaryError moe_weights_alloc(TransformerWeights *w, const Config *cfg, const MoEConfig *mc) {
    if (!mc->is_moe) return TN_OK; /* dense model — nothing to allocate */

    int nl              = cfg->n_layers;
    int ne              = mc->num_experts;
    w->moe.alloc_layers = nl;

    /* Gate weight pointers: one per layer */
    w->moe.gate_w = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w->moe.gate_s = (float *)calloc(nl, sizeof(float));

    /* Expert weight pointers: [n_layers][num_experts] */
    w->moe.w1 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w->moe.w2 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w->moe.w3 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w->moe.s1 = (float **)calloc(nl, sizeof(float *));
    w->moe.s2 = (float **)calloc(nl, sizeof(float *));
    w->moe.s3 = (float **)calloc(nl, sizeof(float *));

    if (!w->moe.gate_w || !w->moe.gate_s || !w->moe.w1 || !w->moe.w2 || !w->moe.w3 || !w->moe.s1 ||
        !w->moe.s2 || !w->moe.s3) {
        moe_weights_free(w, mc);
        return TN_ERR_OOM;
    }

    for (int l = 0; l < nl; l++) {
        w->moe.w1[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w->moe.w2[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w->moe.w3[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w->moe.s1[l] = (float *)calloc(ne, sizeof(float));
        w->moe.s2[l] = (float *)calloc(ne, sizeof(float));
        w->moe.s3[l] = (float *)calloc(ne, sizeof(float));

        if (!w->moe.w1[l] || !w->moe.w2[l] || !w->moe.w3[l] || !w->moe.s1[l] || !w->moe.s2[l] ||
            !w->moe.s3[l]) {
            moe_weights_free(w, mc);
            return TN_ERR_OOM;
        }
    }

    /* Shared expert weight pointers (DeepSeek-V2 style) */
    if (mc->n_shared_experts > 0) {
        w->shared.w1 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->shared.w2 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->shared.w3 = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->shared.s1 = (float *)calloc(nl, sizeof(float));
        w->shared.s2 = (float *)calloc(nl, sizeof(float));
        w->shared.s3 = (float *)calloc(nl, sizeof(float));

        if (!w->shared.w1 || !w->shared.w2 || !w->shared.w3 || !w->shared.s1 || !w->shared.s2 ||
            !w->shared.s3) {
            moe_weights_free(w, mc);
            return TN_ERR_OOM;
        }
    }

    /* Phase 17.6: MLA weight pointer arrays (NULL for non-MLA models) */
    if (mc->mla.enabled) {
        w->mla.wq    = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->mla.wkv_a = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->mla.wkv_b = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
        w->mla.sq    = (float *)calloc(nl, sizeof(float));
        w->mla.skv_a = (float *)calloc(nl, sizeof(float));
        w->mla.skv_b = (float *)calloc(nl, sizeof(float));

        if (!w->mla.wq || !w->mla.wkv_a || !w->mla.wkv_b || !w->mla.sq || !w->mla.skv_a ||
            !w->mla.skv_b) {
            moe_weights_free(w, mc);
            return TN_ERR_OOM;
        }
    }

    return TN_OK;
}

/* -----------------------------------------------------------------------
 * Free
 * ----------------------------------------------------------------------- */

void moe_weights_free(TransformerWeights *w, const MoEConfig *mc) {
    if (!mc || !mc->is_moe) return;

    bool free_routed_payloads = !w->expert_quant.present;

    /* Traverse w->moe_w1 rows until we hit a NULL row (calloc'd to NULL).
     * Null each pointer immediately after freeing so calling this function
     * twice is safe (idempotent). */
    if (w->moe.w1) {
        for (int l = 0; l < w->moe.alloc_layers; l++) {
            if (!w->moe.w1[l] || !w->moe.w2[l] || !w->moe.w3[l]) continue;
            for (int e = 0; e < mc->num_experts; e++) {
                if (free_routed_payloads && w->moe.w1[l][e]) free(w->moe.w1[l][e]);
                if (free_routed_payloads && w->moe.w2[l][e]) free(w->moe.w2[l][e]);
                if (free_routed_payloads && w->moe.w3[l][e]) free(w->moe.w3[l][e]);
                w->moe.w1[l][e] = NULL;
                w->moe.w2[l][e] = NULL;
                w->moe.w3[l][e] = NULL;
            }
            if (w->moe.s1[l]) {
                free(w->moe.s1[l]);
                w->moe.s1[l] = NULL;
            }
            if (w->moe.s2[l]) {
                free(w->moe.s2[l]);
                w->moe.s2[l] = NULL;
            }
            if (w->moe.s3[l]) {
                free(w->moe.s3[l]);
                w->moe.s3[l] = NULL;
            }
            free(w->moe.w1[l]);
            w->moe.w1[l] = NULL;
            free(w->moe.w2[l]);
            w->moe.w2[l] = NULL;
            free(w->moe.w3[l]);
            w->moe.w3[l] = NULL;
        }
        free(w->moe.w1);
        w->moe.w1 = NULL;
        free(w->moe.w2);
        w->moe.w2 = NULL;
        free(w->moe.w3);
        w->moe.w3 = NULL;
        free(w->moe.s1);
        w->moe.s1 = NULL;
        free(w->moe.s2);
        w->moe.s2 = NULL;
        free(w->moe.s3);
        w->moe.s3 = NULL;
    }
    w->moe.alloc_layers = 0;

    if (w->moe.gate_w) {
        /* Gate weight arrays: only the pointer-of-pointers is heap-alloc'd by
         * moe_weights_alloc(). For mmap'd files the data was NOT heap alloc'd
         * (pointed into the mmap), so we only free the pointer array itself.
         * In tests the actual weight buffers are freed separately before this call. */
        free(w->moe.gate_w);
        w->moe.gate_w = NULL;
    }
    if (w->moe.gate_s) {
        free(w->moe.gate_s);
        w->moe.gate_s = NULL;
    }

    /* Shared expert pointer arrays (data is mmap'd, only free the arrays) */
    if (w->shared.w1) {
        free(w->shared.w1);
        w->shared.w1 = NULL;
    }
    if (w->shared.w2) {
        free(w->shared.w2);
        w->shared.w2 = NULL;
    }
    if (w->shared.w3) {
        free(w->shared.w3);
        w->shared.w3 = NULL;
    }
    if (w->shared.s1) {
        free(w->shared.s1);
        w->shared.s1 = NULL;
    }
    if (w->shared.s2) {
        free(w->shared.s2);
        w->shared.s2 = NULL;
    }
    if (w->shared.s3) {
        free(w->shared.s3);
        w->shared.s3 = NULL;
    }

    /* MLA weight pointer arrays (data is mmap'd; only free the pointer arrays) */
    if (w->mla.wq) {
        free(w->mla.wq);
        w->mla.wq = NULL;
    }
    if (w->mla.wkv_a) {
        free(w->mla.wkv_a);
        w->mla.wkv_a = NULL;
    }
    if (w->mla.wkv_b) {
        free(w->mla.wkv_b);
        w->mla.wkv_b = NULL;
    }
    if (w->mla.sq) {
        free(w->mla.sq);
        w->mla.sq = NULL;
    }
    if (w->mla.skv_a) {
        free(w->mla.skv_a);
        w->mla.skv_a = NULL;
    }
    if (w->mla.skv_b) {
        free(w->mla.skv_b);
        w->mla.skv_b = NULL;
    }
}


/* -----------------------------------------------------------------------
 * Map weights from mmap'd binary file
 *
 * Per-layer MoE binary layout (for MoE layers only):
 *   [gate_w: packed_bytes(dim × num_experts)] ALIGN64
 *   [gate_scale: float32]                     ALIGN64
 *   For each expert e in [0, num_experts):
 *     [w1: packed_bytes(dim × expert_hdim)]   ALIGN64
 *     [s1: float32]                           ALIGN64
 *     [w3: packed_bytes(dim × expert_hdim)]   ALIGN64
 *     [s3: float32]                           ALIGN64
 *     [w2: packed_bytes(expert_hdim × dim)]   ALIGN64
 *     [s2: float32]                           ALIGN64
 * ----------------------------------------------------------------------- */

TernaryError moe_weights_map(TransformerWeights *w, const Config *cfg, const MoEConfig *mc,
                             tn_i8 *data, size_t data_size, tn_i8 **ptr_inout) {
    if (!mc->is_moe) return TN_OK;

    int nl        = cfg->n_layers;
    int ne        = mc->num_experts;
    int dim       = cfg->dim;
    int ehdim     = mc->expert_hidden_dim;
    int first_moe = mc->first_k_dense_replace;

    /* Absolute alignment base is 64 (same as weights_map) */
    const size_t offset_base = 64;

#define MOE_MAP_PTR(dest, bytes)                                                                   \
    do {                                                                                           \
        tn_i8 *_ptr = *ptr_inout;                                                                  \
        if ((size_t)(_ptr - data + (bytes)) > data_size) return TN_ERR_INVALID_WEIGHTS;            \
        (dest) = (void *)_ptr;                                                                     \
        *ptr_inout += (bytes);                                                                     \
    } while (0)

#define MOE_ALIGN64()                                                                              \
    do {                                                                                           \
        size_t _abs = (size_t)(*ptr_inout - data) + offset_base;                                   \
        size_t _pad = (64 - (_abs % 64)) % 64;                                                     \
        if (_pad > 0) {                                                                            \
            if ((size_t)(*ptr_inout - data + _pad) > data_size) return TN_ERR_INVALID_WEIGHTS;     \
            *ptr_inout += _pad;                                                                    \
        }                                                                                          \
    } while (0)

#define MOE_COPY_SCALE(dest)                                                                       \
    do {                                                                                           \
        tn_i8 *_ptr = *ptr_inout;                                                                  \
        if ((size_t)(_ptr - data + 4) > data_size) return TN_ERR_INVALID_WEIGHTS;                  \
        memcpy((dest), _ptr, 4);                                                                   \
        *ptr_inout += 4;                                                                           \
        MOE_ALIGN64();                                                                             \
    } while (0)

    for (int l = 0; l < nl; l++) {
        /* Dense layers use the regular FFN — skip MoE block */
        if (l < first_moe) continue;

        /* Gate matrix: [dim × num_experts] packed ternary */
        size_t gate_bytes = packed_bytes((size_t)dim * ne);
        MOE_MAP_PTR(w->moe.gate_w[l], gate_bytes);
        MOE_ALIGN64();
        MOE_COPY_SCALE(&w->moe.gate_s[l]);

        /* Per-expert FFN weights */
        size_t w1_bytes = packed_bytes((size_t)dim * ehdim);
        size_t w2_bytes = packed_bytes((size_t)ehdim * dim);
        size_t w3_bytes = w1_bytes;

        for (int e = 0; e < ne; e++) {
            MOE_MAP_PTR(w->moe.w1[l][e], w1_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->moe.s1[l][e]);

            MOE_MAP_PTR(w->moe.w3[l][e], w3_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->moe.s3[l][e]);

            MOE_MAP_PTR(w->moe.w2[l][e], w2_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->moe.s2[l][e]);
        }

        /* Shared expert FFN (DeepSeek-V2 style):
         * Layout: [sw1] ALIGN64 [ss1] ALIGN64
         *         [sw3] ALIGN64 [ss3] ALIGN64
         *         [sw2] ALIGN64 [ss2] ALIGN64
         * Dimensions: [dim × shared_hdim] for w1/w3, [shared_hdim × dim] for w2 */
        if (mc->n_shared_experts > 0 && w->shared.w1) {
            int shdim        = mc->shared_expert_hidden_dim;
            size_t sw1_bytes = packed_bytes((size_t)dim * shdim);
            size_t sw2_bytes = packed_bytes((size_t)shdim * dim);
            size_t sw3_bytes = sw1_bytes;

            MOE_MAP_PTR(w->shared.w1[l], sw1_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->shared.s1[l]);

            MOE_MAP_PTR(w->shared.w3[l], sw3_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->shared.s3[l]);

            MOE_MAP_PTR(w->shared.w2[l], sw2_bytes);
            MOE_ALIGN64();
            MOE_COPY_SCALE(&w->shared.s2[l]);
        }
    }

#undef MOE_MAP_PTR
#undef MOE_ALIGN64
#undef MOE_COPY_SCALE

    return TN_OK;
}
