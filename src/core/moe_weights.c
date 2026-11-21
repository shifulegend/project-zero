#include "core/moe_weights.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "core/unpack.h"

/* -----------------------------------------------------------------------
 * Allocation
 * ----------------------------------------------------------------------- */

/* One pointer array hanging off TransformerWeights: where the pointer lives and how big an element
 * is. The arrays are all per-layer (calloc'd, so unset entries read NULL). */
typedef struct {
    size_t offset;
    size_t elem;
} ArraySlot;

#define PTR_SLOT(field)                                                                            \
    { offsetof(TransformerWeights, field), sizeof(void *) }
#define F32_SLOT(field)                                                                            \
    { offsetof(TransformerWeights, field), sizeof(float) }

static const ArraySlot k_moe_layer_arrays[] = {
    PTR_SLOT(moe.gate_w), F32_SLOT(moe.gate_s), PTR_SLOT(moe.w1), PTR_SLOT(moe.w2),
    PTR_SLOT(moe.w3),     PTR_SLOT(moe.s1),     PTR_SLOT(moe.s2), PTR_SLOT(moe.s3),
};
/* Per layer, each of these is an array of rows, one row per expert. */
static const ArraySlot k_moe_expert_rows[] = {
    {offsetof(TransformerWeights, moe.w1), sizeof(void *)},
    {offsetof(TransformerWeights, moe.w2), sizeof(void *)},
    {offsetof(TransformerWeights, moe.w3), sizeof(void *)},
    {offsetof(TransformerWeights, moe.s1), sizeof(float)},
    {offsetof(TransformerWeights, moe.s2), sizeof(float)},
    {offsetof(TransformerWeights, moe.s3), sizeof(float)},
};
/* DeepSeek-V2 style shared expert. */
static const ArraySlot k_shared_arrays[] = {
    PTR_SLOT(shared.w1), PTR_SLOT(shared.w2), PTR_SLOT(shared.w3),
    F32_SLOT(shared.s1), F32_SLOT(shared.s2), F32_SLOT(shared.s3),
};
/* Phase 17.6: MLA weight pointer arrays (NULL for non-MLA models). */
static const ArraySlot k_mla_arrays[] = {
    PTR_SLOT(mla.wq), PTR_SLOT(mla.wkv_a), PTR_SLOT(mla.wkv_b),
    F32_SLOT(mla.sq), F32_SLOT(mla.skv_a), F32_SLOT(mla.skv_b),
};

#define N_SLOTS(t) (sizeof(t) / sizeof((t)[0]))

static void *slot_get(const void *base, size_t offset) {
    void *p;
    memcpy(&p, (const char *)base + offset, sizeof p);
    return p;
}

static void slot_set(void *base, size_t offset, void *p) {
    memcpy((char *)base + offset, &p, sizeof p);
}

/* Frees the pointer stored at base+offset and nulls it (idempotent). */
static void slot_free(void *base, size_t offset) {
    free(slot_get(base, offset));
    slot_set(base, offset, NULL);
}

/* calloc(count, elem) into each slot; 0 when any allocation failed (all are attempted). */
static int alloc_slots(TransformerWeights *w, const ArraySlot *slots, size_t n, size_t count) {
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        void *p = calloc(count, slots[i].elem);
        slot_set(w, slots[i].offset, p);
        ok &= p != NULL;
    }
    return ok;
}

TernaryError moe_weights_alloc(TransformerWeights *w, const Config *cfg, const MoEConfig *mc) {
    if (!mc->is_moe) return TN_OK; /* dense model: nothing to allocate */

    size_t nl           = (size_t)cfg->n_layers;
    size_t ne           = (size_t)mc->num_experts;
    w->moe.alloc_layers = cfg->n_layers;

    /* Gate weight pointers and expert weight pointers: one per layer / [n_layers][num_experts] */
    int ok = alloc_slots(w, k_moe_layer_arrays, N_SLOTS(k_moe_layer_arrays), nl);
    for (size_t l = 0; ok && l < nl; l++) {
        for (size_t k = 0; k < N_SLOTS(k_moe_expert_rows); k++) {
            void **rows = (void **)slot_get(w, k_moe_expert_rows[k].offset);
            rows[l]     = calloc(ne, k_moe_expert_rows[k].elem);
            ok &= rows[l] != NULL;
        }
    }
    if (ok && mc->n_shared_experts > 0)
        ok = alloc_slots(w, k_shared_arrays, N_SLOTS(k_shared_arrays), nl);
    if (ok && mc->mla.enabled) ok = alloc_slots(w, k_mla_arrays, N_SLOTS(k_mla_arrays), nl);
    if (ok) return TN_OK;
    moe_weights_free(w, mc);
    return TN_ERR_OOM;
}

/* -----------------------------------------------------------------------
 * Free
 * ----------------------------------------------------------------------- */

/* A row of per-expert payload pointers: free each payload (unless the quantised-expert path owns
 * them) and null it. */
static void clear_payload_row(void **payloads, int n_experts, int free_payloads) {
    for (int e = 0; e < n_experts; e++) {
        if (free_payloads) free(payloads[e]);
        payloads[e] = NULL;
    }
}

/* Frees one layer's expert rows. The routed payloads are those of the first three row arrays (w1,
 * w2, w3). Every array and row is checked on its own, so a layer left partly allocated by a failed
 * moe_weights_alloc() frees what it has. */
static void free_layer_rows(TransformerWeights *w, const MoEConfig *mc, int l, int free_payloads) {
    for (size_t k = 0; k < N_SLOTS(k_moe_expert_rows); k++) {
        void **arr = (void **)slot_get(w, k_moe_expert_rows[k].offset);
        if (!arr) continue;
        if (k < 3 && arr[l]) clear_payload_row((void **)arr[l], mc->num_experts, free_payloads);
        free(arr[l]);
        arr[l] = NULL;
    }
}

void moe_weights_free(TransformerWeights *w, const MoEConfig *mc) {
    if (!mc || !mc->is_moe) return;

    int free_payloads = !w->expert_quant.present;

    /* Null each pointer immediately after freeing so calling this function twice is safe
     * (idempotent). */
    if (w->moe.w1) {
        for (int l = 0; l < w->moe.alloc_layers; l++) free_layer_rows(w, mc, l, free_payloads);
        for (size_t k = 0; k < N_SLOTS(k_moe_expert_rows); k++)
            slot_free(w, k_moe_expert_rows[k].offset);
    }
    w->moe.alloc_layers = 0;

    /* Only the pointer arrays are heap-allocated by moe_weights_alloc(): for mmap'd files the data
     * points into the mmap. In tests the actual weight buffers are freed separately before this
     * call. */
    slot_free(w, offsetof(TransformerWeights, moe.gate_w));
    slot_free(w, offsetof(TransformerWeights, moe.gate_s));
    for (size_t k = 0; k < N_SLOTS(k_shared_arrays); k++) slot_free(w, k_shared_arrays[k].offset);
    for (size_t k = 0; k < N_SLOTS(k_mla_arrays); k++) slot_free(w, k_mla_arrays[k].offset);
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
