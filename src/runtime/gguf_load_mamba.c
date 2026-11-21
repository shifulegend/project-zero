/* Mamba (pure state-space model) weight loader. */

#include <stdio.h>

#include "core/moe_config.h"
#include "core/platform.h"
#include "math/matmul_q4k_x8.h"
#include "runtime/gguf_load_util.h"
#include "runtime/gguf_loader.h"
#include "runtime/mamba_state.h"

/* ── weights_from_gguf_mamba() — Phase 22-SSM Mamba-1 loader ─────────────── */

/* Mamba projection: shared typed policy (gguf_ld_proj_typed above). */
static TernaryError mamba_load_proj(const GGUFHeader *hdr, GGUFWeightStore *store, const char *name,
                                    int d_rows, int n_inner, tn_i8 **out, int *type_out) {
    return gguf_ld_proj_typed(hdr, store, name, d_rows, n_inner, out, type_out);
}

TernaryError weights_from_gguf_mamba(TransformerWeights *w, const Config *cfg,
                                     const GGUFHeader *hdr, const MoEConfig *mc,
                                     GGUFWeightStore *store) {
    TernaryError e = gguf_ld_bf16_embedding_and_output(w, cfg, hdr, store);
    if (e != TN_OK) return e;
    w->rms_final_weight = gguf_ld_norm_to_f32(hdr, "output_norm.weight", (size_t)cfg->dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    MambaWeights *mw = NULL;
    e                = mamba_weights_alloc(&mw, cfg->n_layers);
    if (e != TN_OK) return e;
    w->mamba = mw; /* owned by w from here on (weights_free_pointers) */

    const int dim = cfg->dim, di = mc->ssm_inner_size, ds = mc->ssm_state_size;
    const int dc = mc->ssm_conv_kernel, dr = mc->ssm_time_step_rank;
    char nm[128];
#define MAMBA_NORM(dst, suffix, count)                                                             \
    do {                                                                                           \
        snprintf(nm, sizeof(nm), "blk.%d." suffix, l);                                             \
        (dst) = gguf_ld_norm_to_f32(hdr, nm, (size_t)(count), store);                              \
        if (!(dst)) return TN_ERR_INVALID_WEIGHTS;                                                 \
    } while (0)
#define MAMBA_PROJ(dst, type, suffix, rows, cols)                                                  \
    do {                                                                                           \
        snprintf(nm, sizeof(nm), "blk.%d." suffix, l);                                             \
        TernaryError pe = mamba_load_proj(hdr, store, nm, (rows), (cols), &(dst), &(type));        \
        if (pe != TN_OK) return pe;                                                                \
    } while (0)
    for (int l = 0; l < cfg->n_layers; l++) {
        MAMBA_NORM(mw->norm[l], "attn_norm.weight", dim);
        MAMBA_NORM(mw->conv_w[l], "ssm_conv1d.weight", (size_t)di * dc);
        MAMBA_NORM(mw->conv_b[l], "ssm_conv1d.bias", di);
        MAMBA_NORM(mw->dt_b[l], "ssm_dt.bias", di);
        MAMBA_NORM(mw->A[l], "ssm_a", (size_t)di * ds);
        MAMBA_NORM(mw->D[l], "ssm_d", di);
        MAMBA_PROJ(mw->in_proj[l], mw->in_type[l], "ssm_in.weight", 2 * di, dim);
        MAMBA_PROJ(mw->x_proj[l], mw->x_type[l], "ssm_x.weight", dr + 2 * ds, di);
        MAMBA_PROJ(mw->dt_proj[l], mw->dt_type[l], "ssm_dt.weight", di, dr);
        MAMBA_PROJ(mw->out_proj[l], mw->out_type[l], "ssm_out.weight", dim, di);
    }
#undef MAMBA_NORM
#undef MAMBA_PROJ
    weights_build_classifier_quant(w, cfg);
    return TN_OK;
}
