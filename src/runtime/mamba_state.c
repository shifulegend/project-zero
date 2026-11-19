/*
 * src/core/mamba_state.c
 *
 * Allocation of the Mamba weight tables and recurrent state. Split out of transformer/mamba.c so the
 * core weight and run-state code can free them without including the transformer module; the code
 * is unchanged.
 */

#include "runtime/mamba_state.h"

#include <stdlib.h>
#include <string.h>

#include "core/platform.h"
#include "memory/aligned_alloc.h"
#include "memory/safe_alloc.h"

TernaryError mamba_weights_alloc(MambaWeights **out, int n_layers) {
    if (!out || n_layers <= 0) return TN_ERR_INVALID_ARGS;
    MambaWeights *mw = (MambaWeights *)calloc(1, sizeof(*mw));
    if (!mw) return TN_ERR_OOM;
    mw->n_layers = n_layers;
    size_t L     = (size_t)n_layers;
    mw->norm     = calloc(L, sizeof(float *));
    mw->conv_w   = calloc(L, sizeof(float *));
    mw->conv_b   = calloc(L, sizeof(float *));
    mw->dt_b     = calloc(L, sizeof(float *));
    mw->A        = calloc(L, sizeof(float *));
    mw->D        = calloc(L, sizeof(float *));
    mw->in_proj  = calloc(L, sizeof(tn_i8 *));
    mw->x_proj   = calloc(L, sizeof(tn_i8 *));
    mw->dt_proj  = calloc(L, sizeof(tn_i8 *));
    mw->out_proj = calloc(L, sizeof(tn_i8 *));
    mw->in_type  = calloc(L, sizeof(int));
    mw->x_type   = calloc(L, sizeof(int));
    mw->dt_type  = calloc(L, sizeof(int));
    mw->out_type = calloc(L, sizeof(int));
    if (!mw->norm || !mw->conv_w || !mw->conv_b || !mw->dt_b || !mw->A || !mw->D || !mw->in_proj ||
        !mw->x_proj || !mw->dt_proj || !mw->out_proj || !mw->in_type || !mw->x_type ||
        !mw->dt_type || !mw->out_type) {
        mamba_weights_free(mw);
        return TN_ERR_OOM;
    }
    *out = mw;
    return TN_OK;
}

void mamba_weights_free(MambaWeights *mw) {
    if (!mw) return;
    /* Tensor data is owned by the mmap / GGUF weight store, not by us. */
    free(mw->norm);
    free(mw->conv_w);
    free(mw->conv_b);
    free(mw->dt_b);
    free(mw->A);
    free(mw->D);
    free(mw->in_proj);
    free(mw->x_proj);
    free(mw->dt_proj);
    free(mw->out_proj);
    free(mw->in_type);
    free(mw->x_type);
    free(mw->dt_type);
    free(mw->out_type);
    free(mw);
}

TernaryError mamba_state_alloc(MambaState **out, int n_layers, const MoEConfig *mc) {
    if (!out || !mc || n_layers <= 0 || mc->ssm_inner_size <= 0 || mc->ssm_state_size <= 0 ||
        mc->ssm_conv_kernel <= 0 || mc->ssm_time_step_rank <= 0)
        return TN_ERR_INVALID_ARGS;
    MambaState *ms = (MambaState *)calloc(1, sizeof(*ms));
    if (!ms) return TN_ERR_OOM;
    ms->n_layers = n_layers;
    ms->d_inner  = mc->ssm_inner_size;
    ms->d_state  = mc->ssm_state_size;
    ms->d_conv   = mc->ssm_conv_kernel;
    ms->dt_rank  = mc->ssm_time_step_rank;
    size_t conv_n, ssm_n;
    if (tn_size_mul3((size_t)n_layers, (size_t)ms->d_inner,
                     (size_t)(ms->d_conv > 1 ? ms->d_conv - 1 : 1), &conv_n) ||
        tn_size_mul3((size_t)n_layers, (size_t)ms->d_inner, (size_t)ms->d_state, &ssm_n)) {
        free(ms);
        return TN_ERR_OOM;
    }
    ms->conv_state = (float *)tn_aligned_calloc(conv_n, sizeof(float), TN_SIMD_ALIGN);
    ms->ssm_state  = (float *)tn_aligned_calloc(ssm_n, sizeof(float), TN_SIMD_ALIGN);
    ms->xz  = (float *)tn_aligned_calloc((size_t)ms->d_inner * 2, sizeof(float), TN_SIMD_ALIGN);
    ms->xc  = (float *)tn_aligned_calloc((size_t)ms->d_inner, sizeof(float), TN_SIMD_ALIGN);
    ms->xdb = (float *)tn_aligned_calloc((size_t)ms->dt_rank + 2u * (size_t)ms->d_state,
                                         sizeof(float), TN_SIMD_ALIGN);
    ms->dt  = (float *)tn_aligned_calloc((size_t)ms->d_inner, sizeof(float), TN_SIMD_ALIGN);
    ms->y   = (float *)tn_aligned_calloc((size_t)ms->d_inner, sizeof(float), TN_SIMD_ALIGN);
    if (!ms->conv_state || !ms->ssm_state || !ms->xz || !ms->xc || !ms->xdb || !ms->dt || !ms->y) {
        mamba_state_free(ms);
        return TN_ERR_OOM;
    }
    *out = ms;
    return TN_OK;
}

void mamba_state_free(MambaState *ms) {
    if (!ms) return;
    tn_aligned_free(ms->conv_state);
    tn_aligned_free(ms->ssm_state);
    tn_aligned_free(ms->xz);
    tn_aligned_free(ms->xc);
    tn_aligned_free(ms->xdb);
    tn_aligned_free(ms->dt);
    tn_aligned_free(ms->y);
    free(ms);
}

void mamba_state_reset(MambaState *ms) {
    if (!ms) return;
    int cw = ms->d_conv > 1 ? ms->d_conv - 1 : 1;
    memset(ms->conv_state, 0, (size_t)ms->n_layers * ms->d_inner * cw * sizeof(float));
    memset(ms->ssm_state, 0, (size_t)ms->n_layers * ms->d_inner * ms->d_state * sizeof(float));
}
