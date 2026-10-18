#include "transformer/mamba.h"
#include "transformer/dense_matmul_dispatch.h"
#include "math/simd_dispatch.h"
#include "memory/aligned_alloc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TernaryError mamba_weights_alloc(MambaWeights **out, int n_layers) {
    if (!out || n_layers <= 0) return TN_ERR_INVALID_ARGS;
    MambaWeights *mw = (MambaWeights *)calloc(1, sizeof(*mw));
    if (!mw) return TN_ERR_OOM;
    mw->n_layers = n_layers;
    size_t L = (size_t)n_layers;
    mw->norm = calloc(L, sizeof(float *));    mw->conv_w = calloc(L, sizeof(float *));
    mw->conv_b = calloc(L, sizeof(float *));  mw->dt_b = calloc(L, sizeof(float *));
    mw->A = calloc(L, sizeof(float *));       mw->D = calloc(L, sizeof(float *));
    mw->in_proj = calloc(L, sizeof(tn_i8 *)); mw->x_proj = calloc(L, sizeof(tn_i8 *));
    mw->dt_proj = calloc(L, sizeof(tn_i8 *)); mw->out_proj = calloc(L, sizeof(tn_i8 *));
    mw->in_type = calloc(L, sizeof(int));     mw->x_type = calloc(L, sizeof(int));
    mw->dt_type = calloc(L, sizeof(int));     mw->out_type = calloc(L, sizeof(int));
    if (!mw->norm || !mw->conv_w || !mw->conv_b || !mw->dt_b || !mw->A || !mw->D ||
        !mw->in_proj || !mw->x_proj || !mw->dt_proj || !mw->out_proj ||
        !mw->in_type || !mw->x_type || !mw->dt_type || !mw->out_type) {
        mamba_weights_free(mw);
        return TN_ERR_OOM;
    }
    *out = mw;
    return TN_OK;
}

void mamba_weights_free(MambaWeights *mw) {
    if (!mw) return;
    /* Tensor data is owned by the mmap / GGUF weight store, not by us. */
    free(mw->norm); free(mw->conv_w); free(mw->conv_b); free(mw->dt_b);
    free(mw->A); free(mw->D); free(mw->in_proj); free(mw->x_proj);
    free(mw->dt_proj); free(mw->out_proj); free(mw->in_type); free(mw->x_type);
    free(mw->dt_type); free(mw->out_type);
    free(mw);
}

TernaryError mamba_state_alloc(MambaState **out, int n_layers, const MoEConfig *mc) {
    if (!out || !mc || n_layers <= 0 || mc->ssm_inner_size <= 0 || mc->ssm_state_size <= 0 ||
        mc->ssm_conv_kernel <= 0 || mc->ssm_time_step_rank <= 0)
        return TN_ERR_INVALID_ARGS;
    MambaState *ms = (MambaState *)calloc(1, sizeof(*ms));
    if (!ms) return TN_ERR_OOM;
    ms->n_layers = n_layers;
    ms->d_inner = mc->ssm_inner_size;
    ms->d_state = mc->ssm_state_size;
    ms->d_conv = mc->ssm_conv_kernel;
    ms->dt_rank = mc->ssm_time_step_rank;
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
    tn_aligned_free(ms->conv_state); tn_aligned_free(ms->ssm_state);
    tn_aligned_free(ms->xz); tn_aligned_free(ms->xc); tn_aligned_free(ms->xdb);
    tn_aligned_free(ms->dt); tn_aligned_free(ms->y);
    free(ms);
}

void mamba_state_reset(MambaState *ms) {
    if (!ms) return;
    int cw = ms->d_conv > 1 ? ms->d_conv - 1 : 1;
    memset(ms->conv_state, 0, (size_t)ms->n_layers * ms->d_inner * cw * sizeof(float));
    memset(ms->ssm_state, 0, (size_t)ms->n_layers * ms->d_inner * ms->d_state * sizeof(float));
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }
static inline float softplus(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

void mamba_block_forward(float *x, float *xb, float *xb2, MambaState *ms,
                         const MambaWeights *mw, const Config *cfg, const MoEConfig *mc,
                         int layer, int pos, ThreadPool *tp) {
    (void)mc;
    const int dim = cfg->dim, di = ms->d_inner, ds = ms->d_state;
    const int dc = ms->d_conv, dr = ms->dt_rank, cw = dc > 1 ? dc - 1 : 1;
    /* Each layer resets its own state at position 0 (2026-10-18: was "layer 0
     * resets all", which a distributed worker running layers a..b, a > 0,
     * would never trigger). */
    if (pos == 0) {
        memset(ms->conv_state + (size_t)layer * di * cw, 0, (size_t)di * cw * sizeof(float));
        memset(ms->ssm_state + (size_t)layer * di * ds, 0, (size_t)di * ds * sizeof(float));
    }

    /* norm + in-projection -> [u | z] */
    tn_rmsnorm(xb, x, mw->norm[layer], dim, cfg->rms_norm_eps);
    tn_dense_matmul_dispatch(ms->xz, xb, mw->in_proj[layer], mw->in_type[layer], dim, 2 * di, tp);
    const float *u = ms->xz, *z = ms->xz + di;

    /* depthwise causal conv1d with rolling state, then SiLU */
    float *cst = ms->conv_state + (size_t)layer * di * cw;
    const float *cwgt = mw->conv_w[layer], *cb = mw->conv_b[layer];
    for (int c = 0; c < di; c++) {
        float *st = cst + (size_t)c * cw;
        const float *wc = cwgt + (size_t)c * dc;
        float acc = cb ? cb[c] : 0.0f;
        for (int k = 0; k < dc - 1; k++) acc += wc[k] * st[k];
        acc += wc[dc - 1] * u[c];
        for (int k = 0; k < dc - 2; k++) st[k] = st[k + 1];
        if (dc > 1) st[dc - 2] = u[c];
        ms->xc[c] = silu(acc);
    }

    /* x-projection -> [dt_r | B | C]; dt = softplus(W_dt dt_r + b) */
    tn_dense_matmul_dispatch(ms->xdb, ms->xc, mw->x_proj[layer], mw->x_type[layer], di, dr + 2 * ds, tp);
    tn_dense_matmul_dispatch(ms->dt, ms->xdb, mw->dt_proj[layer], mw->dt_type[layer], dr, di, tp);
    const float *B = ms->xdb + dr, *C = ms->xdb + dr + ds;
    const float *A = mw->A[layer], *Dv = mw->D[layer], *dtb = mw->dt_b[layer];

    /* selective scan (one step), gated by SiLU(z) */
    float *sst = ms->ssm_state + (size_t)layer * di * ds;
    for (int c = 0; c < di; c++) {
        float dt = softplus(ms->dt[c] + (dtb ? dtb[c] : 0.0f));
        float xu = ms->xc[c];
        float *s = sst + (size_t)c * ds;
        const float *a = A + (size_t)c * ds;
        float yc = 0.0f;
        for (int n = 0; n < ds; n++) {
            s[n] = s[n] * expf(dt * a[n]) + dt * B[n] * xu;
            yc += s[n] * C[n];
        }
        yc += Dv[c] * xu;
        ms->y[c] = yc * silu(z[c]);
    }

    tn_dense_matmul_dispatch(xb2, ms->y, mw->out_proj[layer], mw->out_type[layer], di, dim, tp);
    tn_vec_add(x, x, xb2, dim);
}
