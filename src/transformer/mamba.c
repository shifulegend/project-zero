#include "transformer/mamba.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "math/simd_dispatch.h"
#include "memory/aligned_alloc.h"
#include "transformer/dense_matmul_dispatch.h"

static inline float silu(float v) {
    return v / (1.0f + expf(-v));
}
static inline float softplus(float v) {
    return v > 20.0f ? v : log1pf(expf(v));
}

void mamba_block_forward(float *x, float *xb, float *xb2, MambaState *ms, const MambaWeights *mw,
                         const Config *cfg, const MoEConfig *mc, int layer, int pos,
                         ThreadPool *tp) {
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
    float *cst        = ms->conv_state + (size_t)layer * di * cw;
    const float *cwgt = mw->conv_w[layer], *cb = mw->conv_b[layer];
    for (int c = 0; c < di; c++) {
        float *st       = cst + (size_t)c * cw;
        const float *wc = cwgt + (size_t)c * dc;
        float acc       = cb ? cb[c] : 0.0f;
        for (int k = 0; k < dc - 1; k++) acc += wc[k] * st[k];
        acc += wc[dc - 1] * u[c];
        for (int k = 0; k < dc - 2; k++) st[k] = st[k + 1];
        if (dc > 1) st[dc - 2] = u[c];
        ms->xc[c] = silu(acc);
    }

    /* x-projection -> [dt_r | B | C]; dt = softplus(W_dt dt_r + b) */
    tn_dense_matmul_dispatch(ms->xdb, ms->xc, mw->x_proj[layer], mw->x_type[layer], di, dr + 2 * ds,
                             tp);
    tn_dense_matmul_dispatch(ms->dt, ms->xdb, mw->dt_proj[layer], mw->dt_type[layer], dr, di, tp);
    const float *B = ms->xdb + dr, *C = ms->xdb + dr + ds;
    const float *A = mw->A[layer], *Dv = mw->D[layer], *dtb = mw->dt_b[layer];

    /* selective scan (one step), gated by SiLU(z) */
    float *sst = ms->ssm_state + (size_t)layer * di * ds;
    for (int c = 0; c < di; c++) {
        float dt       = softplus(ms->dt[c] + (dtb ? dtb[c] : 0.0f));
        float xu       = ms->xc[c];
        float *s       = sst + (size_t)c * ds;
        const float *a = A + (size_t)c * ds;
        float yc       = 0.0f;
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
