#ifndef TN_MAMBA_H
#define TN_MAMBA_H

/*
 * mamba.h — Phase 22-SSM: Mamba (selective state-space) blocks.
 *
 * GGUF arch "mamba" (llama.cpp naming: ssm_in / ssm_conv1d / ssm_x / ssm_dt /
 * ssm_a / ssm_d / ssm_out, attn_norm, output_norm; tied embeddings when no
 * output.weight). Each block:
 *   h = RMSNorm(x);  [u | z] = W_in h
 *   u = SiLU(causal_conv1d(u))          (per-channel, kernel d_conv, rolling state)
 *   [dt_r | B | C] = W_x u;  dt = softplus(W_dt dt_r + b_dt)
 *   s = s * exp(dt * A) + dt * B * u;   y = s·C + D * u     (per channel, d_state)
 *   x += W_out (y * SiLU(z))
 * Shapes and hyper-parameters come from GGUF metadata (mamba.ssm.*), never
 * literals. Projections reuse tn_dense_matmul_dispatch, so every weight
 * format the dense path supports works here too.
 *
 * Recurrent: state lives in RunState.mamba and is reset when position 0 is
 * processed. Features that rewind or batch positions (prefix cache, prompt
 * store, speculative decoding) are disabled via moe_is_recurrent().
 */

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include "threading/thread_pool.h"

typedef struct MambaWeights {
    int     n_layers;
    float **norm;      /* [L] dim, F32 */
    tn_i8 **in_proj;   int *in_type;   /* dim -> 2*d_inner */
    float **conv_w;    /* [L] d_inner * d_conv, F32 (taps contiguous per channel) */
    float **conv_b;    /* [L] d_inner, F32 */
    tn_i8 **x_proj;    int *x_type;    /* d_inner -> dt_rank + 2*d_state */
    tn_i8 **dt_proj;   int *dt_type;   /* dt_rank -> d_inner */
    float **dt_b;      /* [L] d_inner, F32 */
    float **A;         /* [L] d_inner * d_state, F32 (already -exp(A_log)) */
    float **D;         /* [L] d_inner, F32 */
    tn_i8 **out_proj;  int *out_type;  /* d_inner -> dim */
} MambaWeights;

typedef struct MambaState {
    int    n_layers, d_inner, d_state, d_conv, dt_rank;
    float *conv_state; /* [L][d_inner][d_conv-1] */
    float *ssm_state;  /* [L][d_inner][d_state]  */
    float *xz, *xc, *xdb, *dt, *y;  /* per-token scratch */
} MambaState;

/* Allocates zeroed per-layer pointer arrays (tensors are filled by the GGUF loader). */
TernaryError mamba_weights_alloc(MambaWeights **out, int n_layers);
void         mamba_weights_free(MambaWeights *mw);

TernaryError mamba_state_alloc(MambaState **out, int n_layers, const MoEConfig *mc);
void         mamba_state_free(MambaState *ms);
void         mamba_state_reset(MambaState *ms);

/*
 * One Mamba block, in place on x[dim] (residual included). `xb`/`xb2` are
 * dim-sized scratch. Resets this layer's recurrent state when pos == 0.
 */
void mamba_block_forward(float *x, float *xb, float *xb2, MambaState *ms,
                         const MambaWeights *mw, const Config *cfg, const MoEConfig *mc,
                         int layer, int pos, ThreadPool *tp);

#endif /* TN_MAMBA_H */
