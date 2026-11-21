#ifndef TN_CORE_MAMBA_STATE_H
#define TN_CORE_MAMBA_STATE_H

/* Mamba (selective state-space) weight tables and recurrent state: the types and their
 * allocation. The block's forward pass is in transformer/mamba.h. */

#include "core/error.h"
#include "core/moe_config.h"
#include "core/platform.h"

typedef struct MambaWeights {
    int n_layers;
    float **norm; /* [L] dim, F32 */
    tn_i8 **in_proj;
    int *in_type;   /* dim -> 2*d_inner */
    float **conv_w; /* [L] d_inner * d_conv, F32 (taps contiguous per channel) */
    float **conv_b; /* [L] d_inner, F32 */
    tn_i8 **x_proj;
    int *x_type; /* d_inner -> dt_rank + 2*d_state */
    tn_i8 **dt_proj;
    int *dt_type; /* dt_rank -> d_inner */
    float **dt_b; /* [L] d_inner, F32 */
    float **A;    /* [L] d_inner * d_state, F32 (already -exp(A_log)) */
    float **D;    /* [L] d_inner, F32 */
    tn_i8 **out_proj;
    int *out_type; /* d_inner -> dim */
} MambaWeights;

typedef struct MambaState {
    int n_layers, d_inner, d_state, d_conv, dt_rank;
    float *conv_state;             /* [L][d_inner][d_conv-1] */
    float *ssm_state;              /* [L][d_inner][d_state]  */
    float *xz, *xc, *xdb, *dt, *y; /* per-token scratch */
} MambaState;

/* Allocates zeroed per-layer pointer arrays (tensors are filled by the GGUF loader). */
TernaryError mamba_weights_alloc(MambaWeights **out, int n_layers);
void mamba_weights_free(MambaWeights *mw);

TernaryError mamba_state_alloc(MambaState **out, int n_layers, const MoEConfig *mc);
void mamba_state_free(MambaState *ms);
void mamba_state_reset(MambaState *ms);

#endif /* TN_CORE_MAMBA_STATE_H */
