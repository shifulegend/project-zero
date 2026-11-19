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

#include "core/mamba_state.h" /* MambaWeights, MambaState and their alloc/free */

/*
 * One Mamba block, in place on x[dim] (residual included). `xb`/`xb2` are
 * dim-sized scratch. Resets this layer's recurrent state when pos == 0.
 */
void mamba_block_forward(float *x, float *xb, float *xb2, MambaState *ms, const MambaWeights *mw,
                         const Config *cfg, const MoEConfig *mc, int layer, int pos,
                         ThreadPool *tp);

#endif /* TN_MAMBA_H */
