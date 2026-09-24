#ifndef TN_LORA_MATMUL_H
#define TN_LORA_MATMUL_H

#include "core/lora.h"
#include "threading/thread_pool.h"

/**
 * Adds a LoRA module's low-rank correction to an already-computed base
 * projection output, in place: out[i] += mod->scale * dot(B_row_i,
 * A @ x) for i in [0,d). A no-op (returns immediately) when
 * `mod == NULL || mod->rank == 0` -- the "LoRA disabled for this
 * (layer, module)" sentinel (see include/core/lora.h).
 *
 * `x` is the same n-wide input activation the base matmul that produced
 * `out` was given; `out` must already hold that base result -- this
 * function only ever adds to it, never overwrites it.
 *
 * Deliberately format-agnostic: A/B are always F32 (decoded once at load
 * time, include/core/lora.h), so this one function works regardless of
 * whether the base weight matrix is ternary/Q4_K/F16/Q2_0/F32 -- unlike a
 * fused per-format kernel, it never needs to know.
 */
void lora_apply(float *out, const float *x, const LoRAModule *mod,
                 int n, int d, ThreadPool *tp);

#endif /* TN_LORA_MATMUL_H */
