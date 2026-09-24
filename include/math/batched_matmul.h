#ifndef TN_BATCHED_MATMUL_H
#define TN_BATCHED_MATMUL_H

#include "core/platform.h"
#include "threading/thread_pool.h"

/**
 * Batched matrix-vector multiplication: N candidate token activations
 * against the same weight matrix in one call, instead of N separate
 * single-vector matmul calls.
 *
 * Phase 18 (speculative decoding) motivation: this engine's inference cost
 * is dominated by streaming weight matrices from RAM (memory-bandwidth
 * bound, not compute bound). A naive N sequential single-vector matmul
 * calls re-reads every weight row from RAM N times. These batched variants
 * read each weight row once and reuse it (resident in L1/L2 cache, since a
 * single row is small relative to cache size) across all N candidate
 * activations -- RAM traffic for the matrix becomes O(matrix_bytes) instead
 * of O(matrix_bytes * n_tokens), which is what makes speculative decoding's
 * verify step actually cheaper than N normal forward-pass steps.
 *
 * Layout convention: `x` is [n_tokens][n] row-major (token k's activation
 * vector is x + k*n); `out` is [n_tokens][d] row-major (token k's output
 * row is out + k*d).
 *
 * v1 scope (2026-09-22): ternary-packed and F16 dense matmul, plus the
 * classifier formats (BF16/INT8/INT4). Q4_K/Q4_K_x8/Q2_0/F32 weight formats
 * are NOT covered (callers fall back to n_tokens sequential single-vector
 * calls for those, via tn_dense_matmul_dispatch_batch() in
 * dense_matmul_dispatch.h).
 *
 * 2026-09-24: each of these kernels now dequantizes/unpacks a weight row to
 * F32 ONCE, then reuses this project's existing SIMD-dispatched tn_vec_dot()
 * (math/simd_dispatch.h) for every one of the n_tokens dot products against
 * that row -- fixing a real regression the v1 scalar-accumulator
 * implementation had (RAM-bandwidth savings were real, but scalar per-token
 * FMA made the batched call slower in wall-clock terms than N sequential
 * SIMD-accelerated single-token calls on the same hardware; see
 * docs/ai/mistakes.md's 2026-09-24 entry for the real-model measurement that
 * caught it). Full ISA-tuned dispatch tiers matching every per-vector
 * kernel's own dispatch surface (ternary_matmul_packed.h's 9-way
 * scalar/AVX2/AVX-512/VNNI/dotprod/NEON tiers) are still a documented
 * follow-up -- tn_vec_dot() itself already SIMD-dispatches, so this fix
 * closes the actual perf gap without needing every tier replicated here.
 */

/**
 * Ternary-packed batched matmul (per-matrix scale, group_size=0 -- matches
 * parallel_ternary_matmul_packed()'s scope, the only mode the real forward
 * pass uses).
 */
void tn_ternary_matmul_packed_batch(float *out, const float *x, const tn_u8 *packed_w,
                                     int n, int d, float scale, int n_tokens, ThreadPool *tp);

/**
 * F16 dense batched matmul.
 */
void tn_matmul_f16_batch(float *out, const float *x, const tn_u16 *w,
                          int n, int d, int n_tokens, ThreadPool *tp);

/**
 * BF16 dense batched matmul (classifier format).
 */
void tn_matmul_bf16_batch(float *out, const float *x, const tn_u16 *w,
                           int n, int d, int n_tokens, ThreadPool *tp);

/**
 * INT8 block-quantized batched matmul (classifier format). `scales` is
 * d * n_blocks floats, TN_CLS_QUANT_BLOCK-element blocks (see weights.h).
 */
void tn_matmul_i8_batch(float *out, const float *x, const tn_u8 *w,
                         const float *scales, int n, int d, int n_tokens, ThreadPool *tp);

/**
 * INT4 packed, block-quantized batched matmul (classifier format).
 */
void tn_matmul_i4_batch(float *out, const float *x, const tn_u8 *w,
                         const float *scales, int n, int d, int n_tokens, ThreadPool *tp);

#endif /* TN_BATCHED_MATMUL_H */
