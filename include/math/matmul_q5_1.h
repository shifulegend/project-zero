#pragma once
/*
 * parallel_matmul_q5_1 — fused Q5_1 × F32 activation matmul (gemv).
 *
 * Q5_1 block layout (24 bytes / 32 elements):
 *   [d:  fp16  2B] — scale
 *   [m:  fp16  2B] — min
 *   [qh: u32   4B] — 32 high bits, one per element (5th bit)
 *   [qs: u8[16]  ] — 32 × 4-bit low nibbles
 *
 * Element i (0..15): (qs[i] & 0xF) | (((qh >> i) & 1) << 4)
 * Element i+16:      (qs[i] >> 4)  | (((qh >> (i+16)) & 1) << 4)
 * Value = int_val * d + m
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fused Q5_1 matmul: out[d] = W_q51 × x[n]
 *   w_q51 : row-major Q5_1 matrix, d rows of (n/32)*24 bytes each
 *   n     : input dim — must be divisible by Q5_1 block size (32)
 *   d     : output dim
 *   tp    : thread pool (may be NULL for single-threaded)
 */
void parallel_matmul_q5_1(float *out, const float *x, const uint8_t *w_q51,
                           int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q5_1_batch — batched Q5_1 × F32 GEMV for k weight matrices.
 *
 * Unlike the Q4K batch (shared input), each expert has its own input xs[i]
 * (the post-SiLU activation after gate×up fusion).
 *
 *   outs:  float*[k]          — k output arrays, each d floats
 *   xs:    float*[k]          — k input arrays, each n floats (per-expert)
 *   ws:    uint8_t*[k]        — k Q5_1 weight arrays
 *   n:     inner dim (multiple of 32)
 *   d:     output rows per expert
 *   k:     number of experts
 *   tp:    thread pool (may be NULL)
 */
void parallel_matmul_q5_1_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
