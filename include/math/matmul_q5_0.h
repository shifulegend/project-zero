#pragma once
/*
 * parallel_matmul_q5_0 — fused Q5_0 × F32 activation matmul (gemv).
 *
 * Q5_0 block layout (22 bytes / 32 elements):
 *   [d:  fp16  2B] — scale
 *   [qh: u32   4B] — 32 high bits, one per element (5th bit)
 *   [qs: u8[16]  ] — 32 × 4-bit low nibbles
 *
 * Element i (0..15):  (qs[i] & 0xF) | (((qh >> i) & 1) << 4)  → unsigned 0..31
 * Element i+16:       (qs[i] >> 4)  | (((qh >> (i+16)) & 1) << 4)
 * Value = (int_val - 16) * d      (signed, no stored min)
 *
 * Dot product: d * sum(int_val * x) - 16*d * sum(x)
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void parallel_matmul_q5_0(float *out, const float *x, const uint8_t *w_q5_0,
                           int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q5_0_batch — batched Q5_0 × F32 GEMV for k weight matrices.
 * Per-expert inputs: each xs[i] is the post-SiLU*up activation for expert i.
 */
void parallel_matmul_q5_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
