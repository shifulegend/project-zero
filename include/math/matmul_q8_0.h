#ifndef TN_MATH_MATMUL_Q8_0_H
#define TN_MATH_MATMUL_Q8_0_H
/*
 * parallel_matmul_q8_0 — fused Q8_0 × F32 activation matmul (gemv).
 *
 * Q8_0 block layout (34 bytes / 32 elements):
 *   [d:  fp16   2B] — scale
 *   [qs: int8[32] ] — 32 signed 8-bit integers
 *
 * Element i: (float)qs[i] * d
 * Dot product: d * sum(qs[i] * x[i])
 */

#include <stdint.h>

#include "threading/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

void parallel_matmul_q8_0(float *out, const float *x, const uint8_t *w_q8_0, int n, int d,
                          ThreadPool *tp);

/*
 * parallel_matmul_q8_0_batch — batched Q8_0 × F32 GEMV for k weight matrices.
 * Per-expert inputs: each xs[i] is the post-SiLU*up activation for expert i.
 */
void parallel_matmul_q8_0_batch(float **outs, float **xs, const uint8_t *const *ws, int n, int d,
                                int k, ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* TN_MATH_MATMUL_Q8_0_H */
