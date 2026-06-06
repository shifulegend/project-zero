#pragma once
/*
 * parallel_matmul_q5k — fused Q5_K × F32 activation matmul (gemv).
 *
 * Q5_K super-block layout (176 bytes / 256 elements):
 *   [d: fp16 2B] [dmin: fp16 2B] [sc12: 12B] [qh: 32B] [qs: 128B]
 *
 * Same 8-subblock scale/min encoding as Q4_K, plus one 5th bit per element.
 * qh[l] (l=0..31) stores the 5th bits across all 4 groups:
 *   bit (2g)   of qh[l] = 5th bit of element g*64 + l       (low  nibble)
 *   bit (2g+1) of qh[l] = 5th bit of element g*64 + l + 32  (high nibble)
 *
 * For group g (0..3), the u1/u2 masks: u1 = 1<<(2g), u2 = 1<<(2g+1)
 * Element value = int_val * scale - min  (same formula as Q4_K)
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fused Q5_K matmul: out[d] = W_q5k × x[n]
 *   w_q5k : row-major Q5_K matrix, d rows of (n/256)*176 bytes each
 *   n     : input dim — must be divisible by 256
 *   d     : output dim
 *   tp    : thread pool (may be NULL for single-threaded)
 */
void parallel_matmul_q5k(float *out, const float *x, const uint8_t *w_q5k,
                          int n, int d, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
