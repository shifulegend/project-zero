#ifndef TN_MATMUL_F16_H
#define TN_MATMUL_F16_H

/*
 * F16 (IEEE 754 half-precision) layer weight matmul.
 *
 * Used for GGUF models whose layer weights are stored as F16 on disk
 * (e.g. SmolLM2-135M-Instruct).  Weights are kept as raw F16 (tn_u16 *)
 * — no heap copy, no F32 expansion at load time.  Each element is
 * converted from F16 to F32 on-the-fly during the dot product.
 *
 * Bandwidth: 2 bytes/weight (vs 4 bytes for F32) — matches llama.cpp.
 *
 * parallel_matmul_f16(out, x, w, n, d, tp):
 *   Computes out[i] = dot(x[0..n-1], w[i*n .. i*n+n-1])  for i in [0,d)
 *   x: float32 activation vector (n elements)
 *   w: F16 weight matrix (d rows × n columns, row-major)
 *   tp: thread pool (NULL = single-threaded)
 */

#include "core/platform.h"
#include "threading/thread_pool.h"

void parallel_matmul_f16(float *out, const float *x, const tn_u16 *w,
                          int n, int d, ThreadPool *tp);

#endif /* TN_MATMUL_F16_H */
