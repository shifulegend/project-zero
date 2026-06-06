#ifndef TN_TERNARY_MATMUL_H
#define TN_TERNARY_MATMUL_H

#include "core/platform.h"

/**
 * Ternary matrix-vector multiplication (scalar reference).
 *
 * out[i] = scale * sum_j( x[j] * w[i*n + j] )  where w ∈ {-1, 0, 1}
 *
 * The multiply-by-weight is replaced with conditional add/subtract/skip.
 *
 * @param out   Output vector of size d (must be pre-allocated)
 * @param x     Input vector of size n
 * @param w     Weight matrix in row-major order, size d * n, values in {-1, 0, 1}
 * @param n     Input dimension (columns)
 * @param d     Output dimension (rows)
 * @param scale Per-matrix scale factor to restore decimal range
 */
void ternary_matmul(float *out, const float *x, const tn_i8 *w,
                    int n, int d, float scale);

#endif /* TN_TERNARY_MATMUL_H */
