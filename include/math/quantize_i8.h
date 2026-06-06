#ifndef TN_QUANTIZE_I8_H
#define TN_QUANTIZE_I8_H

#include "core/platform.h"
#include <stdint.h>

/*
 * Activation quantization for VNNI / dotprod kernels (Phase 16-S.2)
 *
 * Dynamic per-vector quantization: find the absolute max of a float32 vector,
 * then scale all values to fit in int8 [-127, 127] (symmetric, avoiding -128
 * to keep multiplication results representable in int16).
 *
 * The returned scale factor dequantizes: true_value ≈ q_value * scale.
 *
 * Usage pattern (called once per matmul, before the row loop):
 *   int8_t  q_x[n];
 *   float   act_scale = quantize_row_to_i8(x, q_x, n);
 *   int32_t sum_qx    = sum_i8(q_x, n);   // for VNNI bias correction
 *   ... VNNI inner loop uses q_x ...
 *   out[i] = dot_int32 * act_scale * weight_scale[i];
 */

/*
 * Scalar reference: quantize float32 vector x[n] → int8 q[n].
 * Returns the per-vector scale factor (abs_max / 127).
 * If abs_max == 0 (all-zero input), scale = 0 and q is all zeros.
 */
float quantize_row_to_i8(const float *x, int8_t *q, int n);

/*
 * AVX-512 accelerated version — same semantics, faster on AVX-512 CPUs.
 * Falls through to scalar on non-AVX-512 builds.
 */
float quantize_row_to_i8_avx512(const float *x, int8_t *q, int n);

/*
 * AVX2 accelerated version.
 */
float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n);

/*
 * Sum all int8 values in arr[n].  Used for VNNI bias correction:
 *   true_dot = vnni_result - sum_qx
 * because dpbusds computes sum((w+1)*q_x) = dot(w,q_x) + sum(q_x).
 */
int32_t sum_i8(const int8_t *arr, int n);

/*
 * AVX-512 horizontal int8 sum.
 */
int32_t sum_i8_avx512(const int8_t *arr, int n);

#endif /* TN_QUANTIZE_I8_H */
