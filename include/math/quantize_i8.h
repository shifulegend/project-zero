#ifndef TN_QUANTIZE_I8_H
#define TN_QUANTIZE_I8_H

#include <stdint.h>

#include "core/platform.h"

/*
 * Activation quantization for the int8 kernels (VNNI, AVX-VNNI, dotprod, Q2_0).
 *
 * SPECIFICATION (2026-11-18). Every implementation below computes exactly this, bit for bit:
 *
 *   abs_max   = max_i |x[i]|                       (float)
 *   abs_max == 0  ->  q[i] = 0 for all i, scale = 0
 *   scale     = abs_max / 127.0f                   (returned; true_value ~ q * scale)
 *   inv_scale = 127.0f / abs_max
 *   q[i]      = clamp( round_to_nearest_even( x[i] * inv_scale ), -127, +127 )
 *
 * The product x[i]*inv_scale is evaluated in float, the clamp is applied before rounding, and
 * -128 is never produced (symmetric range, products stay representable in int16).
 *
 * Rounding is NEAREST, ties to even (the default IEEE mode: cvtps2dq, vcvtns, lrintf). It is not
 * truncation: truncating shrinks every activation towards zero by half a step on average, a bias
 * correlated with the sign of the activation that does not average out across a 2560-term dot
 * product. Measured on BitNet b1.58 2B (docs/rca/2026-11-18-ternary-output-correctness.md): the V
 * projection of layer 0 had 3.7 % relative error instead of 0.4 %, and perplexity rose 9.7 %.
 * The maximum round-trip error is scale/2.
 *
 * The scalar function is the oracle. tests/test_quantize_i8_exact.c requires every tier to equal
 * it exactly, and tools/check_rounding.py fails CI on a truncating convert in these files.
 *
 * Usage pattern (called once per matmul, before the row loop):
 *   int8_t  q_x[n];
 *   float   act_scale = quantize_row_to_i8(x, q_x, n);
 *   int32_t sum_qx    = sum_i8(q_x, n);   // for VNNI bias correction
 *   ... VNNI inner loop uses q_x ...
 *   out[i] = dot_int32 * act_scale * weight_scale[i];
 */

/*
 * Scalar reference (the oracle): quantize float32 vector x[n] -> int8 q[n] per the specification.
 * Returns the per-vector scale (abs_max / 127), 0 for an all-zero input.
 */
float quantize_row_to_i8(const float *x, int8_t *q, int n);

/*
 * AVX-512 accelerated version, same specification, faster on AVX-512 CPUs.
 * Falls through to scalar on non-AVX-512 builds.
 */
float quantize_row_to_i8_avx512(const float *x, int8_t *q, int n);

/*
 * AVX2 accelerated version.
 */
float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n);

/*
 * NEON (AArch64) version of the same specification. On builds without NEON it is the scalar
 * function, so callers never need a guard.
 */
float quantize_row_to_i8_neon(const float *x, int8_t *q, int n);

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
