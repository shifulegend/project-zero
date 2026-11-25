#ifndef TN_SIMD_PREQ_OPS_H
#define TN_SIMD_PREQ_OPS_H

/* SIMD pre-quantised ternary ops (AVX-512 VNNI tier only).
 * Part of the SIMD dispatch table (math/simd_dispatch.h includes all of it; include this
 * header alone when only these kernels are used). Bound by tn_simd_init(). */

#include <stdint.h>

#include "core/platform.h"

/* Pre-quantised ternary path (AVX-512 VNNI tier only, parallel_matmul.c): the
 * activation quantiser, its int8 sum, and the matmul that consumes int8
 * activations. NULL unless tn_simd_init bound the AVX-512 VNNI tier. */
typedef float (*tn_quantize_i8_fn)(const float *x, int8_t *q, int n);
typedef int32_t (*tn_sum_i8_fn)(const int8_t *arr, int n);
typedef void (*tn_matmul_preq_fn)(float *out, const int8_t *q_x, float act_scale, int32_t sum_qx,
                                  const tn_u8 *packed_w, int n, int d, const float *scales,
                                  int group_size);

/* Dispatch pointers */
extern tn_quantize_i8_fn tn_quantize_act_i8;
extern tn_sum_i8_fn tn_sum_act_i8;
extern tn_matmul_preq_fn tn_ternary_matmul_preq;

/* 1 when the last tn_simd_init() bound the AVX-512 VNNI ternary kernel, the one
 * tier that has a pre-quantised-activation entry (parallel_matmul.c): then
 * tn_quantize_act_i8 / tn_sum_act_i8 / tn_ternary_matmul_preq are all non-NULL.
 * Every other tier, whether forced (TN_FORCE_BACKEND, --simd, calibration) or simply
 * lower, returns 0 so the matmul runs the kernel the tier names (2026-11-18: a forced
 * tier used to change nothing for the ternary matmul on a VNNI native build, and the
 * portable build had no pre-quantised path at all; tests/test_simd_tier_kernel.c).
 * The tier kernels live in ISA-flagged translation units and are referenced only
 * from simd_dispatch.c, so baseline files such as parallel_matmul.c carry no ISA
 * guards (tools/check_isa_guards.py). */
int tn_simd_ternary_preq(void);

#endif /* TN_SIMD_PREQ_OPS_H */
