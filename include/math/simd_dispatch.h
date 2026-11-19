#ifndef TN_SIMD_DISPATCH_H
#define TN_SIMD_DISPATCH_H

#include "core/platform.h"
#include "math/cls_kernels.h"
#include "math/kvq4_kernels.h"

/**
 * SIMD runtime dispatch layer.
 *
 * Provides function pointers that are initialized at startup to the best
 * available SIMD implementation (AVX2 > scalar fallback).
 *
 * Call tn_simd_init() once at program startup before using any math ops.
 */

/* Ternary matmul function pointer type */
typedef void (*tn_matmul_fn)(float *out, const float *x, const tn_i8 *w, int n, int d, float scale);

/* RMSNorm function pointer type */
typedef void (*tn_rmsnorm_fn)(float *out, const float *x, const float *weight, int size, float eps);

/* Softmax function pointer type */
typedef void (*tn_softmax_fn)(float *x, int size);

/* Packed ternary matmul function pointer type (Phase 10) */
typedef void (*tn_matmul_packed_fn)(float *out, const float *x, const tn_u8 *packed_w, int n, int d,
                                    const float *scales, int group_size);

/* Ternary block unpacker function pointer type (Phase 10) */
typedef void (*tn_unpack_fn)(tn_i8 *out, const tn_u8 *packed, int count);

/* Element-wise function pointer types */
typedef void (*tn_vec_add_fn)(float *out, const float *a, const float *b, int n);
typedef void (*tn_vec_mul_fn)(float *out, const float *a, const float *b, int n);
typedef void (*tn_vec_scale_fn)(float *x, float s, int n);
typedef void (*tn_silu_fn)(float *x, int n);
typedef void (*tn_relu2_fn)(float *x, int n);
typedef float (*tn_vec_dot_fn)(const float *a, const float *b, int n);
/* SAXPY: out[i] += scale * v[i]  — fused scale+accumulate for attention values */
typedef void (*tn_vec_saxpy_fn)(float *out, float scale, const float *v, int n);

/* Pre-quantised ternary path (AVX-512 VNNI tier only, parallel_matmul.c): the
 * activation quantiser, its int8 sum, and the matmul that consumes int8
 * activations. NULL unless tn_simd_init bound the AVX-512 VNNI tier. */
typedef float (*tn_quantize_i8_fn)(const float *x, int8_t *q, int n);
typedef int32_t (*tn_sum_i8_fn)(const int8_t *arr, int n);
typedef void (*tn_matmul_preq_fn)(float *out, const int8_t *q_x, float act_scale, int32_t sum_qx,
                                  const tn_u8 *packed_w, int n, int d, const float *scales,
                                  int group_size);

/* Global dispatch table */
extern tn_matmul_fn tn_ternary_matmul;
extern tn_matmul_packed_fn tn_ternary_matmul_packed;
extern tn_unpack_fn tn_unpack_block;
extern tn_rmsnorm_fn tn_rmsnorm;
extern tn_softmax_fn tn_softmax;
extern tn_vec_add_fn tn_vec_add;
extern tn_vec_mul_fn tn_vec_mul;
extern tn_vec_scale_fn tn_vec_scale;
extern tn_silu_fn tn_silu;
extern tn_relu2_fn tn_relu2;
extern tn_vec_dot_fn tn_vec_dot;
/* int4 KV-cache row kernels (math/kvq4_kernels.h); default to scalar before
 * tn_simd_init() so callers that never init still get correct results. */
extern tn_kvq4_dot_fn tn_kvq4_dot;
extern tn_kvq4_axpy_fn tn_kvq4_axpy;
extern tn_vec_saxpy_fn tn_vec_saxpy;
extern tn_quantize_i8_fn tn_quantize_act_i8;
extern tn_sum_i8_fn tn_sum_act_i8;
extern tn_matmul_preq_fn tn_ternary_matmul_preq;
/* INT8 / INT4 classifier (LM head) row kernels (math/cls_kernels.h). Generic (AVX2 or scalar) before
 * tn_simd_init() and on lower or forced tiers; the AVX-512 version on AVX-512F CPUs. */
extern tn_cls_rows_fn tn_cls_i8_rows;
extern tn_cls_rows_fn tn_cls_i4_rows;

/**
 * Initialize SIMD dispatch. Probes CPU features and sets function pointers
 * to the best available implementation. Must be called before any math ops.
 *
 * Returns a human-readable string describing the selected backend
 * (e.g., "AVX2", "Scalar").
 *
 * TODO(QA-ISS-005): SIMD availability is currently determined at compile time
 * via preprocessor flags (#ifdef __AVX2__, etc.). On machines without the
 * required ISA but compiled with those flags, this will crash with SIGILL.
 * A future version should add runtime CPUID detection to select the correct
 * kernel at startup, independent of compile-time flags.
 */
const char *tn_simd_init(void);

/* Name of the backend selected by the last tn_simd_init() ("Scalar" before). */
const char *tn_simd_backend_name(void);

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

/* Activation numerics of the ternary matmul, independent of which SIMD tier runs
 * (2026-11-18). INT8 is the W1.58A8 definition (per-row abs_max int8, include/math/
 * quantize_i8.h), used by bitnet.cpp and the reference model and fastest where VNNI or
 * dotprod exist; FLOAT keeps activations in float32 (slightly more precise, slower).
 * AUTO takes INT8 where the CPU has an int8 dot product and FLOAT elsewhere, i.e. what the
 * dispatcher always did. Setting a mode re-binds the kernels (tn_simd_init). */
typedef enum { TN_ACT_AUTO = 0, TN_ACT_INT8, TN_ACT_FLOAT } TnActQuant;
int tn_act_quant_parse(const char *name, TnActQuant *out); /* 0 = ok, -1 = unknown */
const char *tn_act_quant_name(TnActQuant m);
void tn_act_quant_set(TnActQuant m);
TnActQuant tn_act_quant_get(void);
/* "int8" or "float": what the ternary matmul actually runs after tier selection. */
const char *tn_act_quant_effective(void);

#endif /* TN_SIMD_DISPATCH_H */
