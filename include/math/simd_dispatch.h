#ifndef TN_SIMD_DISPATCH_H
#define TN_SIMD_DISPATCH_H

#include "core/platform.h"

/**
 * SIMD runtime dispatch layer.
 *
 * Provides function pointers that are initialized at startup to the best
 * available SIMD implementation (AVX2 > scalar fallback).
 *
 * Call tn_simd_init() once at program startup before using any math ops.
 */

/* Ternary matmul function pointer type */
typedef void (*tn_matmul_fn)(float *out, const float *x, const tn_i8 *w,
                              int n, int d, float scale);

/* RMSNorm function pointer type */
typedef void (*tn_rmsnorm_fn)(float *out, const float *x, const float *weight, int size, float eps);

/* Softmax function pointer type */
typedef void (*tn_softmax_fn)(float *x, int size);

/* Packed ternary matmul function pointer type (Phase 10) */
typedef void (*tn_matmul_packed_fn)(float *out, const float *x, const tn_u8 *packed_w,
                                     int n, int d, const float *scales, int group_size);

/* Ternary block unpacker function pointer type (Phase 10) */
typedef void (*tn_unpack_fn)(tn_i8 *out, const tn_u8 *packed, int count);

/* Element-wise function pointer types */
typedef void  (*tn_vec_add_fn)(float *out, const float *a, const float *b, int n);
typedef void  (*tn_vec_mul_fn)(float *out, const float *a, const float *b, int n);
typedef void  (*tn_vec_scale_fn)(float *x, float s, int n);
typedef void  (*tn_silu_fn)(float *x, int n);
typedef void  (*tn_relu2_fn)(float *x, int n);
typedef float (*tn_vec_dot_fn)(const float *a, const float *b, int n);
/* SAXPY: out[i] += scale * v[i]  — fused scale+accumulate for attention values */
typedef void  (*tn_vec_saxpy_fn)(float *out, float scale, const float *v, int n);

/* Global dispatch table */
extern tn_matmul_fn         tn_ternary_matmul;
extern tn_matmul_packed_fn  tn_ternary_matmul_packed;
extern tn_unpack_fn         tn_unpack_block;
extern tn_rmsnorm_fn        tn_rmsnorm;
extern tn_softmax_fn        tn_softmax;
extern tn_vec_add_fn        tn_vec_add;
extern tn_vec_mul_fn        tn_vec_mul;
extern tn_vec_scale_fn      tn_vec_scale;
extern tn_silu_fn           tn_silu;
extern tn_relu2_fn          tn_relu2;
extern tn_vec_dot_fn        tn_vec_dot;
extern tn_vec_saxpy_fn      tn_vec_saxpy;

/**
 * Initialize SIMD dispatch. Probes CPU features and sets function pointers
 * to the best available implementation. Must be called before any math ops.
 *
 * Returns a human-readable string describing the selected backend
 * (e.g., "AVX2", "Scalar").
 *
 * TODO (QA-ISS-005): SIMD availability is currently determined at compile time
 * via preprocessor flags (#ifdef __AVX2__, etc.). On machines without the
 * required ISA but compiled with those flags, this will crash with SIGILL.
 * A future version should add runtime CPUID detection to select the correct
 * kernel at startup, independent of compile-time flags.
 */
const char *tn_simd_init(void);

#endif /* TN_SIMD_DISPATCH_H */
