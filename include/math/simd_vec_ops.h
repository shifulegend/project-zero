#ifndef TN_SIMD_VEC_OPS_H
#define TN_SIMD_VEC_OPS_H

/* SIMD vector ops: element-wise kernels, SAXPY, the int4 KV rows and the INT8 / INT4 classifier rows.
 * Part of the SIMD dispatch table (math/simd_dispatch.h includes all of it; include this
 * header alone when only these kernels are used). Bound by tn_simd_init(). */

#include "core/platform.h"
#include "math/cls_kernels.h"
#include "math/kvq4_kernels.h"

/* Element-wise function pointer types */
typedef void (*tn_vec_add_fn)(float *out, const float *a, const float *b, int n);
typedef void (*tn_vec_mul_fn)(float *out, const float *a, const float *b, int n);
typedef void (*tn_vec_scale_fn)(float *x, float s, int n);
typedef void (*tn_silu_fn)(float *x, int n);
typedef void (*tn_relu2_fn)(float *x, int n);
typedef float (*tn_vec_dot_fn)(const float *a, const float *b, int n);
/* SAXPY: out[i] += scale * v[i]  — fused scale+accumulate for attention values */
typedef void (*tn_vec_saxpy_fn)(float *out, float scale, const float *v, int n);

/* Dispatch pointers */
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
/* INT8 / INT4 classifier (LM head) row kernels (math/cls_kernels.h). Generic (AVX2 or scalar) before
 * tn_simd_init() and on lower or forced tiers; the AVX-512 version on AVX-512F CPUs. */
extern tn_cls_rows_fn tn_cls_i8_rows;
extern tn_cls_rows_fn tn_cls_i4_rows;

#endif /* TN_SIMD_VEC_OPS_H */
