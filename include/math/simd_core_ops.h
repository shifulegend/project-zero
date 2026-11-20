#ifndef TN_SIMD_CORE_OPS_H
#define TN_SIMD_CORE_OPS_H

/* SIMD core ops: ternary matmul (plain and packed), unpack, RMSNorm and softmax.
 * Part of the SIMD dispatch table (math/simd_dispatch.h includes all of it; include this
 * header alone when only these kernels are used). Bound by tn_simd_init(). */

#include "core/platform.h"

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

/* Dispatch pointers */
extern tn_matmul_fn tn_ternary_matmul;
extern tn_matmul_packed_fn tn_ternary_matmul_packed;
extern tn_unpack_fn tn_unpack_block;
extern tn_rmsnorm_fn tn_rmsnorm;
extern tn_softmax_fn tn_softmax;

#endif /* TN_SIMD_CORE_OPS_H */
