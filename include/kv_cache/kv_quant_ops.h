#ifndef TN_KV_QUANT_OPS_H
#define TN_KV_QUANT_OPS_H

/*
 * kv_quant_ops.h — plan item 8.5: int8 KV-cache row operations used by the
 * attention read/write paths when the KV cache is stored quantized
 * (--kv-type i8). One absmax scale per (layer, kv_head, position) row of
 * head_dim values. Loops are written to auto-vectorise (int8 -> float
 * convert + FMA) on every SIMD tier, including the portable dist build.
 */

#include "core/platform.h"

typedef enum {
    KV_TYPE_F32 = 0,
    KV_TYPE_I8  = 1
} KVCacheType;

/* "f32" | "i8" -> type, -1 otherwise. */
int         kv_type_parse(const char *s);
const char *kv_type_name(KVCacheType t);

/* dst[i] = round(src[i] / scale), scale = max|src| / 127 (0 for an all-zero row). */
void  kvq_store_row(tn_i8 *dst, float *scale_out, const float *src, int n);
/* sum_i q[i] * k[i] (caller applies the row scale). */
float kvq_dot(const float *q, const tn_i8 *k, int n);
/* out[i] += a * v[i] (fold the row scale into a). */
void  kvq_axpy(float *out, float a, const tn_i8 *v, int n);

#endif /* TN_KV_QUANT_OPS_H */
