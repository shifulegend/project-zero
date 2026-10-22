#ifndef TN_KV_QUANT_OPS_H
#define TN_KV_QUANT_OPS_H

/*
 * kv_quant_ops.h — plan item 8.5: quantized KV-cache row operations used by
 * the attention read/write paths when the KV cache is stored quantized.
 *   --kv-type i8: one absmax scale per (layer, kv_head, position) row.
 *   --kv-type i4 (2026-10-19): Q4_1-style groups of TN_KV_I4_GROUP values,
 *     codes 0..15 with a (scale, min) pair per group;
 *     byte j holds element j (low nibble) and j+16 (high nibble), as ggml. Loops are written to auto-vectorise (int8 -> float
 * convert + FMA) on every SIMD tier, including the portable dist build.
 */

#include "core/platform.h"

typedef enum {
    KV_TYPE_F32 = 0,
    KV_TYPE_I8  = 1,
    KV_TYPE_I4  = 2
} KVCacheType;

#define TN_KV_I4_GROUP 32   /* values per int4 scale (head_dim must be a multiple) */

/* "f32" | "i8" | "i4" -> type, -1 otherwise. */
int         kv_type_parse(const char *s);
const char *kv_type_name(KVCacheType t);

/* dst[i] = round(src[i] / scale), scale = max|src| / 127 (0 for an all-zero row). */
void  kvq_store_row(tn_i8 *dst, float *scale_out, const float *src, int n);
/* sum_i q[i] * k[i] (caller applies the row scale). */
float kvq_dot(const float *q, const tn_i8 *k, int n);
/* out[i] += a * v[i] (fold the row scale into a). */
void  kvq_axpy(float *out, float a, const tn_i8 *v, int n);

/* Storage per head_dim row for a quantized type: data bytes and F32 scales.
 * Returns 0, or -1 if the type cannot store rows of n values (i4 needs
 * n % TN_KV_I4_GROUP == 0; F32 is not a quantized type). */
int   kvq_row_layout(KVCacheType t, int n, int *bytes, int *scales);
/* Type-dispatched row ops; `scales` points at the row's first scale and the
 * results already include them. */
void  kvq_store(KVCacheType t, tn_i8 *dst, float *scales, const float *src, int n);
float kvq_dot_row(KVCacheType t, const float *q, const tn_i8 *k, const float *scales, int n);
void  kvq_axpy_row(KVCacheType t, float *out, float a, const tn_i8 *v, const float *scales, int n);

#endif /* TN_KV_QUANT_OPS_H */
