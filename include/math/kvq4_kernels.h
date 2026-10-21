#ifndef PZ_KVQ4_KERNELS_H
#define PZ_KVQ4_KERNELS_H

/*
 * math/kvq4_kernels — int4 KV-cache row kernels (2026-10-21).
 *
 * Layout (kv_cache/kv_quant_ops.h, --kv-type i4): groups of TN_KV_I4_GROUP
 * (32) values, 16 bytes of nibbles per group (byte j = elem j | elem j+16 << 4),
 * scales[2g] = d, scales[2g+1] = m, value = code*d + m.
 *
 *   dot : sum_i q_i * v_i           (attention scores)
 *   axpy: out_i += a * v_i          (attention-weighted V sum)
 *
 * The scalar versions live in kv_quant_ops.c; AVX2 / AVX-512 versions are
 * separate TUs built with their own ISA flags and chosen at runtime by
 * tn_simd_init() (simd_dispatch: tn_kvq4_dot / tn_kvq4_axpy), so the portable
 * dist binary uses them only on CPUs that have the ISA.
 */

#include <stdint.h>

typedef float (*tn_kvq4_dot_fn)(const float *q, const uint8_t *k, const float *scales, int n);
typedef void (*tn_kvq4_axpy_fn)(float *out, float a, const uint8_t *v, const float *scales, int n);

float kvq4_dot_scalar(const float *q, const uint8_t *k, const float *scales, int n);
void  kvq4_axpy_scalar(float *out, float a, const uint8_t *v, const float *scales, int n);

float kvq4_dot_avx2(const float *q, const uint8_t *k, const float *scales, int n);
void  kvq4_axpy_avx2(float *out, float a, const uint8_t *v, const float *scales, int n);

float kvq4_dot_avx512(const float *q, const uint8_t *k, const float *scales, int n);
void  kvq4_axpy_avx512(float *out, float a, const uint8_t *v, const float *scales, int n);

#endif /* PZ_KVQ4_KERNELS_H */
