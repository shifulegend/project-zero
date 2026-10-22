/*
 * math/kvq4_avx512.c — AVX-512 int4 KV-cache kernels (see math/kvq4_kernels.h).
 *
 * One 32-value group = two 16-float vectors (low nibbles = elems 0..15, high
 * nibbles = elems 16..31); dot = d * sum(q*c) + m * sum(q), reduced once per row.
 */
#include "core/platform.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/kvq4_kernels.h"

#if TN_HAS_AVX512

#include <immintrin.h>

#if TN_KV_I4_GROUP != 32
#error "kvq4 AVX-512 kernels assume 32-value groups"
#endif

static inline void unpack32(const uint8_t *b, __m512 *lo, __m512 *hi) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i x = _mm_loadu_si128((const __m128i *)b);
    *lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(x, m4)));
    *hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(x, 4), m4)));
}

float kvq4_dot_avx512(const float *q, const uint8_t *k, const float *scales, int n) {
    __m512 acc = _mm512_setzero_ps();
    for (int g = 0; g < n / 32; g++) {
        const float *qq = q + g * 32;
        __m512 clo, chi;
        unpack32(k + g * 16, &clo, &chi);
        __m512 q0 = _mm512_loadu_ps(qq), q1 = _mm512_loadu_ps(qq + 16);
        __m512 qc = _mm512_fmadd_ps(q1, chi, _mm512_mul_ps(q0, clo));
        __m512 qs = _mm512_add_ps(q0, q1);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(scales[2 * g]), qc, acc);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(scales[2 * g + 1]), qs, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

void kvq4_axpy_avx512(float *out, float a, const uint8_t *v, const float *scales, int n) {
    for (int g = 0; g < n / 32; g++) {
        float *oo = out + g * 32;
        __m512 clo, chi;
        unpack32(v + g * 16, &clo, &chi);
        __m512 ad = _mm512_set1_ps(a * scales[2 * g]);
        __m512 am = _mm512_set1_ps(a * scales[2 * g + 1]);
        _mm512_storeu_ps(oo, _mm512_fmadd_ps(ad, clo, _mm512_add_ps(_mm512_loadu_ps(oo), am)));
        _mm512_storeu_ps(oo + 16, _mm512_fmadd_ps(ad, chi, _mm512_add_ps(_mm512_loadu_ps(oo + 16), am)));
    }
}

#endif /* TN_HAS_AVX512 */
