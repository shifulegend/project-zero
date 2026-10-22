/*
 * math/kvq4_avx2.c — AVX2 int4 KV-cache kernels (see math/kvq4_kernels.h).
 *
 * Per 32-value group: 16 bytes -> low/high nibbles -> 4 x 8 floats. The dot
 * is computed as d * sum(q*c) + m * sum(q) in vector form and reduced once
 * per row, so the per-group cost is 4 FMAs + 3 adds + 2 FMAs.
 */
#include "core/platform.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/kvq4_kernels.h"

#if TN_HAS_AVX2

#include <immintrin.h>

#if TN_KV_I4_GROUP != 32
#error "kvq4 AVX2 kernels assume 32-value groups"
#endif

static inline void unpack32(const uint8_t *b, __m256 c[4]) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i x  = _mm_loadu_si128((const __m128i *)b);
    __m128i lo = _mm_and_si128(x, m4);                     /* elems 0..15  */
    __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), m4);  /* elems 16..31 */
    c[0] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
    c[1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
    c[2] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
    c[3] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
}

float kvq4_dot_avx2(const float *q, const uint8_t *k, const float *scales, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < n / 32; g++) {
        const float *qq = q + g * 32;
        __m256 c[4];
        unpack32(k + g * 16, c);
        __m256 q0 = _mm256_loadu_ps(qq), q1 = _mm256_loadu_ps(qq + 8);
        __m256 q2 = _mm256_loadu_ps(qq + 16), q3 = _mm256_loadu_ps(qq + 24);
        __m256 qc = _mm256_mul_ps(q0, c[0]);
        qc = _mm256_fmadd_ps(q1, c[1], qc);
        qc = _mm256_fmadd_ps(q2, c[2], qc);
        qc = _mm256_fmadd_ps(q3, c[3], qc);
        __m256 qs = _mm256_add_ps(_mm256_add_ps(q0, q1), _mm256_add_ps(q2, q3));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scales[2 * g]), qc, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scales[2 * g + 1]), qs, acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

void kvq4_axpy_avx2(float *out, float a, const uint8_t *v, const float *scales, int n) {
    for (int g = 0; g < n / 32; g++) {
        float *oo = out + g * 32;
        __m256 c[4];
        unpack32(v + g * 16, c);
        __m256 ad = _mm256_set1_ps(a * scales[2 * g]);
        __m256 am = _mm256_set1_ps(a * scales[2 * g + 1]);
        for (int j = 0; j < 4; j++) {
            __m256 o = _mm256_add_ps(_mm256_loadu_ps(oo + 8 * j), am);
            _mm256_storeu_ps(oo + 8 * j, _mm256_fmadd_ps(ad, c[j], o));
        }
    }
}

#endif /* TN_HAS_AVX2 */
