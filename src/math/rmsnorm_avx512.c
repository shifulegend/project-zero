#include "core/platform.h"

#if TN_HAS_AVX512

#include <immintrin.h>
#include <math.h>

/*
 * AVX-512 RMS Normalization — 16-wide float32.
 *
 * Two-pass algorithm:
 *   Pass 1: sum of squares (16-wide FMA accumulation, reduce at end)
 *   Pass 2: normalize and weight-scale (16-wide multiply)
 *
 * Both passes use AVX2 tails for non-multiple-of-16 sizes.
 */
void rmsnorm_avx512(float *out, const float *x, const float *weight, int size, float eps) {
    /* Pass 1: sum of squares */
    __m512 acc16 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 xv = _mm512_loadu_ps(&x[i]);
        acc16 = _mm512_fmadd_ps(xv, xv, acc16);
    }
    float ss = _mm512_reduce_add_ps(acc16);

    __m256 acc8 = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        acc8 = _mm256_fmadd_ps(xv, xv, acc8);
    }
    __m128 hi = _mm256_extractf128_ps(acc8, 1);
    __m128 lo = _mm256_castps256_ps128(acc8);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s4);
    __m128 s2 = _mm_add_ps(s4, sh);
    ss += _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehl_ps(s2, s2)));
    for (; i < size; i++) ss += x[i] * x[i];

    ss /= size;
    ss = 1.0f / sqrtf(ss + eps);

    /* Pass 2: normalize and scale */
    __m512 norm16 = _mm512_set1_ps(ss);
    __m256 norm8  = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 xv = _mm512_loadu_ps(&x[i]);
        __m512 wv = _mm512_loadu_ps(&weight[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(_mm512_mul_ps(xv, norm16), wv));
    }
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 wv = _mm256_loadu_ps(&weight[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_mul_ps(xv, norm8), wv));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
}

#endif /* TN_HAS_AVX512 */
