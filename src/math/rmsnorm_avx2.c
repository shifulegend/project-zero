#include "core/platform.h"

#if TN_HAS_AVX2

#include <immintrin.h>
#include <math.h>

/**
 * AVX2-accelerated RMS Normalization.
 *
 * out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
 *
 * Two-pass algorithm:
 *   Pass 1: Compute sum of squares using 8-wide SIMD accumulation
 *   Pass 2: Normalize and scale using 8-wide SIMD multiply
 */
void rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps) {
    /* Pass 1: Sum of squares with AVX2 */
    __m256 ss_vec = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        ss_vec = _mm256_fmadd_ps(xv, xv, ss_vec);
    }

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(ss_vec, 1);
    __m128 lo = _mm256_castps256_ps128(ss_vec);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 final = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
    float ss = _mm_cvtss_f32(final);

    /* Scalar tail for sum of squares */
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }

    /* Compute normalization factor: 1 / sqrt(mean + eps) */
    ss /= size;
    ss = 1.0f / sqrtf(ss + eps);
    __m256 ss_broadcast = _mm256_set1_ps(ss);

    /* Pass 2: Normalize and scale with AVX2 */
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 wv = _mm256_loadu_ps(&weight[i]);
        /* out = x * ss * weight */
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(xv, ss_broadcast), wv);
        _mm256_storeu_ps(&out[i], result);
    }

    /* Scalar tail */
    for (; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

#endif /* TN_HAS_AVX2 */
