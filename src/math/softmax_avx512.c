#include "core/platform.h"

#if TN_HAS_AVX512

#include <immintrin.h>
#include <math.h>

/*
 * AVX-512 numerically stable softmax — three-pass algorithm.
 *
 * Pass 1: find max (16-wide SIMD reduction)     — pure SIMD
 * Pass 2: exp(x[i] - max) + sum                — scalar exp, SIMD unavoidable
 * Pass 3: normalize by 1/sum                    — 16-wide SIMD
 *
 * The exp() bottleneck is inherent; there is no fast vectorised expf()
 * without SVML (Intel proprietary) or a polynomial approximation.
 * Passes 1 and 3 are fully vectorised and contribute the bulk of
 * memory bandwidth savings vs. the scalar implementation.
 */
void softmax_avx512(float *x, int size) {
    int i;

    /* Pass 1: find max — 16-wide */
    __m512 max16 = _mm512_set1_ps(-3.4028235e+38f);
    i = 0;
    for (; i + 15 < size; i += 16) {
        max16 = _mm512_max_ps(max16, _mm512_loadu_ps(&x[i]));
    }
    float max_val = _mm512_reduce_max_ps(max16);

    __m256 max8 = _mm256_set1_ps(max_val);
    for (; i + 7 < size; i += 8) {
        max8 = _mm256_max_ps(max8, _mm256_loadu_ps(&x[i]));
    }
    __m128 hi = _mm256_extractf128_ps(max8, 1);
    __m128 lo = _mm256_castps256_ps128(max8);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    float max8_val = _mm_cvtss_f32(_mm_max_ss(m2, _mm_movehdup_ps(m2)));
    if (max8_val > max_val) max_val = max8_val;
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Pass 2: exp(x - max) and sum — scalar exp is unavoidable */
    float sum = 0.0f;
    for (i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Pass 3: normalize — 16-wide */
    float inv_sum = 1.0f / sum;
    __m512 inv16 = _mm512_set1_ps(inv_sum);
    __m256 inv8  = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 15 < size; i += 16) {
        _mm512_storeu_ps(&x[i], _mm512_mul_ps(_mm512_loadu_ps(&x[i]), inv16));
    }
    for (; i + 7 < size; i += 8) {
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(_mm256_loadu_ps(&x[i]), inv8));
    }
    for (; i < size; i++) x[i] *= inv_sum;
}

#endif /* TN_HAS_AVX512 */
