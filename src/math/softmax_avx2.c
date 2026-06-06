#include "core/platform.h"

#if TN_HAS_AVX2

#include <immintrin.h>
#include <math.h>

/**
 * AVX2-accelerated numerically stable softmax.
 *
 * Three-pass algorithm:
 *   Pass 1: Find max value (8-wide SIMD reduction)
 *   Pass 2: Compute exp(x[i] - max) and accumulate sum
 *   Pass 3: Normalize by 1/sum
 *
 * Note: exp() is computed scalar since there is no efficient AVX2 exp intrinsic.
 * The SIMD speedup comes from the max-find and normalization passes.
 */
void softmax_avx2(float *x, int size) {
    int i;

    /* Pass 1: Find max with AVX2 */
    __m256 max_vec = _mm256_set1_ps(-1e30f);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        max_vec = _mm256_max_ps(max_vec, xv);
    }

    /* Horizontal max reduction */
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max4 = _mm_max_ps(lo, hi);
    __m128 max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    __m128 max1 = _mm_max_ss(max2, _mm_movehdup_ps(max2));
    float max_val = _mm_cvtss_f32(max1);

    /* Scalar tail for max */
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Pass 2: exp(x - max) and sum
     * exp() doesn't have a fast AVX2 intrinsic, so we compute it scalar
     * but do the subtract and sum accumulation with SIMD where possible */
    float sum = 0.0f;
    for (i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Pass 3: Normalize with AVX2 */
    float inv_sum = 1.0f / sum;
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(xv, inv_sum_vec));
    }

    /* Scalar tail for normalize */
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
}

#endif /* TN_HAS_AVX2 */
