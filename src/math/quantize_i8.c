#include "math/quantize_i8.h"
#include "core/platform.h"
#include <math.h>
#include <string.h>

/* ── Scalar reference ─────────────────────────────────────────────────────── */

float quantize_row_to_i8(const float *x, int8_t *q, int n) {
    /* Find absolute maximum */
    float abs_max = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > abs_max) abs_max = a;
    }

    if (abs_max == 0.0f) {
        memset(q, 0, (size_t)n);
        return 0.0f;
    }

    float scale     = abs_max / 127.0f;
    float inv_scale = 127.0f / abs_max;

    for (int i = 0; i < n; i++) {
        float v = x[i] * inv_scale;
        /* Clamp to [-127, 127] — avoids -128 which would overflow on multiply */
        if      (v >  127.0f) v =  127.0f;
        else if (v < -127.0f) v = -127.0f;
        q[i] = (int8_t)(int)v;  /* truncation toward zero (same as llama.cpp) */
    }

    return scale;
}

int32_t sum_i8(const int8_t *arr, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)arr[i];
    return s;
}

/* ── AVX-512 accelerated ──────────────────────────────────────────────────── */
#if TN_HAS_AVX512
#include <immintrin.h>

float quantize_row_to_i8_avx512(const float *x, int8_t *q, int n) {
    /* ── Pass 1: find absolute maximum using AVX-512 ── */
    __m512 vmax = _mm512_setzero_ps();
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);  /* 0x80000000 per lane */
    int i = 0;

    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        v = _mm512_andnot_ps(sign_mask, v);  /* abs */
        vmax = _mm512_max_ps(vmax, v);
    }

    float abs_max = _mm512_reduce_max_ps(vmax);

    /* Scalar tail */
    for (; i < n; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > abs_max) abs_max = a;
    }

    if (abs_max == 0.0f) {
        memset(q, 0, (size_t)n);
        return 0.0f;
    }

    float scale     = abs_max / 127.0f;
    float inv_scale = 127.0f / abs_max;

    /* ── Pass 2: quantize using AVX-512 ── */
    __m512 vscale   = _mm512_set1_ps(inv_scale);
    __m512 vpos127  = _mm512_set1_ps( 127.0f);
    __m512 vneg127  = _mm512_set1_ps(-127.0f);

    i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        v = _mm512_mul_ps(v, vscale);
        v = _mm512_min_ps(v, vpos127);
        v = _mm512_max_ps(v, vneg127);

        /* float → int32 (rounding toward zero) → pack to int8 */
        __m512i i32 = _mm512_cvttps_epi32(v);

        /* Pack 16 int32 → 16 int8 (values in [-127,127], safe truncation) */
        __m128i i8  = _mm512_cvtepi32_epi8(i32);
        _mm_storeu_si128((__m128i*)(q + i), i8);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        float v = x[i] * inv_scale;
        if      (v >  127.0f) v =  127.0f;
        else if (v < -127.0f) v = -127.0f;
        q[i] = (int8_t)(int)v;
    }

    return scale;
}

int32_t sum_i8_avx512(const int8_t *arr, int n) {
    /*
     * Accumulate int8 into int32 using sign-extend and add.
     * Process 64 bytes at a time, widen int8→int16 then int16→int32
     * to avoid overflow on long vectors.
     */
    __m512i acc = _mm512_setzero_si512();
    int i = 0;

    for (; i + 63 < n; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(arr + i));

        /* Widen each int8 lane to int16, accumulate into int32 accumulators.
         * We split the 64 bytes into 4 groups of 16, extend each to 32-bit. */
        __m128i b0 = _mm512_extracti32x4_epi32(v, 0);
        __m128i b1 = _mm512_extracti32x4_epi32(v, 1);
        __m128i b2 = _mm512_extracti32x4_epi32(v, 2);
        __m128i b3 = _mm512_extracti32x4_epi32(v, 3);

        /* Sign-extend 16 int8 → 16 int32 for each quarter */
        acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(b0));
        acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(b1));
        acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(b2));
        acc = _mm512_add_epi32(acc, _mm512_cvtepi8_epi32(b3));
    }

    int32_t total = _mm512_reduce_add_epi32(acc);

    /* Scalar tail */
    for (; i < n; i++) total += (int32_t)arr[i];

    return total;
}

#else /* !TN_HAS_AVX512 */

float quantize_row_to_i8_avx512(const float *x, int8_t *q, int n) {
    return quantize_row_to_i8(x, q, n);
}
int32_t sum_i8_avx512(const int8_t *arr, int n) {
    return sum_i8(arr, n);
}

#endif /* TN_HAS_AVX512 */

/* ── AVX2 accelerated ─────────────────────────────────────────────────────── */
#if TN_HAS_AVX2
#include <immintrin.h>

float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n) {
    /* Pass 1: find abs max */
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i = 0;

    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_andnot_ps(sign_mask, v);
        vmax = _mm256_max_ps(vmax, v);
    }

    /* Horizontal max of 8 lanes */
    __m128 hi4  = _mm256_extractf128_ps(vmax, 1);
    __m128 lo4  = _mm256_castps256_ps128(vmax);
    __m128 m4   = _mm_max_ps(lo4, hi4);
    __m128 m2   = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1   = _mm_max_ss(m2, _mm_movehdup_ps(m2));
    float abs_max = _mm_cvtss_f32(m1);

    for (; i < n; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > abs_max) abs_max = a;
    }

    if (abs_max == 0.0f) {
        memset(q, 0, (size_t)n);
        return 0.0f;
    }

    float scale     = abs_max / 127.0f;
    float inv_scale = 127.0f / abs_max;
    __m256 vscale   = _mm256_set1_ps(inv_scale);
    __m256 vpos127  = _mm256_set1_ps( 127.0f);
    __m256 vneg127  = _mm256_set1_ps(-127.0f);

    /* Pass 2: quantize 8 at a time */
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256  v   = _mm256_loadu_ps(&x[i]);
        v = _mm256_mul_ps(v, vscale);
        v = _mm256_min_ps(v, vpos127);
        v = _mm256_max_ps(v, vneg127);

        /* Pack float→int32→int8: float→i32 then two pack steps */
        __m256i i32 = _mm256_cvttps_epi32(v);
        /* Pack int32→int16 (saturating signed) */
        __m128i lo_i32 = _mm256_castsi256_si128(i32);
        __m128i hi_i32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16    = _mm_packs_epi32(lo_i32, hi_i32);
        /* Pack int16→int8 (saturating signed) */
        __m128i i8     = _mm_packs_epi16(i16, _mm_setzero_si128());
        /* Store 8 bytes */
        _mm_storel_epi64((__m128i*)(q + i), i8);
    }

    for (; i < n; i++) {
        float v = x[i] * inv_scale;
        if      (v >  127.0f) v =  127.0f;
        else if (v < -127.0f) v = -127.0f;
        q[i] = (int8_t)(int)v;
    }

    return scale;
}

#else /* !TN_HAS_AVX2 */

float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n) {
    return quantize_row_to_i8(x, q, n);
}

#endif /* TN_HAS_AVX2 */
