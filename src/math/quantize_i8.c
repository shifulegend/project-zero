#include "math/quantize_i8.h"

#include <math.h>
#include <string.h>

#include "core/platform.h"

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
        if (v > 127.0f)
            v = 127.0f;
        else if (v < -127.0f)
            v = -127.0f;
        /* Round to nearest, ties to even (the specification in quantize_i8.h). Truncation here,
         * which this line did until 2026-11-18, gave ~10x the quantization error (docs/rca/...). */
        q[i] = (int8_t)lrintf(v);
    }

    return scale;
}

int32_t sum_i8(const int8_t *arr, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)arr[i];
    return s;
}

/* The AVX-512 versions (quantize_row_to_i8_avx512, sum_i8_avx512) live in
 * quantize_i8_avx512.c, a translation unit built with the AVX-512 flags in every
 * x86 build: this file is compiled at the AVX2/portable baseline, where a
 * TN_HAS_AVX512 guard here would silently turn them into scalar fallbacks
 * (2026-11-18, docs/ai/mistakes.md). */

/* ── AVX2 accelerated ─────────────────────────────────────────────────────── */
#if TN_HAS_AVX2
#include <immintrin.h>

float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n) {
    /* Pass 1: find abs max */
    __m256 vmax            = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i                  = 0;

    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v        = _mm256_andnot_ps(sign_mask, v);
        vmax     = _mm256_max_ps(vmax, v);
    }

    /* Horizontal max of 8 lanes */
    __m128 hi4    = _mm256_extractf128_ps(vmax, 1);
    __m128 lo4    = _mm256_castps256_ps128(vmax);
    __m128 m4     = _mm_max_ps(lo4, hi4);
    __m128 m2     = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1     = _mm_max_ss(m2, _mm_movehdup_ps(m2));
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
    __m256 vpos127  = _mm256_set1_ps(127.0f);
    __m256 vneg127  = _mm256_set1_ps(-127.0f);

    /* Pass 2: quantize 8 at a time */
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v        = _mm256_mul_ps(v, vscale);
        v        = _mm256_min_ps(v, vpos127);
        v        = _mm256_max_ps(v, vneg127);

        /* Pack float→int32→int8: float→i32 (round to nearest even, MXCSR default) then two pack
         * steps */
        __m256i i32 = _mm256_cvtps_epi32(v);
        /* Pack int32→int16 (saturating signed) */
        __m128i lo_i32 = _mm256_castsi256_si128(i32);
        __m128i hi_i32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16    = _mm_packs_epi32(lo_i32, hi_i32);
        /* Pack int16→int8 (saturating signed) */
        __m128i i8 = _mm_packs_epi16(i16, _mm_setzero_si128());
        /* Store 8 bytes */
        _mm_storel_epi64((__m128i *)(q + i), i8);
    }

    for (; i < n; i++) {
        float v = x[i] * inv_scale;
        if (v > 127.0f)
            v = 127.0f;
        else if (v < -127.0f)
            v = -127.0f;
        q[i] = (int8_t)lrintf(v);
    }

    return scale;
}

#else /* !TN_HAS_AVX2 */

float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n) {
    return quantize_row_to_i8(x, q, n);
}

#endif /* TN_HAS_AVX2 */

/* ── NEON (AArch64) ───────────────────────────────────────────────────────── */
#if TN_HAS_NEON && defined(__aarch64__)
#include <arm_neon.h>

float quantize_row_to_i8_neon(const float *x, int8_t *q, int n) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i            = 0;
    for (; i + 3 < n; i += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + i)));
    float abs_max = vmaxvq_f32(vmax);
    for (; i < n; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > abs_max) abs_max = a;
    }
    if (abs_max == 0.0f) {
        memset(q, 0, (size_t)n);
        return 0.0f;
    }
    float scale          = abs_max / 127.0f;
    float inv_scale      = 127.0f / abs_max;
    const float32x4_t hi = vdupq_n_f32(127.0f), lo = vdupq_n_f32(-127.0f);
    for (i = 0; i + 7 < n; i += 8) {
        float32x4_t a = vminq_f32(vmaxq_f32(vmulq_n_f32(vld1q_f32(x + i), inv_scale), lo), hi);
        float32x4_t b = vminq_f32(vmaxq_f32(vmulq_n_f32(vld1q_f32(x + i + 4), inv_scale), lo), hi);
        /* vcvtnq: float -> int32 round to nearest, ties to even (independent of FPCR) */
        int16x8_t h = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(a)), vqmovn_s32(vcvtnq_s32_f32(b)));
        vst1_s8(q + i, vqmovn_s16(h));
    }
    for (; i < n; i++) {
        float v = x[i] * inv_scale;
        if (v > 127.0f)
            v = 127.0f;
        else if (v < -127.0f)
            v = -127.0f;
        q[i] = (int8_t)lrintf(v);
    }
    return scale;
}

#else /* no AArch64 NEON */

float quantize_row_to_i8_neon(const float *x, int8_t *q, int n) {
    return quantize_row_to_i8(x, q, n);
}

#endif
