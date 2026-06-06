#include "core/platform.h"

#if TN_HAS_AVX2

#include <immintrin.h>

/**
 * AVX2 ternary matrix-vector multiplication.
 *
 * Processes 8 floats per iteration using conditional add/subtract
 * with SIMD masks — ZERO floating-point multiplications in the
 * inner loop. Only the final scale multiply uses the FPU multiplier.
 *
 * Weight values must be in {-1, 0, 1} stored as int8_t.
 * Input x must be 32-byte aligned (64-byte from our allocator satisfies this).
 *
 * Phase 10 Evolution (Weight Packing):
 * Currently this kernel takes tn_i8 weights (1 byte per weight). When Phase 10
 * arrives, we pack 4 ternary weights per byte (2 bits each). The modification
 * is surgical: a _mm256_shuffle_epi8 LUT unpack step is inserted at the top of
 * the inner loop to "explode" packed bytes into the int8 values this kernel
 * already expects. The masking logic below (cmpeq → and_ps → add/sub) remains
 * EXACTLY the same. The per-matrix `scale` parameter will also change to a
 * `const float *scales` array for per-group scale factors (see Phase 10.1).
 */
void ternary_matmul_avx2(float *out, const float *x, const tn_i8 *w,
                          int n, int d, float scale) {
    const __m256i ones  = _mm256_set1_epi32(1);
    const __m256i neg1  = _mm256_set1_epi32(-1);

    for (int i = 0; i < d; i++) {
        const tn_i8 *row = w + (size_t)i * n;
        __m256 accum = _mm256_setzero_ps();

        int j = 0;
        /* Main AVX2 loop: 8 elements per iteration */
        for (; j + 7 < n; j += 8) {
            /* Load 8 input floats (aligned) */
            __m256 x_vec = _mm256_loadu_ps(&x[j]);

            /* Load 8 int8 weights and sign-extend to int32 */
            /* Load 8 bytes into a 64-bit value, then use epi8->epi32 conversion */
            __m128i w8 = _mm_loadl_epi64((const __m128i *)&row[j]);
            /* Sign-extend int8 -> int16 -> int32 */
            __m128i w16 = _mm_cvtepi8_epi16(w8);
            __m256i w32 = _mm256_cvtepi16_epi32(w16);

            /* Build masks: where weight == 1, where weight == -1 */
            __m256i mask_pos = _mm256_cmpeq_epi32(w32, ones);
            __m256i mask_neg = _mm256_cmpeq_epi32(w32, neg1);

            /* Filter input: zero out elements where weight is not +1 or -1 */
            __m256 to_add = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_pos));
            __m256 to_sub = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_neg));

            /* Accumulate: add where weight=+1, subtract where weight=-1 */
            accum = _mm256_add_ps(accum, to_add);
            accum = _mm256_sub_ps(accum, to_sub);
        }

        /* Horizontal sum of 8 floats in accumulator */
        /* [a0 a1 a2 a3 | a4 a5 a6 a7] */
        __m128 hi = _mm256_extractf128_ps(accum, 1);
        __m128 lo = _mm256_castps256_ps128(accum);
        __m128 sum4 = _mm_add_ps(lo, hi);
        /* [s0 s1 s2 s3] -> [s2 s3 s0 s1] */
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        __m128 final = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
        float val = _mm_cvtss_f32(final);

        /* Scalar tail: process remaining elements (n % 8 != 0) */
        for (; j < n; j++) {
            tn_i8 weight = row[j];
            if (weight == 1) {
                val += x[j];
            } else if (weight == -1) {
                val -= x[j];
            }
        }

        out[i] = val * scale;
    }
}

#endif /* TN_HAS_AVX2 */
