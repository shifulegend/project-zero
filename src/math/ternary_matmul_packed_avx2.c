#include "core/platform.h"

#if TN_HAS_AVX2

#include "math/ternary_matmul_packed.h"
#include "core/unpack.h"
#include <immintrin.h>
#include <string.h>

/**
 * AVX2 fused unpack+matmul for 2-bit packed ternary weights.
 *
 * Strategy: unpack 8 weights at a time from packed bytes, then apply
 * the same conditional add/subtract SIMD pipeline as ternary_matmul_avx2.
 *
 * For per-group scales, the accumulation is partitioned at group boundaries
 * and each group's partial sum is multiplied by its scale factor.
 */

static inline __m256i unpack8_to_epi32(const tn_u8 *row, int j) {
    /*
     * Optimized SIMD unpack:
     * 8 weights are in 16 bits (2 bytes) at row + j/4.
     * We load these 16 bits, broadcast to all 8 lanes,
     * and shift each lane by its bit-offset (0,2..14).
     */
    uint16_t bits;
    memcpy(&bits, row + (j >> 2), sizeof(bits));

    __m256i v = _mm256_set1_epi32((int)bits);
    __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

    /* Parallel shift and mask */
    v = _mm256_srlv_epi32(v, shifts);
    v = _mm256_and_si256(v, _mm256_set1_epi32(3));

    /* uint2 -> ternary (v - 1) */
    return _mm256_sub_epi32(v, _mm256_set1_epi32(1));
}

#define TN_PREFETCH_ROWS 8

void ternary_matmul_packed_avx2(float *out, const float *x, const tn_u8 *packed_w,
                                int n, int d, const float *scales, int group_size) {
    size_t row_bytes = ((size_t)n + 3) >> 2;

    const __m256i ones = _mm256_set1_epi32(1);
    const __m256i neg1 = _mm256_set1_epi32(-1);

    for (int i = 0; i < d; i++) {
        const tn_u8 *row = packed_w + (size_t)i * row_bytes;

        if (i + TN_PREFETCH_ROWS < d) {
            _mm_prefetch((const char *)(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes),
                         _MM_HINT_T1);
        }

        if (group_size <= 0) {
            /* Per-matrix scale mode */
            __m256 accum = _mm256_setzero_ps();
            int j = 0;

            for (; j + 7 < n; j += 8) {
                __m256 x_vec = _mm256_loadu_ps(&x[j]);
                __m256i w32 = unpack8_to_epi32(row, j);

                __m256i mask_pos = _mm256_cmpeq_epi32(w32, ones);
                __m256i mask_neg = _mm256_cmpeq_epi32(w32, neg1);

                __m256 to_add = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_pos));
                __m256 to_sub = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_neg));

                accum = _mm256_add_ps(accum, to_add);
                accum = _mm256_sub_ps(accum, to_sub);
            }

            /* Horizontal sum */
            __m128 hi = _mm256_extractf128_ps(accum, 1);
            __m128 lo = _mm256_castps256_ps128(accum);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sum2 = _mm_add_ps(sum4, shuf);
            __m128 final = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
            float val = _mm_cvtss_f32(final);

            /* Scalar tail */
            for (; j < n; j++) {
                tn_i8 w = unpack_ternary(row, j);
                if (w == 1) val += x[j];
                else if (w == -1) val -= x[j];
            }

            out[i] = val * scales[0];

        } else {
            /* Per-group scale mode */
            float total = 0.0f;
            int n_groups = (n + group_size - 1) / group_size;

            for (int g = 0; g < n_groups; g++) {
                int start = g * group_size;
                int end = start + group_size;
                if (end > n) end = n;

                __m256 accum = _mm256_setzero_ps();
                int j = start;

                for (; j + 7 < end; j += 8) {
                    __m256 x_vec = _mm256_loadu_ps(&x[j]);
                    __m256i w32 = unpack8_to_epi32(row, j);

                    __m256i mask_pos = _mm256_cmpeq_epi32(w32, ones);
                    __m256i mask_neg = _mm256_cmpeq_epi32(w32, neg1);

                    __m256 to_add = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_pos));
                    __m256 to_sub = _mm256_and_ps(x_vec, _mm256_castsi256_ps(mask_neg));

                    accum = _mm256_add_ps(accum, to_add);
                    accum = _mm256_sub_ps(accum, to_sub);
                }

                __m128 hi = _mm256_extractf128_ps(accum, 1);
                __m128 lo = _mm256_castps256_ps128(accum);
                __m128 sum4 = _mm_add_ps(lo, hi);
                __m128 shuf = _mm_movehdup_ps(sum4);
                __m128 sum2 = _mm_add_ps(sum4, shuf);
                __m128 final = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
                float group_sum = _mm_cvtss_f32(final);

                /* Scalar tail for this group */
                for (; j < end; j++) {
                    tn_i8 w = unpack_ternary(row, j);
                    if (w == 1) group_sum += x[j];
                    else if (w == -1) group_sum -= x[j];
                }

                int scale_idx = i * n_groups + g;
                total += group_sum * scales[scale_idx];
            }
            out[i] = total;
        }
    }
}

#endif /* TN_HAS_AVX2 */
