/*
 * cls_kernels.c — generic (AVX2 or scalar) INT8 / INT4 classifier row kernels.
 * Contract and layout: include/math/cls_kernels.h. Built at the AVX2 baseline (Makefile AVX2_TUS);
 * the AVX-512 versions are in cls_kernels_avx512.c and bound by tn_simd_init().
 */
#include "math/cls_kernels.h"

#include <string.h>

#if TN_HAS_AVX2
#include <immintrin.h>

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s        = _mm_add_ps(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehl_ps(s, s)));
}
#endif

void tn_cls_i8_rows_generic(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                            int n_blocks, int start, int end) {
    const int nbf = n / TN_CLS_QUANT_BLOCK; /* full blocks */
    for (int i = start; i < end; i++) {
        const tn_u8 *row = w + (size_t)i * n;
        const float *rs  = scales + (size_t)i * n_blocks;
        float total;
#if TN_HAS_AVX2
        __m256 acc         = _mm256_setzero_ps();
        const __m256i bias = _mm256_set1_epi32(128);
        for (int b = 0; b < nbf; b++) {
            const tn_u8 *bw = row + (size_t)b * TN_CLS_QUANT_BLOCK;
            const float *bx = x + (size_t)b * TN_CLS_QUANT_BLOCK;
            __m256 p        = _mm256_setzero_ps();
            for (int j = 0; j < TN_CLS_QUANT_BLOCK; j += 8) {
                __m256 wv = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(bw + j))), bias));
                p         = _mm256_fmadd_ps(wv, _mm256_loadu_ps(bx + j), p);
            }
            acc = _mm256_fmadd_ps(p, _mm256_set1_ps(rs[b]), acc);
        }
        total = hsum256(acc);
#else
        total = 0.0f;
        for (int b = 0; b < nbf; b++) {
            float v = 0.0f;
            for (int j = 0; j < TN_CLS_QUANT_BLOCK; j++)
                v += (float)((int)row[b * TN_CLS_QUANT_BLOCK + j] - 128) *
                     x[b * TN_CLS_QUANT_BLOCK + j];
            total += v * rs[b];
        }
#endif
        if (nbf < n_blocks)
            total += tn_cls_i8_row_tail(row, x, nbf * TN_CLS_QUANT_BLOCK, n) * rs[nbf];
        out[i] = total;
    }
}

void tn_cls_i4_rows_generic(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                            int n_blocks, int start, int end) {
    const int nbf          = n / TN_CLS_QUANT_BLOCK;
    const size_t row_bytes = ((size_t)n + 1) / 2;
    for (int i = start; i < end; i++) {
        const tn_u8 *row = w + (size_t)i * row_bytes;
        const float *rs  = scales + (size_t)i * n_blocks;
        float total;
#if TN_HAS_AVX2
        __m256 acc          = _mm256_setzero_ps();
        const __m128i mask4 = _mm_set1_epi8(0x0F);
        const __m256i bias  = _mm256_set1_epi32(8);
        for (int b = 0; b < nbf; b++) {
            const tn_u8 *bw = row + (size_t)b * (TN_CLS_QUANT_BLOCK / 2);
            const float *bx = x + (size_t)b * TN_CLS_QUANT_BLOCK;
            __m256 p        = _mm256_setzero_ps();
            for (int j = 0; j < TN_CLS_QUANT_BLOCK; j += 8) {
                tn_u32 four;
                memcpy(&four, bw + j / 2, 4); /* 4 bytes = 8 nibbles */
                __m128i packed = _mm_cvtsi32_si128((int)four);
                __m128i lo     = _mm_and_si128(packed, mask4);
                __m128i hi     = _mm_and_si128(_mm_srli_epi16(packed, 4), mask4);
                __m256 wv      = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi)), bias));
                p = _mm256_fmadd_ps(wv, _mm256_loadu_ps(bx + j), p);
            }
            acc = _mm256_fmadd_ps(p, _mm256_set1_ps(rs[b]), acc);
        }
        total = hsum256(acc);
#else
        total = 0.0f;
        for (int b = 0; b < nbf; b++) {
            float v = 0.0f;
            for (int j = 0; j < TN_CLS_QUANT_BLOCK; j++) {
                int e      = b * TN_CLS_QUANT_BLOCK + j;
                int nibble = (e & 1) ? (row[e / 2] >> 4) & 0x0F : row[e / 2] & 0x0F;
                v += (float)(nibble - 8) * x[e];
            }
            total += v * rs[b];
        }
#endif
        if (nbf < n_blocks)
            total += tn_cls_i4_row_tail(row, x, nbf * TN_CLS_QUANT_BLOCK, n) * rs[nbf];
        out[i] = total;
    }
}
