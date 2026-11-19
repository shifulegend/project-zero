/*
 * cls_kernels_avx512.c — AVX-512F INT8 / INT4 classifier row kernels.
 *
 * Built with the AVX-512 flags in every x86 build (Makefile AVX512_TUS, CMake PZ_AVX512_SRCS) and bound at
 * runtime from simd_dispatch.c, so the portable dist binary has them (docs/ai/engineering-rules.md, ISA
 * segregation). On a toolchain without AVX-512 the same names forward to the generic kernels (never bound there).
 * Contract and layout: include/math/cls_kernels.h. One horizontal reduction per row.
 */
#include <stdlib.h>

#include "math/cls_kernels.h"

#if TN_HAS_AVX512
#include <immintrin.h>

void tn_cls_i8_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end) {
    const int nbf      = n / TN_CLS_QUANT_BLOCK;
    const __m512i bias = _mm512_set1_epi32(128);
    for (int i = start; i < end; i++) {
        const tn_u8 *row = w + (size_t)i * n;
        const float *rs  = scales + (size_t)i * n_blocks;
        __m512 acc       = _mm512_setzero_ps();
        for (int b = 0; b < nbf; b++) {
            const tn_u8 *bw = row + (size_t)b * TN_CLS_QUANT_BLOCK;
            const float *bx = x + (size_t)b * TN_CLS_QUANT_BLOCK;
            __m512 w0       = _mm512_cvtepi32_ps(
                _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)bw)), bias));
            __m512 w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(bw + 16))), bias));
            __m512 p  = _mm512_fmadd_ps(w1, _mm512_loadu_ps(bx + 16),
                                        _mm512_mul_ps(w0, _mm512_loadu_ps(bx)));
            acc       = _mm512_fmadd_ps(p, _mm512_set1_ps(rs[b]), acc);
        }
        float total = _mm512_reduce_add_ps(acc);
        if (nbf < n_blocks)
            total += tn_cls_i8_row_tail(row, x, nbf * TN_CLS_QUANT_BLOCK, n) * rs[nbf];
        out[i] = total;
    }
}

/* INT4: a block's 16 bytes hold elements (2k, 2k+1) in (low, high) nibbles. Instead of unpacking every weight
 * block into element order (6 shuffles and masks per block), the activation is de-interleaved ONCE per call into
 * even/odd halves per block, so a weight block needs only: widen 16 bytes to 16 lanes, mask (low nibble = even
 * elements), shift (high nibble = odd elements). The +8 code bias is folded into one correction per row:
 *   sum_b s_b * sum_j (q_j - 8) x_j = sum_b s_b * sum_j q_j x_j - 8 * sum_b s_b * sx_b,   sx_b = sum of block b of x. */
void tn_cls_i4_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end) {
    const int nbf          = n / TN_CLS_QUANT_BLOCK;
    const size_t row_bytes = ((size_t)n + 1) / 2;
    float *buf =
        nbf ? (float *)malloc(sizeof(float) * (size_t)nbf * (TN_CLS_QUANT_BLOCK + 1)) : NULL;
    if (nbf && !buf) { /* out of memory: slower but correct */
        tn_cls_i4_rows_generic(out, x, w, scales, n, n_blocks, start, end);
        return;
    }
    float *sx =
        buf + (size_t)nbf * TN_CLS_QUANT_BLOCK; /* per-block sums follow the even/odd halves */
    const __m512i idx_even =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_odd =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    for (int b = 0; b < nbf; b++) {
        __m512 x0 = _mm512_loadu_ps(x + (size_t)b * TN_CLS_QUANT_BLOCK);
        __m512 x1 = _mm512_loadu_ps(x + (size_t)b * TN_CLS_QUANT_BLOCK + 16);
        _mm512_storeu_ps(buf + (size_t)b * TN_CLS_QUANT_BLOCK,
                         _mm512_permutex2var_ps(x0, idx_even, x1));
        _mm512_storeu_ps(buf + (size_t)b * TN_CLS_QUANT_BLOCK + 16,
                         _mm512_permutex2var_ps(x0, idx_odd, x1));
        sx[b] = _mm512_reduce_add_ps(_mm512_add_ps(x0, x1));
    }
    const __m512i mask4 = _mm512_set1_epi32(0x0F);
    for (int i = start; i < end; i++) {
        const tn_u8 *row = w + (size_t)i * row_bytes;
        const float *rs  = scales + (size_t)i * n_blocks;
        __m512 acc       = _mm512_setzero_ps();
        for (int b = 0; b < nbf; b++) {
            __m512i v = _mm512_cvtepu8_epi32(
                _mm_loadu_si128((const __m128i *)(row + (size_t)b * (TN_CLS_QUANT_BLOCK / 2))));
            __m512 wlo = _mm512_cvtepi32_ps(_mm512_and_si512(v, mask4)); /* elements 0, 2, ... 30 */
            __m512 whi = _mm512_cvtepi32_ps(_mm512_srli_epi32(v, 4));    /* elements 1, 3, ... 31 */
            const float *bx = buf + (size_t)b * TN_CLS_QUANT_BLOCK;
            __m512 p        = _mm512_fmadd_ps(whi, _mm512_loadu_ps(bx + 16),
                                              _mm512_mul_ps(wlo, _mm512_loadu_ps(bx)));
            acc             = _mm512_fmadd_ps(p, _mm512_set1_ps(rs[b]), acc);
        }
        __m512 corr = _mm512_setzero_ps(); /* sum_b s_b * sx_b */
        int b       = 0;
        for (; b + 15 < nbf; b += 16)
            corr = _mm512_fmadd_ps(_mm512_loadu_ps(rs + b), _mm512_loadu_ps(sx + b), corr);
        float c = _mm512_reduce_add_ps(corr);
        for (; b < nbf; b++) c += rs[b] * sx[b];
        float total = _mm512_reduce_add_ps(acc) - 8.0f * c;
        if (nbf < n_blocks)
            total += tn_cls_i4_row_tail(row, x, nbf * TN_CLS_QUANT_BLOCK, n) * rs[nbf];
        out[i] = total;
    }
    free(buf);
}

#else /* !TN_HAS_AVX512 */

void tn_cls_i8_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end) {
    tn_cls_i8_rows_generic(out, x, w, scales, n, n_blocks, start, end);
}

void tn_cls_i4_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end) {
    tn_cls_i4_rows_generic(out, x, w, scales, n, n_blocks, start, end);
}

#endif
