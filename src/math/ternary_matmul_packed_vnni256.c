/* VNNI-256: 256-bit VNNI via AVX-512VNNI (EVEX encoding). No ZMM — no frequency throttle. */
#include "core/platform.h"

#if TN_HAS_AVX512VNNI

#include "math/ternary_matmul_packed.h"
#include "math/quantize_i8.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

/*
 * VNNI-256 (EVEX-256 via AVX-512VNNI) ternary matmul (Phase 16-S.3 — K-1b)
 *
 * Targets Alder Lake (12th Gen), Raptor Lake (13th Gen), Arrow Lake (14th Gen),
 * and AMD Zen 3 (Ryzen 5000 series) — CPUs with 256-bit VNNI (_mm256_dpbusds_epi32)
 * but WITHOUT AVX-512.  Without this kernel these CPUs fall back to the plain
 * float32 AVX2 kernel and get none of the int8 VNNI speedup.
 *
 * Algorithm: identical to the AVX-512 VNNI kernel but using 256-bit registers:
 *   - Processes 32 weights per VNNI call (vs 64 for AVX-512 VNNI)
 *   - Uses _mm256_dpbusds_epi32 (AVX-VNNI, requires -mavxvnni or -march=alderlake+)
 *   - Same w_enc bias-correction trick: dpbusds(w_enc_u8, q_x_i8) = dot+sum_qx
 *
 * Compile requirement: -mavxvnni (or -march=native on supporting CPUs).
 *   The Makefile sets -mavxvnni for this TU specifically; it does not affect
 *   other files and does not enable AVX-512.
 */

#define TN_AVXVNNI_MAX_N 16384
#define TN_PREFETCH_ROWS 8

/*
 * Unpack 8 packed 2-bit weights (2 bytes) into 8 uint8 w_enc values {0,1,2}
 * returned in a __m128i (low 8 bytes used).
 */
static inline __m128i unpack8_to_wenc_u8(const tn_u8 *row, int j)
{
    uint16_t bits;
    memcpy(&bits, row + (j >> 2), sizeof(bits));

    __m256i v      = _mm256_set1_epi32((int)bits);
    __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    v = _mm256_srlv_epi32(v, shifts);
    v = _mm256_and_si256(v, _mm256_set1_epi32(3));
    /* Pack int32 → int8 via int32→int16→int8 (values 0-2, no saturation) */
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i i16 = _mm_packs_epi32(lo, hi);          /* 8 int32 → 8 int16 */
    __m128i i8  = _mm_packs_epi16(i16, _mm_setzero_si128()); /* 8 int16 → 8 int8 */
    return i8;
}

void ternary_matmul_packed_vnni256(float *out, const float *x, const tn_u8 *packed_w,
                                     int n, int d, const float *scales, int group_size)
{
    if (n > TN_AVXVNNI_MAX_N) {
        /* Fallback to plain AVX2 float kernel */
        extern void ternary_matmul_packed_avx2(float*, const float*, const tn_u8*,
                                                int, int, const float*, int);
        ternary_matmul_packed_avx2(out, x, packed_w, n, d, scales, group_size);
        return;
    }

    /* Quantize activations once (reused for all d rows) */
    int8_t  q_x[TN_AVXVNNI_MAX_N];
    float   act_scale = quantize_row_to_i8_avx512(x, q_x, n);

    if (act_scale == 0.0f) {
        memset(out, 0, (size_t)d * sizeof(float));
        return;
    }

    int32_t sum_qx = sum_i8(q_x, n);

    size_t row_bytes = ((size_t)n + 3) >> 2;

    if (group_size <= 0) {
        float w_scale = scales[0];

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d) {
                _mm_prefetch((const char *)(packed_w +
                             (size_t)(i + TN_PREFETCH_ROWS) * row_bytes), _MM_HINT_T1);
            }

            __m256i acc = _mm256_setzero_si256();
            int j = 0;

            /* 32-weight AVX-VNNI loop */
            for (; j + 31 < n; j += 32) {
                __m128i w0 = unpack8_to_wenc_u8(row, j);
                __m128i w1 = unpack8_to_wenc_u8(row, j + 8);
                __m128i w2 = unpack8_to_wenc_u8(row, j + 16);
                __m128i w3 = unpack8_to_wenc_u8(row, j + 24);

                /* Combine 4 × 8-byte groups into 32-byte __m256i */
                __m128i wenc_lo = _mm_unpacklo_epi64(w0, w1);
                __m128i wenc_hi = _mm_unpacklo_epi64(w2, w3);
                __m256i wenc = _mm256_set_m128i(wenc_hi, wenc_lo);

                /* Load 32 int8 activations */
                __m256i qxv = _mm256_loadu_si256((const __m256i*)(q_x + j));

                /* AVX-VNNI: acc += Σ(wenc[k] * qxv[k]) in groups of 4 → 8 int32 */
                acc = _mm256_dpbusds_epi32(acc, wenc, qxv);
            }

            /* Horizontal sum of 8 int32 lanes */
            __m128i lo128 = _mm256_castsi256_si128(acc);
            __m128i hi128 = _mm256_extracti128_si256(acc, 1);
            __m128i sum4  = _mm_add_epi32(lo128, hi128);
            __m128i sum2  = _mm_hadd_epi32(sum4, sum4);
            __m128i sum1  = _mm_hadd_epi32(sum2, sum2);
            int32_t vnni_result = _mm_cvtsi128_si32(sum1);

            /* Scalar tail */
            for (; j < n; j++) {
                int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                vnni_result += w_enc * (int32_t)q_x[j];
            }

            int32_t true_dot = vnni_result - sum_qx;
            out[i] = (float)true_dot * act_scale * w_scale;
        }

    } else {
        int n_groups = (n + group_size - 1) / group_size;
        int safe_groups = n_groups < 1024 ? n_groups : 1024;

        int32_t sum_qx_group[1024];
        for (int g = 0; g < safe_groups; g++) {
            int gs = g * group_size;
            int ge = gs + group_size;
            if (ge > n) ge = n;
            sum_qx_group[g] = sum_i8(q_x + gs, ge - gs);
        }

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d) {
                _mm_prefetch((const char *)(packed_w +
                             (size_t)(i + TN_PREFETCH_ROWS) * row_bytes), _MM_HINT_T1);
            }

            float total = 0.0f;

            for (int g = 0; g < safe_groups; g++) {
                int gs = g * group_size;
                int ge = gs + group_size;
                if (ge > n) ge = n;

                __m256i acc = _mm256_setzero_si256();
                int j = gs;

                for (; j + 31 < ge; j += 32) {
                    __m128i w0 = unpack8_to_wenc_u8(row, j);
                    __m128i w1 = unpack8_to_wenc_u8(row, j + 8);
                    __m128i w2 = unpack8_to_wenc_u8(row, j + 16);
                    __m128i w3 = unpack8_to_wenc_u8(row, j + 24);
                    __m128i wlo = _mm_unpacklo_epi64(w0, w1);
                    __m128i whi = _mm_unpacklo_epi64(w2, w3);
                    __m256i wenc = _mm256_set_m128i(whi, wlo);
                    __m256i qxv = _mm256_loadu_si256((const __m256i*)(q_x + j));
                    acc = _mm256_dpbusds_epi32(acc, wenc, qxv);
                }

                __m128i lo128 = _mm256_castsi256_si128(acc);
                __m128i hi128 = _mm256_extracti128_si256(acc, 1);
                __m128i sum4  = _mm_add_epi32(lo128, hi128);
                __m128i sum2  = _mm_hadd_epi32(sum4, sum4);
                __m128i sum1  = _mm_hadd_epi32(sum2, sum2);
                int32_t vnni_result = _mm_cvtsi128_si32(sum1);

                for (; j < ge; j++) {
                    int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                    vnni_result += w_enc * (int32_t)q_x[j];
                }

                int32_t true_dot = vnni_result - sum_qx_group[g];
                total += (float)true_dot * act_scale * scales[i * n_groups + g];
            }

            out[i] = total;
        }
    }
}

#endif /* TN_HAS_AVX512VNNI */
