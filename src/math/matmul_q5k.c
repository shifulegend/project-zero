/*
 * parallel_matmul_q5k — fused Q5_K × F32 activation matmul (gemv).
 *
 * Decodes Q5_K super-blocks on-the-fly, eliminating the large F32 buffer.
 * Reuses the same scale/min decode as Q4_K (identical 12-byte packed format).
 * Additional 5th bit per element comes from the 32-byte qh field.
 */

#include "math/matmul_q5k.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX512 || TN_HAS_AVX2
#  include <immintrin.h>
#endif

/* ── Q5_K constants ──────────────────────────────────────────────────────── */
#define Q5K_SUPER  256
#define Q5K_NSUB   8
#define Q5K_BYTES  176   /* bytes per 256-element super-block */

static inline float q5k_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

/* Decode 8 (scale, min) pairs — identical to Q4_K. */
static inline void q5k_decode_scales(const uint8_t *sc12, float d, float dmin,
                                      float *scales, float *mins) {
    for (int j = 0; j < Q5K_NSUB; j++) {
        uint8_t sc, m;
        if (j < 4) {
            sc = sc12[j]     & 63;
            m  = sc12[j + 4] & 63;
        } else {
            int k = j - 4;
            sc = (sc12[k + 8] & 0x0F) | ((sc12[k]     >> 6) << 4);
            m  = (sc12[k + 8] >>    4) | ((sc12[k + 4] >> 6) << 4);
        }
        scales[j] = sc * d;
        mins[j]   = m  * dmin;
    }
}

/*
 * dot_q5k_row: fused dot product of one Q5_K-encoded row with float32 x.
 *
 * Mirrors the loop structure of gguf_dequant_q5_k():
 *   4 groups × 64 elements each.
 *   Group g uses: ql[g*32..], qh[0..31] bits (2g) and (2g+1), scales[2g], mins[2g..2g+1].
 *   5th bit for lo half: (qh[l] & u1) != 0, u1 = 1<<(2g)
 *   5th bit for hi half: (qh[l] & u2) != 0, u2 = 1<<(2g+1)
 */
static float dot_q5k_row(const uint8_t *row_q5k, const float *x, int n) {
    int n_super = n / Q5K_SUPER;
    float dot = 0.0f;

    for (int b = 0; b < n_super; b++) {
        const uint8_t *blk   = row_q5k + (size_t)b * Q5K_BYTES;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d    = q5k_f16_to_f32(d_bits);
        float dmin = q5k_f16_to_f32(dmin_bits);

        const uint8_t *sc12 = blk + 4;   /* 12 bytes */
        const uint8_t *qh   = blk + 16;  /* 32 bytes */
        const uint8_t *ql   = blk + 48;  /* 128 bytes */

        float scales[Q5K_NSUB], mins[Q5K_NSUB];
        q5k_decode_scales(sc12, d, dmin, scales, mins);

        const float *xb = x + (size_t)b * Q5K_SUPER;

#if TN_HAS_AVX512
        /*
         * AVX-512 path: scalar-decode 32 5-bit integers per half-group into
         * a stack int32[32] array, then 16-wide FMA.  Same strategy as Q5_1.
         * Per group: 32 lo elements + 32 hi elements, each needing the qh 5th bit.
         */
        {
            uint8_t u1 = 1, u2 = 2;  /* 5th-bit masks, advance << 2 per group */
            for (int g = 0; g < 4; g++) {
                float d1 = scales[g * 2],     m1 = mins[g * 2];
                float d2 = scales[g * 2 + 1], m2 = mins[g * 2 + 1];
                const uint8_t *q = ql + g * 32;
                const float *xl  = xb + g * 64;
                const float *xh  = xb + g * 64 + 32;

                int32_t vals_lo[32], vals_hi[32];
                for (int l = 0; l < 32; l++) {
                    vals_lo[l] = (q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                    vals_hi[l] = (q[l] >>   4) + ((qh[l] & u2) ? 16 : 0);
                }

                __m512i iv_lo0 = _mm512_loadu_si512((const void *)vals_lo);
                __m512i iv_lo1 = _mm512_loadu_si512((const void *)(vals_lo + 16));
                __m512  fv_lo0 = _mm512_cvtepi32_ps(iv_lo0);
                __m512  fv_lo1 = _mm512_cvtepi32_ps(iv_lo1);

                __m512i iv_hi0 = _mm512_loadu_si512((const void *)vals_hi);
                __m512i iv_hi1 = _mm512_loadu_si512((const void *)(vals_hi + 16));
                __m512  fv_hi0 = _mm512_cvtepi32_ps(iv_hi0);
                __m512  fv_hi1 = _mm512_cvtepi32_ps(iv_hi1);

                __m512 xvl0 = _mm512_loadu_ps(xl);
                __m512 xvl1 = _mm512_loadu_ps(xl + 16);
                __m512 xvh0 = _mm512_loadu_ps(xh);
                __m512 xvh1 = _mm512_loadu_ps(xh + 16);

                /* dot_lo = sum(vals_lo[i] * xl[i]), sum_xl = sum(xl[i]) */
                __m512 acc_lo = _mm512_fmadd_ps(fv_lo0, xvl0, _mm512_mul_ps(fv_lo1, xvl1));
                __m512 sum_xl = _mm512_add_ps(xvl0, xvl1);
                __m512 acc_hi = _mm512_fmadd_ps(fv_hi0, xvh0, _mm512_mul_ps(fv_hi1, xvh1));
                __m512 sum_xh = _mm512_add_ps(xvh0, xvh1);

                float dot_lo = _mm512_reduce_add_ps(acc_lo);
                float dot_hi = _mm512_reduce_add_ps(acc_hi);
                float sx_lo  = _mm512_reduce_add_ps(sum_xl);
                float sx_hi  = _mm512_reduce_add_ps(sum_xh);

                dot += d1 * dot_lo - m1 * sx_lo + d2 * dot_hi - m2 * sx_hi;

                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
#elif TN_HAS_AVX2
        {
            uint8_t u1 = 1, u2 = 2;
            for (int g = 0; g < 4; g++) {
                float d1 = scales[g * 2],     m1 = mins[g * 2];
                float d2 = scales[g * 2 + 1], m2 = mins[g * 2 + 1];
                const uint8_t *q = ql + g * 32;
                const float *xl  = xb + g * 64;
                const float *xh  = xb + g * 64 + 32;

                int32_t vals_lo[32], vals_hi[32];
                for (int l = 0; l < 32; l++) {
                    vals_lo[l] = (q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                    vals_hi[l] = (q[l] >>   4) + ((qh[l] & u2) ? 16 : 0);
                }

                __m256 acc_lo = _mm256_setzero_ps();
                __m256 acc_hi = _mm256_setzero_ps();
                __m256 sum_xl = _mm256_setzero_ps();
                __m256 sum_xh = _mm256_setzero_ps();
                for (int j = 0; j < 32; j += 8) {
                    __m256i ivl = _mm256_loadu_si256((const __m256i *)(vals_lo + j));
                    __m256i ivh = _mm256_loadu_si256((const __m256i *)(vals_hi + j));
                    __m256  fvl = _mm256_cvtepi32_ps(ivl);
                    __m256  fvh = _mm256_cvtepi32_ps(ivh);
                    __m256  xvl = _mm256_loadu_ps(xl + j);
                    __m256  xvh = _mm256_loadu_ps(xh + j);
                    acc_lo = _mm256_fmadd_ps(fvl, xvl, acc_lo);
                    acc_hi = _mm256_fmadd_ps(fvh, xvh, acc_hi);
                    sum_xl = _mm256_add_ps(sum_xl, xvl);
                    sum_xh = _mm256_add_ps(sum_xh, xvh);
                }

                __m128 a4l = _mm_add_ps(_mm256_castps256_ps128(acc_lo), _mm256_extractf128_ps(acc_lo, 1));
                __m128 a4h = _mm_add_ps(_mm256_castps256_ps128(acc_hi), _mm256_extractf128_ps(acc_hi, 1));
                __m128 s4l = _mm_add_ps(_mm256_castps256_ps128(sum_xl), _mm256_extractf128_ps(sum_xl, 1));
                __m128 s4h = _mm_add_ps(_mm256_castps256_ps128(sum_xh), _mm256_extractf128_ps(sum_xh, 1));
                a4l = _mm_add_ps(a4l, _mm_movehl_ps(a4l, a4l)); a4l = _mm_add_ss(a4l, _mm_movehdup_ps(a4l));
                a4h = _mm_add_ps(a4h, _mm_movehl_ps(a4h, a4h)); a4h = _mm_add_ss(a4h, _mm_movehdup_ps(a4h));
                s4l = _mm_add_ps(s4l, _mm_movehl_ps(s4l, s4l)); s4l = _mm_add_ss(s4l, _mm_movehdup_ps(s4l));
                s4h = _mm_add_ps(s4h, _mm_movehl_ps(s4h, s4h)); s4h = _mm_add_ss(s4h, _mm_movehdup_ps(s4h));

                dot += d1 * _mm_cvtss_f32(a4l) - m1 * _mm_cvtss_f32(s4l)
                     + d2 * _mm_cvtss_f32(a4h) - m2 * _mm_cvtss_f32(s4h);

                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
#else
        /* Scalar fallback: mirrors gguf_dequant_q5_k exactly */
        {
            uint8_t u1 = 1, u2 = 2;
            for (int g = 0; g < 4; g++) {
                float d1 = scales[g * 2],     m1 = mins[g * 2];
                float d2 = scales[g * 2 + 1], m2 = mins[g * 2 + 1];
                const uint8_t *q = ql + g * 32;
                const float *xl  = xb + g * 64;
                const float *xh  = xb + g * 64 + 32;
                for (int l = 0; l < 32; l++) {
                    float v0 = d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    float v1 = d2 * (float)((q[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                    dot += v0 * xl[l] + v1 * xh[l];
                }
                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
#endif
    }
    return dot;
}

/* ── Thread pool dispatch ─────────────────────────────────────────────────── */

typedef struct {
    float         *out;
    const float   *x;
    const uint8_t *w;
    int n, d;
    size_t row_bytes;
} MatmulQ5KArgs;

static void matmul_q5k_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ5KArgs *a = (MatmulQ5KArgs *)arg;
    for (int i = start; i < end; i++) {
        a->out[i] = dot_q5k_row(a->w + (size_t)i * a->row_bytes, a->x, a->n);
    }
}

void parallel_matmul_q5k(float *out, const float *x, const uint8_t *w_q5k,
                          int n, int d, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q5K_SUPER) * Q5K_BYTES;
    MatmulQ5KArgs args = {
        .out       = out,
        .x         = x,
        .w         = w_q5k,
        .n         = n,
        .d         = d,
        .row_bytes = row_bytes,
    };
    if (!tp) {
        matmul_q5k_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_q5k_task, &args, d);
}
