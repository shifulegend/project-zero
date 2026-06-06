/*
 * matmul_q2k.c — Fused Q2_K × float32 matrix-vector product.
 *
 * Dequantizes Q2_K blocks on-the-fly during dot-product accumulation,
 * eliminating the intermediate float32 buffer and reducing DRAM traffic
 * by ~25× compared to the dequant-then-matmul approach.
 *
 * AVX2 fast path: processes 16 elements per inner loop iteration using
 * SIMD integer extract + FP32 FMA.  Falls back to scalar on non-AVX2
 * targets (e.g., ARM or older x86).
 */

#include "math/matmul_q2k.h"
#include "core/platform.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if TN_HAS_AVX2
#  include <immintrin.h>
#endif

/* ── Q2_K constants ─────────────────────────────────────────────────────────── */
#define Q2K_SUPER  256     /* elements per super-block */
#define Q2K_BYTES   84     /* bytes per super-block */

/* ── fp16 → f32 (self-contained; mirrors gguf_quant.c) ─────────────────────── */
static inline float q2k_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t u;
    if      (exp == 0x1fu) { u = sign | 0x7f800000u | (mant << 13); }
    else if (exp == 0u)    {
        if (mant == 0u) { u = sign; }
        else {
            exp = 1u;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            u = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else { u = sign | ((exp + 112u) << 23) | (mant << 13); }
    float f; memcpy(&f, &u, 4); return f;
}

/* ── AVX2 horizontal sum ────────────────────────────────────────────────────── */
#if TN_HAS_AVX2
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

/* ── Per-thread task ─────────────────────────────────────────────────────────── */
typedef struct {
    float         *out;
    const float   *inp;
    const uint8_t *w;
    int            in_dim;
    int            out_dim;
} Q2KMatvecArgs;

/*
 * Compute output rows [row_start, row_end) of the Q2K × float32 matvec.
 *
 * Inner structure mirrors gguf_dequant_q2_k() but accumulates a dot product
 * instead of writing decoded floats to a buffer.
 */
static void matvec_q2k_task(void *arg_v, int tid, int row_start, int row_end) {
    (void)tid;
    const Q2KMatvecArgs *a   = (const Q2KMatvecArgs *)arg_v;
    const float         *inp = a->inp;
    const uint8_t       *w   = a->w;
    float               *out = a->out;
    const int bpr             = a->in_dim >> 8;   /* blocks per row = in_dim / 256 */

    for (int o = row_start; o < row_end; o++) {
        const uint8_t *row_w = w + (size_t)o * (size_t)bpr * Q2K_BYTES;
        float sum = 0.0f;

        for (int b = 0; b < bpr; b++) {
            const uint8_t *blk = row_w + b * Q2K_BYTES;
            const uint8_t *sc  = blk;        /* scales[16] at offset 0  */
            const uint8_t *qs  = blk + 16;   /* qs[64]     at offset 16 */
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits,    blk + 80, 2);
            memcpy(&dmin_bits, blk + 82, 2);
            const float d    = q2k_fp16_to_f32(d_bits);
            const float dmin = q2k_fp16_to_f32(dmin_bits);
            if (!(d == d) || !(dmin == dmin)) continue;  /* NaN guard: skip corrupt block */

            const float *inp_b = inp + b * Q2K_SUPER; /* inp[b*256 .. (b+1)*256) */
            int is = 0;

            /* Two 128-element halves: n = 0, 128 */
            for (int half = 0; half < 2; half++) {
                const uint8_t *q    = qs    + half * 32;
                const float   *inph = inp_b + half * 128;
                int shift = 0;

                /* Four shift-groups per half, each covering 32 elements */
                for (int j = 0; j < 4; j++) {
                    const float dl  = d    * (float)(sc[is]   & 0xF);
                    const float ml  = dmin * (float)(sc[is++] >> 4);
                    const float dl2 = d    * (float)(sc[is]   & 0xF);
                    const float ml2 = dmin * (float)(sc[is++] >> 4);

                    const float *inp0 = inph + j * 32;       /* first  16 inputs */
                    const float *inp1 = inph + j * 32 + 16;  /* second 16 inputs */

#if TN_HAS_AVX2
                    /* ── AVX2 fast path: process 16 elements in two 8-wide rounds ── */
                    __m128i vq0 = _mm_loadu_si128((const __m128i *)q);       /* q[0..15]  */
                    __m128i vq1 = _mm_loadu_si128((const __m128i *)(q + 16));/* q[16..31] */
                    const __m128i mask3 = _mm_set1_epi16(3);

                    /* first 16 elements use q[0..15], scale (dl, ml) */
                    __m128i vq0lo = _mm_unpacklo_epi8(vq0, _mm_setzero_si128()); /* q[0..7]  as i16 */
                    __m128i vq0hi = _mm_unpackhi_epi8(vq0, _mm_setzero_si128()); /* q[8..15] as i16 */
                    __m256 vwf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                                    _mm_and_si128(_mm_srli_epi16(vq0lo, shift), mask3)));
                    __m256 vwf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                                    _mm_and_si128(_mm_srli_epi16(vq0hi, shift), mask3)));
                    __m256 vinp0a = _mm256_loadu_ps(inp0);
                    __m256 vinp0b = _mm256_loadu_ps(inp0 + 8);
                    float wdot0 = hsum256_ps(_mm256_fmadd_ps(vwf0, vinp0a,
                                              _mm256_mul_ps(vwf1, vinp0b)));
                    float isum0 = hsum256_ps(_mm256_add_ps(vinp0a, vinp0b));
                    sum += dl * wdot0 - ml * isum0;

                    /* second 16 elements use q[16..31], scale (dl2, ml2) */
                    __m128i vq1lo = _mm_unpacklo_epi8(vq1, _mm_setzero_si128()); /* q[16..23] as i16 */
                    __m128i vq1hi = _mm_unpackhi_epi8(vq1, _mm_setzero_si128()); /* q[24..31] as i16 */
                    __m256 vwf2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                                    _mm_and_si128(_mm_srli_epi16(vq1lo, shift), mask3)));
                    __m256 vwf3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                                    _mm_and_si128(_mm_srli_epi16(vq1hi, shift), mask3)));
                    __m256 vinp1a = _mm256_loadu_ps(inp1);
                    __m256 vinp1b = _mm256_loadu_ps(inp1 + 8);
                    float wdot1 = hsum256_ps(_mm256_fmadd_ps(vwf2, vinp1a,
                                              _mm256_mul_ps(vwf3, vinp1b)));
                    float isum1 = hsum256_ps(_mm256_add_ps(vinp1a, vinp1b));
                    sum += dl2 * wdot1 - ml2 * isum1;
#else
                    /* ── Scalar fallback ─────────────────────────────────────── */
                    float wdot0 = 0.0f, isum0 = 0.0f;
                    float wdot1 = 0.0f, isum1 = 0.0f;
                    for (int l = 0; l < 16; l++) {
                        float qi0 = (float)((q[l]      >> shift) & 3);
                        float qi1 = (float)((q[l + 16] >> shift) & 3);
                        wdot0 += qi0 * inp0[l];
                        isum0 += inp0[l];
                        wdot1 += qi1 * inp1[l];
                        isum1 += inp1[l];
                    }
                    sum += dl * wdot0 - ml * isum0;
                    sum += dl2 * wdot1 - ml2 * isum1;
#endif
                    shift += 2;
                } /* j */
            } /* half */
        } /* b */

        out[o] = sum;
    } /* o */
}

/* ── Public API ─────────────────────────────────────────────────────────────── */
void parallel_matvec_q2k(float *out, const float *inp, const uint8_t *w,
                          int in_dim, int out_dim, ThreadPool *tp) {
    Q2KMatvecArgs args = { out, inp, w, in_dim, out_dim };
    if (!tp) {
        matvec_q2k_task(&args, 0, 0, out_dim);
        return;
    }
    threadpool_dispatch(tp, matvec_q2k_task, &args, out_dim);
}
