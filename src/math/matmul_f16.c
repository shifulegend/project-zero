/*
 * parallel_matmul_f16 — IEEE 754 F16 layer weight matmul.
 *
 * Reads F16 weights directly from mmap'd data (2 bytes/weight) and
 * converts to F32 on-the-fly during the dot product.  Halves bandwidth
 * vs the F32 path, matching llama.cpp's F16 inference bandwidth.
 *
 * SIMD strategy mirrors parallel_matmul_bf16 in parallel_matmul.c:
 *   AVX-512: 16-wide F16→F32 conversion + FMA, hardware rounding
 *   AVX2:     8-wide F16→F32 via _mm256_cvtph_ps + FMA
 *   Scalar:   element-wise f16_to_f32 (same function as gguf_loader.c)
 *
 * Prefetch note: No software prefetch is used.  When the model is warm in
 * the OS page cache (standard after one warmup run), the hardware prefetcher
 * handles sequential within-row access correctly.  Adding even one
 * _mm_prefetch per row adds instruction overhead that hurts performance
 * for the small row widths in SmolLM2-class models (n=576, 72 FMA iters
 * per row), where the extra instruction dominates over any latency hiding.
 */

#include "math/matmul_f16.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX512
#  include <immintrin.h>
#elif TN_HAS_AVX2
#  include <immintrin.h>
#endif

/* Scalar F16→F32 helper (mirrors f16_to_f32 in gguf_loader.c). */
static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

typedef struct {
    float       *out;
    const float *x;
    const tn_u16 *w;
    int n, d;
} MatmulF16Args;

static void matmul_f16_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulF16Args *a = (MatmulF16Args *)arg;

#if TN_HAS_AVX512
    /*
     * AVX-512: 16-wide F16→F32 conversion + FMA.
     * _mm512_cvtph_ps uses hardware IEEE 754 rounding — correct for F16.
     * Loads 16 × uint16_t (256 bits) then converts to 16 × float32 (512 bits).
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;

        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < a->n; j += 16) {
            __m256i f16_16 = _mm256_loadu_si256((const __m256i *)&row[j]);
            __m512  wv     = _mm512_cvtph_ps(f16_16);
            __m512  xv     = _mm512_loadu_ps(&a->x[j]);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        float val = _mm512_reduce_add_ps(acc);
        /* AVX2 tail (8-wide) */
        if (j + 7 < a->n) {
            __m128i f16_8 = _mm_loadu_si128((const __m128i *)&row[j]);
            __m256  wv8   = _mm256_cvtph_ps(f16_8);
            __m256  xv8   = _mm256_loadu_ps(&a->x[j]);
            __m256  p8    = _mm256_mul_ps(wv8, xv8);
            __m128  h = _mm256_extractf128_ps(p8, 1), l = _mm256_castps256_ps128(p8);
            __m128  s4 = _mm_add_ps(l, h);
            __m128  sh = _mm_movehdup_ps(s4);
            val += _mm_cvtss_f32(_mm_add_ss(_mm_add_ps(s4, sh),
                                             _mm_movehl_ps(sh, sh)));
            j += 8;
        }
        for (; j < a->n; j++) {
            val += f16_to_f32_scalar(row[j]) * a->x[j];
        }
        a->out[i] = val;
    }

#elif TN_HAS_AVX2
    /*
     * AVX2: 8-wide F16→F32 conversion + FMA.
     * _mm256_cvtph_ps is available as F16C extension (present on all AVX2 CPUs
     * that support F16C; we guard on TN_HAS_AVX2 as a proxy — if the target
     * doesn't have F16C the scalar fallback below is always safe).
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < a->n; j += 8) {
            __m128i f16_8 = _mm_loadu_si128((const __m128i *)&row[j]);
            __m256  wv    = _mm256_cvtph_ps(f16_8);
            __m256  xv    = _mm256_loadu_ps(&a->x[j]);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        __m128 hi   = _mm256_extractf128_ps(acc, 1);
        __m128 lo   = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        float  val  = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2)));
        for (; j < a->n; j++) {
            val += f16_to_f32_scalar(row[j]) * a->x[j];
        }
        a->out[i] = val;
    }

#else
    /* Scalar fallback */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        float acc = 0.0f;
        for (int j = 0; j < a->n; j++) {
            acc += f16_to_f32_scalar(row[j]) * a->x[j];
        }
        a->out[i] = acc;
    }
#endif
}

void parallel_matmul_f16(float *out, const float *x, const tn_u16 *w,
                          int n, int d, ThreadPool *tp) {
    MatmulF16Args args = { .out = out, .x = x, .w = w, .n = n, .d = d };
    if (!tp) {
        matmul_f16_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_f16_task, &args, d);
}
