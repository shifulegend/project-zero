/*
 * parallel_matmul_q8_0 — fused Q8_0 × F32 activation matmul (gemv).
 *
 * Decodes Q8_0 blocks on-the-fly during the dot product, eliminating the
 * dequant-entire-matrix-to-F32 overhead (dominates for expert down projections
 * in DeepSeek-V2-Lite layers 1,2,5,8,11,14,17,20,23,24,25,26 which use Q8_0).
 *
 * Q8_0 block layout (34 bytes / 32 elements):
 *   [d:  fp16   2B] — scale (F16, big-endian)
 *   [qs: int8[32] ] — 32 signed 8-bit integers in [-127, 127]
 *
 * Decode: val_i = (float)qs[i] * d
 * Dot product: d * sum_i(qs[i] * x_i)
 *
 * Bandwidth: 34 bytes / 32 elements = 1.0625 bytes/elem (vs 4 bytes/elem F32)
 * Expected ~3-4× bandwidth reduction vs dequant fallback path.
 */

#include "math/matmul_q8_0.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX2 || TN_HAS_AVX512
#  include <immintrin.h>
#endif

/* ── Q8_0 constants ──────────────────────────────────────────────────────── */
#define Q8_0_BLOCK   32
#define Q8_0_BYTES   34   /* 2B scale + 32B int8 weights */

static inline float q8_0_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112u) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

/*
 * dot_q8_0_row: fused dot product of one Q8_0-encoded row with float32 x.
 * row_q8_0 : pointer to first byte of this row's Q8_0 data (n/32 × 34 bytes)
 * x         : float32 activation vector, length n
 * n         : number of elements (must be multiple of Q8_0_BLOCK=32)
 */
static float dot_q8_0_row(const uint8_t *row_q8_0, const float *x, int n) {
    int n_blocks = n / Q8_0_BLOCK;

#if TN_HAS_AVX512
    /*
     * AVX-512: process all 32 elements per block in two passes of 16.
     * Accumulate into a single __m512 across ALL blocks, reduce ONCE at end.
     * This avoids 64× _mm512_reduce_add_ps calls (the original per-block
     * reduce caused ~4× throughput regression from instruction overhead).
     *
     * d is scalar per block; multiply into running float accumulator:
     *   float_acc += d * (cvt_f32(qs[0..15]) * x[0..15])
     *              + d * (cvt_f32(qs[16..31]) * x[16..31])
     */
    __m512 float_acc = _mm512_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q8_0 + (size_t)b * Q8_0_BYTES;
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        __m512 dv = _mm512_set1_ps(q8_0_f16_to_f32(d_bits));
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float  *xb = x + (size_t)b * Q8_0_BLOCK;

        __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)qs)));
        float_acc = _mm512_fmadd_ps(_mm512_mul_ps(dv, f0), _mm512_loadu_ps(xb), float_acc);

        __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(qs + 16))));
        float_acc = _mm512_fmadd_ps(_mm512_mul_ps(dv, f1), _mm512_loadu_ps(xb + 16), float_acc);
    }
    return _mm512_reduce_add_ps(float_acc);

#elif TN_HAS_AVX2
    /*
     * AVX2: process 32 elements per block in four passes of 8.
     * Single __m256 accumulator across all blocks, reduced once at end.
     */
    __m256 float_acc = _mm256_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q8_0 + (size_t)b * Q8_0_BYTES;
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        __m256 dv = _mm256_set1_ps(q8_0_f16_to_f32(d_bits));
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float  *xb = x + (size_t)b * Q8_0_BLOCK;
        for (int j = 0; j < 32; j += 8) {
            __m256 fv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_loadl_epi64((const __m128i *)(qs + j))));
            float_acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, fv),
                                         _mm256_loadu_ps(xb + j), float_acc);
        }
    }
    /* Single horizontal sum across 8 lanes */
    __m128 lo = _mm256_castps256_ps128(float_acc);
    __m128 hi = _mm256_extractf128_ps(float_acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));

#else
    /* Scalar fallback */
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q8_0 + (size_t)b * Q8_0_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = q8_0_f16_to_f32(d_bits);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float  *xb = x + (size_t)b * Q8_0_BLOCK;
        float sum = 0.0f;
        for (int i = 0; i < Q8_0_BLOCK; i++) sum += (float)qs[i] * xb[i];
        dot += d * sum;
    }
    return dot;
#endif
}

/* ── Single-matrix thread pool dispatch ─────────────────────────────────── */

typedef struct {
    float         *out;
    const float   *x;
    const uint8_t *w;
    int n, d;
    size_t row_bytes;
} MatmulQ8_0Args;

static void matmul_q8_0_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ8_0Args *a = (MatmulQ8_0Args *)arg;
    for (int i = start; i < end; i++)
        a->out[i] = dot_q8_0_row(a->w + (size_t)i * a->row_bytes, a->x, a->n);
}

void parallel_matmul_q8_0(float *out, const float *x, const uint8_t *w_q8_0,
                           int n, int d, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q8_0_BLOCK) * Q8_0_BYTES;
    MatmulQ8_0Args args = { .out = out, .x = x, .w = w_q8_0,
                             .n = n, .d = d, .row_bytes = row_bytes };
    if (!tp) { matmul_q8_0_task(&args, 0, 0, d); return; }
    threadpool_dispatch(tp, matmul_q8_0_task, &args, d);
}

/* ── Batched: k weight matrices, per-expert inputs ───────────────────────── */

typedef struct {
    float        **outs;
    float        **xs;
    const uint8_t *const *ws;
    int n_in, n_out, k;
    size_t row_bytes;
} MatmulQ8_0BatchArgs;

static void matmul_q8_0_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ8_0BatchArgs *a = (MatmulQ8_0BatchArgs *)arg;
    int n_out = a->n_out;
    for (int r = start; r < end; r++) {
        int ei = r / n_out;
        int ri = r % n_out;
        a->outs[ei][ri] = dot_q8_0_row(a->ws[ei] + (size_t)ri * a->row_bytes,
                                        a->xs[ei], a->n_in);
    }
}

void parallel_matmul_q8_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q8_0_BLOCK) * Q8_0_BYTES;
    MatmulQ8_0BatchArgs args = {
        .outs = outs, .xs = xs, .ws = ws,
        .n_in = n, .n_out = d, .k = k,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q8_0_batch_task(&args, 0, 0, k * d); return; }
    threadpool_dispatch(tp, matmul_q8_0_batch_task, &args, k * d);
}
