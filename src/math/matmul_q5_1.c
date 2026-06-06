/*
 * parallel_matmul_q5_1 — fused Q5_1 × F32 activation matmul (gemv).
 *
 * Decodes Q5_1 blocks on-the-fly during the dot product, eliminating the
 * ~68 b/w bandwidth of dequant-entire-matrix-to-F32 + read-F32 again.
 * New bandwidth: ~6 b/w (read Q5_1 data only, no intermediate buffer).
 *
 * Strategy (AVX2 path, used on both AVX2 and AVX-512 machines):
 *   For each output row, iterate over n/32 blocks of 32 elements.
 *   Per block: scalar-decode 32 5-bit integers into a float[32] staging array
 *   (avoids int32→float type-pun store-to-load stall), then four 8-wide FMA
 *   passes accumulate (d*u + m)*x into a single __m256 across all blocks.
 *   ONE hsum at the end — eliminates per-block horizontal reduces.
 */

#include "math/matmul_q5_1.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX512 || TN_HAS_AVX2
#  include <immintrin.h>
#endif

/* ── Q5_1 constants ──────────────────────────────────────────────────────── */
#define Q5_1_BLOCK   32
#define Q5_1_BYTES   24   /* bytes per 32-element block */

static inline float q5_1_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

/*
 * dot_q5_1_row: fused dot product of one Q5_1-encoded row with float32 x.
 *
 * row_q5_1 : pointer to first byte of this row's Q5_1 data
 *             (n/32 blocks × 24 bytes each)
 * x         : float32 activation vector, length n
 * n         : number of elements (must be multiple of Q5_1_BLOCK=32)
 */
/*
 * dot_q5_1_row: fused dot product of one Q5_1-encoded row with float32 x.
 *
 * Q5_1 decode: decoded[i] = d * u[i] + m  (u = unsigned 5-bit, 0..31)
 * Dot product: sum((d*u[i] + m) * x[i]) accumulated across all blocks.
 *
 * SIMD decode: same llama.cpp-style 256-bit bytes_from_bits_32 trick as Q5_0,
 * but u5 values stay unsigned (0..31) — no -16 offset.  Bit placement uses
 * 0x10 flag rather than 0xF0, then cvtepu8 (zero-extend, not sign-extend).
 * Accumulated decode: dot_acc += (d*u5 + m) * x  in a single __m256.
 */
static float dot_q5_1_row(const uint8_t *row_q5_1, const float *x, int n) {
    int n_blocks = n / Q5_1_BLOCK;

#if TN_HAS_AVX2
    static const int8_t shuf256_data[32] = {
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    };
    const __m256i shuf256  = _mm256_loadu_si256((const __m256i *)shuf256_data);
    const __m256i bit_mask = _mm256_set1_epi64x((int64_t)0x7fbfdfeff7fbfdfeULL);
    const __m256i hi_flag  = _mm256_set1_epi8(0x10); /* bit 4 = high bit of Q5_1 */
    const __m256i nib_mask = _mm256_set1_epi8(0x0F);
    const __m256i all_ones = _mm256_set1_epi8((char)0xFF);

    /* AVX2 path only — AVX-512 triggered frequency throttle (+2ms/call).
     * Bit expansion uses 256-bit shuffle; float FMA stays 8-wide. */
    __m256 dot_acc = _mm256_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q5_1 + (size_t)b * Q5_1_BYTES;

        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        __m256 dv = _mm256_set1_ps(q5_1_f16_to_f32(d_bits));
        __m256 mv = _mm256_set1_ps(q5_1_f16_to_f32(m_bits));

        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t *qs = blk + 8;
        const float   *xb = x + (size_t)b * Q5_1_BLOCK;

        /* Unpack nibbles: lower 128 = low nibbles (elems 0-15),
         *                 upper 128 = high nibbles (elems 16-31) */
        const __m128i tmp128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i nibbles = _mm256_and_si256(
            nib_mask,
            _mm256_set_m128i(_mm_srli_epi16(tmp128, 4), tmp128));

        /* Spread qh bits → 0xFF where set, 0x00 where clear */
        __m256i qh_spread = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf256);
        __m256i bit_set   = _mm256_cmpeq_epi8(_mm256_or_si256(qh_spread, bit_mask), all_ones);

        /* 0x10 where qh bit is set, 0x00 where clear (unsigned 5th bit) */
        __m256i hi_contrib = _mm256_and_si256(bit_set, hi_flag);

        /* u5 unsigned (0..31) as uint8 */
        __m256i u5 = _mm256_or_si256(nibbles, hi_contrib);

        /* Zero-extend uint8 → int32 → float, 4 groups of 8 */
        __m128i lo128 = _mm256_castsi256_si128(u5);
        __m128i hi128 = _mm256_extracti128_si256(u5, 1);
        __m256 fv0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo128));
        __m256 fv1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo128, 8)));
        __m256 fv2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi128));
        __m256 fv3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi128, 8)));

        /* dot_acc += (d*u + m) * x */
        dot_acc = _mm256_fmadd_ps(_mm256_fmadd_ps(dv, fv0, mv), _mm256_loadu_ps(xb),      dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_fmadd_ps(dv, fv1, mv), _mm256_loadu_ps(xb +  8), dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_fmadd_ps(dv, fv2, mv), _mm256_loadu_ps(xb + 16), dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_fmadd_ps(dv, fv3, mv), _mm256_loadu_ps(xb + 24), dot_acc);
    }

    __m128 lo = _mm256_castps256_ps128(dot_acc);
    __m128 hi = _mm256_extractf128_ps(dot_acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));

#else
    /* Scalar fallback */
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q5_1 + (size_t)b * Q5_1_BYTES;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        float d = q5_1_f16_to_f32(d_bits);
        float m = q5_1_f16_to_f32(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t *qs = blk + 8;
        const float   *xb = x + (size_t)b * Q5_1_BLOCK;
        for (int i = 0; i < 16; i++) {
            float q0 = (float)((qs[i] & 0xF) | (((qh >>  i)       & 1) << 4));
            float q1 = (float)((qs[i] >>   4) | (((qh >> (i + 16)) & 1) << 4));
            dot += (d * q0 + m) * xb[i] + (d * q1 + m) * xb[i + 16];
        }
    }
    return dot;
#endif
}

/* ── Thread pool dispatch ─────────────────────────────────────────────────── */

typedef struct {
    float         *out;
    const float   *x;
    const uint8_t *w;   /* Q5_1 weight matrix: d rows, each (n/32)*24 bytes */
    int n, d;
    size_t row_bytes;
} MatmulQ5_1Args;

static void matmul_q5_1_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ5_1Args *a = (MatmulQ5_1Args *)arg;
    for (int i = start; i < end; i++) {
        a->out[i] = dot_q5_1_row(a->w + (size_t)i * a->row_bytes, a->x, a->n);
    }
}

void parallel_matmul_q5_1(float *out, const float *x, const uint8_t *w_q5_1,
                           int n, int d, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q5_1_BLOCK) * Q5_1_BYTES;
    MatmulQ5_1Args args = {
        .out       = out,
        .x         = x,
        .w         = w_q5_1,
        .n         = n,
        .d         = d,
        .row_bytes = row_bytes,
    };
    if (!tp) {
        matmul_q5_1_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_q5_1_task, &args, d);
}

/* ── Batched Q5_1: k weight matrices, per-expert inputs ─────────────────── */

typedef struct {
    float         **outs;
    float         **xs;       /* per-expert inputs (post-SiLU*up, one per expert) */
    const uint8_t  *const *ws;
    int n_in, n_out, k;
    size_t row_bytes;
} MatmulQ5_1BatchArgs;

static void matmul_q5_1_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ5_1BatchArgs *a = (MatmulQ5_1BatchArgs *)arg;
    int    n_out     = a->n_out;
    size_t row_bytes = a->row_bytes;

    /* Software prefetch to hide DRAM latency (row_bytes = 1056 bytes = 17 cache lines). */
#define Q5_1_PF_DIST 4
    for (int r = start; r < end; r++) {
        if (r + Q5_1_PF_DIST < end) {
            int pr = r + Q5_1_PF_DIST;
            int ei_p = pr / n_out;
            int ri_p = pr % n_out;
            const char *pfx = (const char *)(a->ws[ei_p] + (size_t)ri_p * row_bytes);
            for (int p = 0; p < (int)row_bytes; p += 64)
                TN_PREFETCH_T1(pfx + p);
        }
        int ei = r / n_out;
        int ri = r % n_out;
        a->outs[ei][ri] = dot_q5_1_row(a->ws[ei] + (size_t)ri * row_bytes,
                                        a->xs[ei], a->n_in);
    }
#undef Q5_1_PF_DIST
}

void parallel_matmul_q5_1_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q5_1_BLOCK) * Q5_1_BYTES;
    MatmulQ5_1BatchArgs args = {
        .outs      = outs,
        .xs        = xs,
        .ws        = ws,
        .n_in      = n,
        .n_out     = d,
        .k         = k,
        .row_bytes = row_bytes,
    };
    if (!tp) {
        matmul_q5_1_batch_task(&args, 0, 0, k * d);
        return;
    }
    threadpool_dispatch(tp, matmul_q5_1_batch_task, &args, k * d);
}
