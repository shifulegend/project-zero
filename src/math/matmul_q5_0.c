/*
 * parallel_matmul_q5_0 — fused Q5_0 × F32 activation matmul (gemv).
 *
 * Q5_0 block (22 bytes / 32 elements):
 *   [d: fp16 2B] [qh: u32 4B] [qs: u8×16 16B]
 *
 * Decode: val_i = signed_i * d
 *   where signed_i = (nibble_i | (qh_bit_i << 4)) - 16  ∈ [-16, 15]
 *
 * SIMD decode (AVX2): uses llama.cpp-style 256-bit bytes_from_bits_32 trick.
 *   Step 1 — bytes_from_nibbles_32: unpack qs[16] → 32 nibbles in __m256i.
 *             Lower 128 bits = low nibbles (elements 0..15),
 *             upper 128 bits = high nibbles (elements 16..31).
 *   Step 2 — bytes_from_bits_32: spread qh bits to 32 bytes (0xFF or 0x00).
 *             shuffle_epi8 broadcasts each qh byte to 8 positions, then
 *             OR+cmpeq pattern converts: bit=1 → 0xFF, bit=0 → 0x00.
 *   Step 3 — andnot with 0xF0: converts 0xFF→0x00, 0x00→0xF0.
 *             OR with nibbles: q5=0 → nibble|0xF0 = -16..-1 (signed int8);
 *                              q5=1 → nibble|0x00 = 0..15 (signed int8).
 *   Step 4 — cvtepi8_epi32 + cvtepi32_ps: 4 groups of 8 int8 → float.
 *   Step 5 — FMA: dot_acc += d * float_vals * x_vals (single acc, one hsum).
 */

#include "math/matmul_q5_0.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX512 || TN_HAS_AVX2
#  include <immintrin.h>
#endif

/* ── Q5_0 constants ──────────────────────────────────────────────────────── */
#define Q5_0_BLOCK   32
#define Q5_0_BYTES   22   /* bytes per 32-element block */

static inline float q5_0_f16_to_f32(uint16_t h) {
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
 * dot_q5_0_row: fused dot product of one Q5_0-encoded row with float32 x.
 */
static float dot_q5_0_row(const uint8_t *row_q5_0, const float *x, int n) {
    int n_blocks = n / Q5_0_BLOCK;

#if TN_HAS_AVX2
    /*
     * SIMD decode constants (outside the block loop):
     *
     * shuf256: shuffle mask to spread qh bytes to 32 byte positions.
     *   _mm256_shuffle_epi8 operates per 128-bit lane; lower lane gets bytes
     *   0 and 1 of qh (indices 0,1), upper lane gets bytes 2 and 3 (indices 2,3).
     *   Since the source is _mm256_set1_epi32(qh), both lanes have all 4 bytes.
     *
     * bit_mask256: 0x7fbfdfeff7fbfdfe in each 64-bit word.
     *   Each of the 8 bytes within a 64-bit chunk has one bit clear (bit i).
     *   After OR with qh_byte, a byte becomes 0xFF iff that qh bit was set.
     *
     * hi_flag256: 0xF0 in every byte (bit 4 of Q5 high contribution).
     *   andnot(cmpeq_result, 0xF0) → 0x00 where qh_bit=1, 0xF0 where qh_bit=0.
     *   ORing with 4-bit nibble gives signed int8: nibble|0xF0 ∈ [-16,-1],
     *   nibble|0x00 ∈ [0,15].
     */
    static const int8_t shuf256_data[32] = {
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    };
    const __m256i shuf256  = _mm256_loadu_si256((const __m256i *)shuf256_data);
    const __m256i bit_mask = _mm256_set1_epi64x((int64_t)0x7fbfdfeff7fbfdfeULL);
    const __m256i hi_flag  = _mm256_set1_epi8((char)0xF0);
    const __m256i nib_mask = _mm256_set1_epi8(0x0F);
    const __m256i all_ones = _mm256_set1_epi8((char)0xFF);

#if TN_HAS_AVX512
    /* AVX-512 path: 16-wide FMA, one reduce at end */
    __m512 dot_acc = _mm512_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q5_0 + (size_t)b * Q5_0_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float    d_val   = q5_0_f16_to_f32(d_bits);

        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t *qs = blk + 6;
        const float   *xb = x + (size_t)b * Q5_0_BLOCK;

        /* bytes_from_nibbles_32: lower = low nibbles, upper = high nibbles */
        const __m128i tmp128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i nibbles = _mm256_and_si256(
            nib_mask,
            _mm256_set_m128i(_mm_srli_epi16(tmp128, 4), tmp128));

        /* bytes_from_bits_32: spread qh bits → 0xFF or 0x00 per byte */
        __m256i qh_spread = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf256);
        __m256i bxhi      = _mm256_cmpeq_epi8(_mm256_or_si256(qh_spread, bit_mask), all_ones);

        /* andnot+OR → signed int8: -16..15 */
        __m256i q5_s8 = _mm256_or_si256(nibbles,
                            _mm256_andnot_si256(bxhi, hi_flag));

        /* Convert signed int8 → float (two 16-wide groups via sign-extend) */
        __m128i lo128 = _mm256_castsi256_si128(q5_s8);
        __m128i hi128 = _mm256_extracti128_si256(q5_s8, 1);
        __m512 fv0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo128));
        __m512 fv1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi128));

        __m512 dv = _mm512_set1_ps(d_val);
        dot_acc = _mm512_fmadd_ps(_mm512_mul_ps(dv, fv0), _mm512_loadu_ps(xb),      dot_acc);
        dot_acc = _mm512_fmadd_ps(_mm512_mul_ps(dv, fv1), _mm512_loadu_ps(xb + 16), dot_acc);
    }
    return _mm512_reduce_add_ps(dot_acc);

#else /* AVX2 only */
    /* AVX2 path: 8-wide FMA, four groups, one hsum at end */
    __m256 dot_acc = _mm256_setzero_ps();
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q5_0 + (size_t)b * Q5_0_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        __m256 dv = _mm256_set1_ps(q5_0_f16_to_f32(d_bits));

        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t *qs = blk + 6;
        const float   *xb = x + (size_t)b * Q5_0_BLOCK;

        const __m128i tmp128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i nibbles = _mm256_and_si256(
            nib_mask,
            _mm256_set_m128i(_mm_srli_epi16(tmp128, 4), tmp128));

        __m256i qh_spread = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf256);
        __m256i bxhi      = _mm256_cmpeq_epi8(_mm256_or_si256(qh_spread, bit_mask), all_ones);
        __m256i q5_s8     = _mm256_or_si256(nibbles, _mm256_andnot_si256(bxhi, hi_flag));

        /* Sign-extend 8 int8 → 8 int32 → float, four groups */
        __m128i lo128 = _mm256_castsi256_si128(q5_s8);
        __m128i hi128 = _mm256_extracti128_si256(q5_s8, 1);
        __m256 fv0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128));
        __m256 fv1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo128, 8)));
        __m256 fv2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128));
        __m256 fv3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi128, 8)));

        dot_acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, fv0), _mm256_loadu_ps(xb),      dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, fv1), _mm256_loadu_ps(xb +  8), dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, fv2), _mm256_loadu_ps(xb + 16), dot_acc);
        dot_acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, fv3), _mm256_loadu_ps(xb + 24), dot_acc);
    }
    __m128 lo = _mm256_castps256_ps128(dot_acc);
    __m128 hi = _mm256_extractf128_ps(dot_acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
#endif /* TN_HAS_AVX512 */

#else /* no AVX2 */
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q5_0 + (size_t)b * Q5_0_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = q5_0_f16_to_f32(d_bits);
        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t *qs = blk + 6;
        const float   *xb = x + (size_t)b * Q5_0_BLOCK;
        for (int i = 0; i < 16; i++) {
            float q0 = (float)((qs[i] & 0xF) | (((qh >>  i)       & 1) << 4)) - 16.0f;
            float q1 = (float)((qs[i] >>   4) | (((qh >> (i + 16)) & 1) << 4)) - 16.0f;
            dot += d * (q0 * xb[i] + q1 * xb[i + 16]);
        }
    }
    return dot;
#endif /* TN_HAS_AVX2 */
}

/* ── Single-matrix thread pool dispatch ─────────────────────────────────── */

typedef struct {
    float         *out;
    const float   *x;
    const uint8_t *w;
    int n, d;
    size_t row_bytes;
} MatmulQ5_0Args;

static void matmul_q5_0_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ5_0Args *a = (MatmulQ5_0Args *)arg;
    for (int i = start; i < end; i++)
        a->out[i] = dot_q5_0_row(a->w + (size_t)i * a->row_bytes, a->x, a->n);
}

void parallel_matmul_q5_0(float *out, const float *x, const uint8_t *w_q5_0,
                           int n, int d, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q5_0_BLOCK) * Q5_0_BYTES;
    MatmulQ5_0Args args = { .out = out, .x = x, .w = w_q5_0,
                             .n = n, .d = d, .row_bytes = row_bytes };
    if (!tp) { matmul_q5_0_task(&args, 0, 0, d); return; }
    threadpool_dispatch(tp, matmul_q5_0_task, &args, d);
}

/* ── Batched: k weight matrices, per-expert inputs ───────────────────────── */

typedef struct {
    float        **outs;
    float        **xs;
    const uint8_t *const *ws;
    int n_in, n_out, k;
    size_t row_bytes;
} MatmulQ5_0BatchArgs;

static void matmul_q5_0_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ5_0BatchArgs *a = (MatmulQ5_0BatchArgs *)arg;
    int n_out = a->n_out;
    for (int r = start; r < end; r++) {
        int ei = r / n_out;
        int ri = r % n_out;
        a->outs[ei][ri] = dot_q5_0_row(a->ws[ei] + (size_t)ri * a->row_bytes,
                                        a->xs[ei], a->n_in);
    }
}

void parallel_matmul_q5_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp) {
    size_t row_bytes = (size_t)(n / Q5_0_BLOCK) * Q5_0_BYTES;
    MatmulQ5_0BatchArgs args = {
        .outs = outs, .xs = xs, .ws = ws,
        .n_in = n, .n_out = d, .k = k,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q5_0_batch_task(&args, 0, 0, k * d); return; }
    threadpool_dispatch(tp, matmul_q5_0_batch_task, &args, k * d);
}
