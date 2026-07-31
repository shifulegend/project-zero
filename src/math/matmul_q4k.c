/*
 * parallel_matmul_q4k — fused Q4_K × Q8K-activation matmul (gemv).
 *
 * Activations are pre-quantized to Q8K (int8 + bsums) once per forward call.
 * The inner kernel uses _mm256_maddubs_epi16 (AVX2) or _mm256_dpbusds_epi32
 * (AVX-512 VNNI) for integer dot products, 3× lower bandwidth than F32 path.
 *
 * Q4_K super-block layout (144 bytes, 256 elements):
 *   [d:     fp16 (2 bytes)] — super-block scale
 *   [dmin:  fp16 (2 bytes)] — super-block min
 *   [sc12: 12 bytes]        — 8 sub-blocks × packed (6-bit scale + 6-bit min)
 *   [qs:  128 bytes]        — 256 × 4-bit values, packed 2/byte
 *
 * Group layout (g = 0..3, each covers 64 elements):
 *   lo 32: low  nibbles of qs[g*32 .. g*32+31], scale=sc[2g],   min=mn[2g]
 *   hi 32: high nibbles of qs[g*32 .. g*32+31], scale=sc[2g+1], min=mn[2g+1]
 *
 * Q8K activation block (292 bytes, 256 elements):
 *   [d:      f32 (4 bytes)]  — activation scale = abs_max / 127
 *   [qs:     int8[256]]      — quantized activations
 *   [bsums:  int16[16]]      — bsums[j] = sum(qs[j*16 .. j*16+15])
 *
 * Dot product formula (per superblock b):
 *   dot_b  = d_a * (d_w * int_dot - dmin_w * int_min)
 *   int_dot = sum_g[ sc[2g] * sum(lo_nibbles[g] * qs_lo[g])
 *                  + sc[2g+1] * sum(hi_nibbles[g] * qs_hi[g]) ]
 *   int_min = sum_j(mn[j] * (bsums[2j] + bsums[2j+1]))  for j=0..7
 */

#include "math/matmul_q4k.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if TN_HAS_AVX2 || TN_HAS_AVX512
#  include <immintrin.h>
#endif
/* TN_PREFETCH_T1 is defined in core/platform.h (included above via matmul_q4k.h).
 * Uses __builtin_prefetch for portability across x86 and ARM (macOS M-series). */

/* ── Q4_K constants ────────────────────────────────────────────────────────── */
#define Q4K_SUPER   256
#define Q4K_NSUB    8
#define Q4K_BYTES   144

/* Q8K_BLOCK mirrors TN_Q8K_BLOCK from the header — kept local for clarity */
#define Q8K_BLOCK   256

/* ── Module-level Q8K activation buffer (reallocated as needed) ─────────────── */
static TnQ8KActBlock *s_q8k_buf       = NULL;
static int            s_q8k_buf_blocks = 0;

static TnQ8KActBlock *q8k_buf_ensure(int n_blocks) {
    if (s_q8k_buf_blocks < n_blocks) {
        free(s_q8k_buf);
        s_q8k_buf = (TnQ8KActBlock *)malloc((size_t)n_blocks * sizeof(TnQ8KActBlock));
        s_q8k_buf_blocks = s_q8k_buf ? n_blocks : 0;
    }
    return s_q8k_buf;
}

/* ── Scalar F16→F32 helper ────────────────────────────────────────────────── */
static inline float q4k_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t bits;
    if (exp == 0) {
        uint32_t m = (uint32_t)(h & 0x3ff);
        if (m == 0) return 0.0f;
        int e_adj = 0;
        while ((m & 0x400u) == 0) { m <<= 1; e_adj++; }
        bits = sign | ((uint32_t)(113 - e_adj) << 23) | ((m & 0x3ffu) << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | ((uint32_t)(h & 0x3ff) << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | ((uint32_t)(h & 0x3ff) << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

/* ── Decode 8 raw integer (scale, min) pairs from sc12[12] ──────────────────── *
 * Only used by dot_q4k_row_q8k's scalar (no-SIMD) fallback below — the AVX2
 * path decodes scales inline via bit ops instead (see that function's own
 * comment), so this is compiled out on AVX2 builds to avoid an unused-
 * function warning there. */
#if !TN_HAS_AVX2
static inline void q4k_decode_scales_raw(const uint8_t *sc12,
                                          uint8_t *raw_sc, uint8_t *raw_mn) {
    for (int j = 0; j < Q4K_NSUB; j++) {
        uint8_t sc, m;
        if (j < 4) {
            sc = sc12[j]     & 63;
            m  = sc12[j + 4] & 63;
        } else {
            int k = j - 4;
            sc = (sc12[k + 8] & 0x0F) | ((sc12[k]     >> 6) << 4);
            m  = (sc12[k + 8] >>    4) | ((sc12[k + 4] >> 6) << 4);
        }
        raw_sc[j] = sc;
        raw_mn[j] = m;
    }
}
#endif /* !TN_HAS_AVX2 */

/* ── Quantize float activation → Q8K blocks (one per 256 elements) ──────────── */
static void quantize_to_q8k(TnQ8KActBlock *out, const float *x, int n_blocks) {
#if TN_HAS_AVX2
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * Q8K_BLOCK;
        TnQ8KActBlock *ob = out + b;

        /* Pass 1: find abs max over 256 floats using AVX2 */
        __m256 vmax = _mm256_setzero_ps();
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        for (int i = 0; i < 256; i += 8) {
            __m256 v = _mm256_loadu_ps(xb + i);
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
        }
        /* Horizontal max of 8 lanes */
        __m128 hi4  = _mm256_extractf128_ps(vmax, 1);
        __m128 lo4  = _mm256_castps256_ps128(vmax);
        __m128 m4   = _mm_max_ps(lo4, hi4);
        __m128 m2   = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        __m128 m1   = _mm_max_ss(m2, _mm_movehdup_ps(m2));
        float abs_max = _mm_cvtss_f32(m1);

        if (abs_max == 0.0f) {
            ob->d = 0.0f;
            memset(ob->qs, 0, 256);
            memset(ob->bsums, 0, 32);
            continue;
        }

        ob->d = abs_max / 127.0f;
        float inv = 127.0f / abs_max;

        /* Pass 2: quantize + compute bsums[16] (each bsum covers 16 elements) */
        __m256 vscale   = _mm256_set1_ps(inv);
        __m256 vpos127  = _mm256_set1_ps( 127.0f);
        __m256 vneg127  = _mm256_set1_ps(-127.0f);

        for (int j = 0; j < 16; j++) {
            /* Quantize 16 elements */
            const float *xj = xb + j * 16;
            int8_t *qj = ob->qs + j * 16;

            __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(xj),     vscale);
            __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(xj + 8), vscale);
            v0 = _mm256_min_ps(_mm256_max_ps(v0, vneg127), vpos127);
            v1 = _mm256_min_ps(_mm256_max_ps(v1, vneg127), vpos127);

            /* float → int32 (round-to-nearest, matches llama.cpp) → pack to int8 */
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
            __m256i i32_0 = _mm256_cvtps_epi32(v0);
            __m256i i32_1 = _mm256_cvtps_epi32(v1);
            __m128i lo0 = _mm256_castsi256_si128(i32_0);
            __m128i hi0 = _mm256_extracti128_si256(i32_0, 1);
            __m128i lo1 = _mm256_castsi256_si128(i32_1);
            __m128i hi1 = _mm256_extracti128_si256(i32_1, 1);
            __m128i i16_0 = _mm_packs_epi32(lo0, hi0);  /* 8 int16 */
            __m128i i16_1 = _mm_packs_epi32(lo1, hi1);  /* 8 int16 */
            __m128i i8_16 = _mm_packs_epi16(i16_0, i16_1); /* 16 int8 */
            _mm_storeu_si128((__m128i *)qj, i8_16);

            /* Compute bsum[j] = sum of 16 int8 values */
            __m128i sum16 = _mm_add_epi16(i16_0, i16_1); /* 8 int16 sums */
            /* Hadd twice: 8→4→2 */
            sum16 = _mm_hadd_epi16(sum16, _mm_setzero_si128());
            sum16 = _mm_hadd_epi16(sum16, _mm_setzero_si128());
            sum16 = _mm_hadd_epi16(sum16, _mm_setzero_si128());
            ob->bsums[j] = (int16_t)_mm_extract_epi16(sum16, 0);
        }
    }
#else
    /* Scalar fallback */
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * Q8K_BLOCK;
        TnQ8KActBlock *ob = out + b;

        float abs_max = 0.0f;
        for (int i = 0; i < 256; i++) {
            float a = xb[i] < 0 ? -xb[i] : xb[i];
            if (a > abs_max) abs_max = a;
        }
        if (abs_max == 0.0f) {
            ob->d = 0.0f;
            memset(ob->qs, 0, 256);
            memset(ob->bsums, 0, 32);
            continue;
        }
        ob->d = abs_max / 127.0f;
        float inv = 127.0f / abs_max;

        for (int j = 0; j < 16; j++) {
            int16_t bsum = 0;
            for (int k = 0; k < 16; k++) {
                float v = xb[j * 16 + k] * inv;
                if (v >  127.0f) v =  127.0f;
                if (v < -127.0f) v = -127.0f;
                int8_t q = (int8_t)(int)nearbyintf(v); /* round-to-nearest, matches llama.cpp */
                ob->qs[j * 16 + k] = q;
                bsum += q;
            }
            ob->bsums[j] = bsum;
        }
    }
#endif
}

/* ── llama.cpp-style scale shuffle table for Q4K ────────────────────────────── */
/*
 * get_scale_shuffle_k4(i): returns a 256-bit shuffle mask that broadcasts
 * scales[i] (int16) to all 16 positions of a __m256i.
 * Each entry covers 32 bytes. For i=0: {0,1,0,1,...,0,1} → scales[0] broadcast.
 * Used instead of 8 × _mm256_set1_epi16 per superblock to avoid register pressure
 * and scalar-to-SIMD bypass penalties. Identical to llama.cpp's get_scale_shuffle_k4.
 */
#if TN_HAS_AVX2
static inline __m256i q4k_scale_shuffle(int i) {
    static const uint8_t k_shuf[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuf + i);
}

#endif

/*
 * dot_q4k_row_q8k: Q4K weight row × Q8K activation blocks.
 *
 * Optimized to match llama.cpp ggml_vec_dot_q4_K_q8_K (AVX2 path):
 *   - Scale decode: uint32 bit ops (no scalar loop, no q4k_decode_scales_raw)
 *   - Scale broadcast: q4k_scale_shuffle table + 1 shuffle_epi8 per sub-block
 *     (eliminates 8 × set1_epi16 + scalar→SIMD bypass penalty)
 *   - Min correction: accumulated into __m128 acc_m across ALL blocks
 *   - Main dot: accumulated into __m256 acc across ALL blocks → 1 hsum per row
 *     (vs old: 1 hsum per block = 8 hsums per row)
 *
 * Integer overflow budget (same as before):
 *   maddubs: max int16 = 2 × 15 × 127 = 3810 < 32767  ✓
 *   madd_epi16(scale, p16): max int32 = 2 × 63 × 3810 = 480060 < 2^31/4  ✓
 *   sumi after 4 groups: 4 × 2 × 480060 = 3.84M < 2^31  ✓
 */
static float dot_q4k_row_q8k(const uint8_t *row_q4k,
                               const TnQ8KActBlock *acts, int n_blocks) {
    float dot = 0.0f;
#if TN_HAS_AVX2
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const __m256i mask4 = _mm256_set1_epi8(0x0F);

    __m256 acc   = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q4k + (size_t)b * Q4K_BYTES;
        const TnQ8KActBlock *act = acts + b;
        float d_a = act->d;
        if (d_a == 0.0f) continue;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d_w    = q4k_f16_to_f32(d_bits);
        float dmin_w = q4k_f16_to_f32(dmin_bits);

        /* ── Scale/min decode: uint32 bit ops (llama.cpp style) ─────────────
         * 12 bytes → utmp[0..3] where:
         *   utmp[0] bytes: [sc[0], sc[1], sc[2], sc[3]] (6-bit scales)
         *   utmp[1] bytes: [sc[4], sc[5], sc[6], sc[7]]
         *   utmp[2] bytes: [mn[0], mn[1], mn[2], mn[3]] (6-bit mins)
         *   utmp[3] bytes: [mn[4], mn[5], mn[6], mn[7]]
         * All in SIMD domain — no scalar-to-SIMD bypass penalty.
         */
        uint32_t utmp[4];
        memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        /* Load all 8 scales (lo 128) + 8 mins (hi 128) as int16 in one register */
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        /* ── Min correction (once per block, into __m128 acc_m) ─────────────
         * acc_m += (-d_a * dmin_w) × sum_j(mins[j] * (bsums[2j] + bsums[2j+1]))
         */
        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)act->bsums);
        const __m128i q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod_min = _mm_madd_epi16(
            _mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(-d_a * dmin_w),
                              _mm_cvtepi32_ps(prod_min), acc_m);

        /* ── Main dot product: 4 groups of 64 elements each ─────────────────
         * Scales broadcast via shuffle table — stays in SIMD, no register spill.
         * Single __m256i sumi per block, accumulated into __m256 acc across blocks.
         */
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        const uint8_t *qs     = blk + 16;
        const int8_t  *act_qs = act->qs;

        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, q4k_scale_shuffle(2 * j));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, q4k_scale_shuffle(2 * j + 1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i *)(qs + j * 32));
            const __m256i q4l = _mm256_and_si256(q4bits, mask4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), mask4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i *)(act_qs + j * 64));
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i *)(act_qs + j * 64 + 32));
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d_a * d_w),
                               _mm256_cvtepi32_ps(sumi), acc);
    }

    /* Single hsum of 8 float lanes + min correction scalar */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    float main_dot = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    return main_dot + _mm_cvtss_f32(_mm_add_ss(acc_m, _mm_movehdup_ps(acc_m)));

#else
    /* Scalar fallback (no SIMD) */
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q4k + (size_t)b * Q4K_BYTES;
        const TnQ8KActBlock *act = acts + b;
        float d_a = act->d;
        if (d_a == 0.0f) continue;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);
        float d_w    = q4k_f16_to_f32(d_bits);
        float dmin_w = q4k_f16_to_f32(dmin_bits);

        uint8_t raw_sc[8], raw_mn[8];
        q4k_decode_scales_raw(blk + 4, raw_sc, raw_mn);

        const uint8_t *qs = blk + 16;
        int32_t int_dot = 0, int_min = 0;

        for (int g = 0; g < 4; g++) {
            const uint8_t *q = qs + g * 32;
            const int8_t *al = act->qs + g * 64;
            const int8_t *ah = act->qs + g * 64 + 32;
            int32_t sum_lo = 0, sum_hi = 0;
            for (int l = 0; l < 32; l++) {
                sum_lo += (int32_t)(q[l] & 0x0F) * (int32_t)al[l];
                sum_hi += (int32_t)(q[l] >>    4) * (int32_t)ah[l];
            }
            int_dot += (int32_t)raw_sc[2 * g]     * sum_lo;
            int_dot += (int32_t)raw_sc[2 * g + 1] * sum_hi;
        }
        for (int j = 0; j < 8; j++) {
            int32_t bsum_sb = (int32_t)act->bsums[2 * j] + (int32_t)act->bsums[2 * j + 1];
            int_min += (int32_t)raw_mn[j] * bsum_sb;
        }
        dot += d_a * (d_w * (float)int_dot - dmin_w * (float)int_min);
    }
#endif /* TN_HAS_AVX2 */
    return dot;
}

/* ── Thread pool dispatch ─────────────────────────────────────────────────── */

typedef struct {
    float              *out;
    const uint8_t      *w;
    const TnQ8KActBlock *acts;
    int n_blocks, d;
    size_t row_bytes;
} MatmulQ4KArgs;

static void matmul_q4k_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MatmulQ4KArgs *a = (const MatmulQ4KArgs *)arg;
    for (int i = start; i < end; i++)
        a->out[i] = dot_q4k_row_q8k(a->w + (size_t)i * a->row_bytes,
                                      a->acts, a->n_blocks);
}

void parallel_matmul_q4k(float *out, const float *x, const uint8_t *w_q4k,
                          int n, int d, ThreadPool *tp) {
    int n_blocks   = n / Q4K_SUPER;
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;

    TnQ8KActBlock *acts = q8k_buf_ensure(n_blocks);
    if (!acts) return;  /* OOM — caller will get zero output silently */
    quantize_to_q8k(acts, x, n_blocks);

    MatmulQ4KArgs args = {
        .out       = out,
        .w         = w_q4k,
        .acts      = acts,
        .n_blocks  = n_blocks,
        .d         = d,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q4k_task(&args, 0, 0, d); return; }
    threadpool_dispatch(tp, matmul_q4k_task, &args, d);
}

/* ── Batched Q4K: k weight matrices, shared input x ─────────────────────── */

typedef struct {
    float * const      *outs;
    const uint8_t * const *ws;
    const TnQ8KActBlock *acts;
    int n_blocks, n_out, k;
    size_t row_bytes;
} MatmulQ4KBatchArgs;

static void matmul_q4k_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MatmulQ4KBatchArgs *a = (const MatmulQ4KBatchArgs *)arg;
    int    n_out     = a->n_out;
    size_t row_bytes = a->row_bytes;

    /* Software prefetch: issue prefetch for rows PF_DIST ahead to hide DRAM latency.
     * Each row = row_bytes (1152 bytes for n=2048) = 18 cache lines.
     * PF_DIST=4 gives 4×compute_time lead — enough to cover DRAM latency (~100ns). */
#define Q4K_PF_DIST 4
    for (int r = start; r < end; r++) {
        /* Prefetch the row PF_DIST ahead */
        if (r + Q4K_PF_DIST < end) {
            int pr = r + Q4K_PF_DIST;
            int ei_p = pr / n_out;
            int ri_p = pr % n_out;
            const char *pfx = (const char *)(a->ws[ei_p] + (size_t)ri_p * row_bytes);
            for (int p = 0; p < (int)row_bytes; p += 64)
                TN_PREFETCH_T1(pfx + p);
        }
        int ei = r / n_out;
        int ri = r % n_out;
        a->outs[ei][ri] = dot_q4k_row_q8k(
                a->ws[ei] + (size_t)ri * row_bytes,
                a->acts, a->n_blocks);
    }
#undef Q4K_PF_DIST
}

void parallel_matmul_q4k_batch(float * const *outs, const float *x,
                                const uint8_t * const *ws,
                                int n, int d, int k, ThreadPool *tp) {
    int n_blocks   = n / Q4K_SUPER;
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;

    /* Quantize x to Q8K once — all k×d rows reuse the same activation blocks */
    TnQ8KActBlock *acts = q8k_buf_ensure(n_blocks);
    if (!acts) return;
    quantize_to_q8k(acts, x, n_blocks);

    MatmulQ4KBatchArgs args = {
        .outs      = outs,
        .ws        = ws,
        .acts      = acts,
        .n_blocks  = n_blocks,
        .n_out     = d,
        .k         = k,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q4k_batch_task(&args, 0, 0, k * d); return; }
    threadpool_dispatch(tp, matmul_q4k_batch_task, &args, k * d);
}

/* ── Public quantization entry point (declared in header) ───────────────── */
void tn_quantize_q8k(TnQ8KActBlock *out, const float *x, int n_blocks) {
    quantize_to_q8k(out, x, n_blocks);
}

/* ── Pre-quantized single matmul (skips internal quantization) ───────────── */
void parallel_matmul_q4k_preq(float *out, const TnQ8KActBlock *acts,
                               const uint8_t *w_q4k,
                               int n, int d, ThreadPool *tp) {
    int    n_blocks  = n / Q4K_SUPER;
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;

    MatmulQ4KArgs args = {
        .out       = out,
        .w         = w_q4k,
        .acts      = acts,
        .n_blocks  = n_blocks,
        .d         = d,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q4k_task(&args, 0, 0, d); return; }
    threadpool_dispatch(tp, matmul_q4k_task, &args, d);
}

/* ── Pre-quantized batch matmul (skips internal quantization) ────────────── */
void parallel_matmul_q4k_batch_preq(float * const *outs,
                                     const TnQ8KActBlock *acts,
                                     const uint8_t * const *ws,
                                     int n, int d, int k, ThreadPool *tp) {
    int    n_blocks  = n / Q4K_SUPER;
    size_t row_bytes = (size_t)n_blocks * Q4K_BYTES;

    MatmulQ4KBatchArgs args = {
        .outs      = outs,
        .ws        = ws,
        .acts      = acts,
        .n_blocks  = n_blocks,
        .n_out     = d,
        .k         = k,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q4k_batch_task(&args, 0, 0, k * d); return; }
    threadpool_dispatch(tp, matmul_q4k_batch_task, &args, k * d);
}
