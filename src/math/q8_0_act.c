/*
 * q8_0_act.c — Q8_0 activation quantization + integer row dots for Q5_0,
 * Q5_1 and Q8_0 weights (2026-10-21). See include/math/q8_0_act.h for why.
 *
 * Block layouts (GGUF, little-endian):
 *   Q5_0 (22 B / 32): d fp16 | qh u32 | qs[16]   w = ((nib | qh_bit<<4) - 16) * d
 *   Q5_1 (24 B / 32): d fp16 | m fp16 | qh u32 | qs[16]   w = (nib | qh_bit<<4) * d + m
 *   Q8_0 (34 B / 32): d fp16 | qs[32] int8                 w = q * d
 * Element j < 16 uses the low nibble of qs[j], j >= 16 the high nibble of
 * qs[j-16]; qh bit j is element j's 5th bit.
 *
 * AVX2 path = llama.cpp's ggml_vec_dot_q5_0_q8_0 / q5_1_q8_1 / q8_0_q8_0
 * shape (maddubs + madd). VNNI path (256-bit EVEX vpdpbusd, no ZMM so no
 * AVX-512 frequency licence) replaces maddubs+madd with one instruction;
 * Q5_0 runs unsigned (q+16) and subtracts 16*sum(a) once per block via the
 * precomputed activation sum, avoiding the sign trick entirely.
 */

#include "math/q8_0_act.h"
#include "math/cpu_features.h"
#include "core/gguf_reader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if TN_HAS_AVX2
#  include <immintrin.h>
#endif

#define Q5_0_BYTES 22
#define Q5_1_BYTES 24
#define Q8_0_BYTES 34

/* VNNI 256-bit dpbusd needs AVX512VNNI + AVX512VL at compile time. */
#if TN_HAS_AVX2 && TN_HAS_AVX512VNNI && defined(__AVX512VL__)
#  define Q8A_HAS_VNNI 1
#else
#  define Q8A_HAS_VNNI 0
#endif

static inline float q8a_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {         /* subnormal: renormalise */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            bits = sign | ((uint32_t)(112 - e) << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
#endif
}

int tn_q8_0_act_vnni_ok(void) {
#if Q8A_HAS_VNNI
    /* PZ_Q8A_VNNI=0 forces the AVX2 path (A/B aid); main thread only */
    static int env = -1;
    if (env < 0) {
        const char *e = getenv("PZ_Q8A_VNNI");
        env = (e && e[0] == '0') ? 0 : 1;
    }
    return (env && tn_cpu_features_detect()->avx512vnni) ? 1 : 0;
#else
    return 0;
#endif
}

/* ── Activation quantization ─────────────────────────────────────────────── */

void tn_quantize_q8_0_act(TnQ8_0ActBlock *out, const float *x, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * TN_Q8_0_BLOCK;
        TnQ8_0ActBlock *o = out + b;
#if TN_HAS_AVX2
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 v0 = _mm256_loadu_ps(xb),      v1 = _mm256_loadu_ps(xb + 8);
        __m256 v2 = _mm256_loadu_ps(xb + 16), v3 = _mm256_loadu_ps(xb + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        /* integer sum for s (before packing reorders lanes) */
        __m256i isum = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(isum), _mm256_extracti128_si256(isum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int sum = _mm_cvtsi128_si32(s4);
        /* pack 32 x i32 -> 32 x i8 and undo the AVX2 in-lane interleave */
        __m256i p01 = _mm256_packs_epi32(i0, i1);
        __m256i p23 = _mm256_packs_epi32(i2, i3);
        __m256i p   = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)o->qs, p);
        o->d = d;
        o->s = d * (float)sum;
#else
        float amax = 0.0f;
        for (int i = 0; i < TN_Q8_0_BLOCK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0;
        for (int i = 0; i < TN_Q8_0_BLOCK; i++) {
            int q = (int)nearbyintf(xb[i] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            o->qs[i] = (int8_t)q;
            sum += q;
        }
        o->d = d;
        o->s = d * (float)sum;
#endif
    }
}

/* ── AVX2 helpers (llama.cpp ggml-cpu/arch/x86 shapes) ───────────────────── */
#if TN_HAS_AVX2
static inline __m256i q8a_nibbles32(const uint8_t *qs) {
    const __m128i t = _mm_loadu_si128((const __m128i *)qs);
    return _mm256_and_si256(_mm256_set1_epi8(0x0F), _mm256_set_m128i(_mm_srli_epi16(t, 4), t));
}
/* byte i = 0xFF iff bit i of the 32-bit qh is set */
static inline __m256i q8a_bits32(const uint8_t *qhp) {
    uint32_t qh; memcpy(&qh, qhp, 4);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL,
                                           0x0101010101010101LL, 0x0000000000000000LL);
    __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shuf);
    b = _mm256_or_si256(b, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    return _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1));
}
static inline float q8a_hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}
/* unsigned u8 x signed i8 -> 8 x i32 pair sums (AVX2) */
static inline __m256i q8a_dot_us(__m256i u, __m256i s) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
}
#endif

/* ── Q5_0 ─────────────────────────────────────────────────────────────────── */
float tn_dot_q5_0_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int nb, int vnni) {
#if TN_HAS_AVX2
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
#  if Q8A_HAS_VNNI
    if (vnni) {
        const __m256i hb = _mm256_set1_epi8(0x10);
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * Q5_0_BYTES;
            const float dw = q8a_f16(blk);
            const __m256i u = _mm256_or_si256(q8a_nibbles32(blk + 6),
                                              _mm256_and_si256(q8a_bits32(blk + 2), hb));
            const __m256i qa = _mm256_loadu_si256((const __m256i *)a[b].qs);
            const __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, qa);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * a[b].d), _mm256_cvtepi32_ps(p), acc);
            corr += dw * a[b].s;
        }
        return q8a_hsum8(acc) - 16.0f * corr;
    }
#  endif
    (void)vnni; (void)corr;
    const __m256i hf = _mm256_set1_epi8((char)0xF0);
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q5_0_BYTES;
        const float dw = q8a_f16(blk);
        __m256i qx = _mm256_or_si256(q8a_nibbles32(blk + 6),
                                     _mm256_andnot_si256(q8a_bits32(blk + 2), hf));
        const __m256i qa = _mm256_loadu_si256((const __m256i *)a[b].qs);
        const __m256i p = q8a_dot_us(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qa, qx));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * a[b].d), _mm256_cvtepi32_ps(p), acc);
    }
    return q8a_hsum8(acc);
#else
    (void)vnni;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q5_0_BYTES;
        const float dw = q8a_f16(blk);
        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t *qs = blk + 6;
        int32_t si = 0;
        for (int j = 0; j < 16; j++) {
            int q0 = (int)((qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
            int q1 = (int)((qs[j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            si += q0 * a[b].qs[j] + q1 * a[b].qs[j + 16];
        }
        sum += dw * a[b].d * (float)si;
    }
    return sum;
#endif
}

/* ── Q5_1 ─────────────────────────────────────────────────────────────────── */
float tn_dot_q5_1_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int nb, int vnni) {
    float mins = 0.0f;
#if TN_HAS_AVX2
    __m256 acc = _mm256_setzero_ps();
    const __m256i hb = _mm256_set1_epi8(0x10);
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q5_1_BYTES;
        const float dw = q8a_f16(blk), mw = q8a_f16(blk + 2);
        const __m256i u = _mm256_or_si256(q8a_nibbles32(blk + 8),
                                          _mm256_and_si256(q8a_bits32(blk + 4), hb));
        const __m256i qa = _mm256_loadu_si256((const __m256i *)a[b].qs);
        __m256i p;
#  if Q8A_HAS_VNNI
        if (vnni) p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, qa);
        else
#  endif
        p = q8a_dot_us(u, qa);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * a[b].d), _mm256_cvtepi32_ps(p), acc);
        mins += mw * a[b].s;
    }
    (void)vnni;
    return q8a_hsum8(acc) + mins;
#else
    (void)vnni;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q5_1_BYTES;
        const float dw = q8a_f16(blk), mw = q8a_f16(blk + 2);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t *qs = blk + 8;
        int32_t si = 0;
        for (int j = 0; j < 16; j++) {
            int q0 = (int)((qs[j] & 0x0F) | (((qh >> j) & 1u) << 4));
            int q1 = (int)((qs[j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
            si += q0 * a[b].qs[j] + q1 * a[b].qs[j + 16];
        }
        sum += dw * a[b].d * (float)si;
        mins += mw * a[b].s;
    }
    return sum + mins;
#endif
}

/* ── Q8_0 ─────────────────────────────────────────────────────────────────── */
float tn_dot_q8_0_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int nb, int vnni) {
#if TN_HAS_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q8_0_BYTES;
        const float dw = q8a_f16(blk);
        const __m256i qw = _mm256_loadu_si256((const __m256i *)(blk + 2));
        const __m256i qa = _mm256_loadu_si256((const __m256i *)a[b].qs);
        const __m256i ax = _mm256_sign_epi8(qw, qw);
        const __m256i sy = _mm256_sign_epi8(qa, qw);
        __m256i p;
#  if Q8A_HAS_VNNI
        if (vnni) p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
        else
#  endif
        p = q8a_dot_us(ax, sy);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * a[b].d), _mm256_cvtepi32_ps(p), acc);
    }
    (void)vnni;
    return q8a_hsum8(acc);
#else
    (void)vnni;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b * Q8_0_BYTES;
        const float dw = q8a_f16(blk);
        const int8_t *qw = (const int8_t *)(blk + 2);
        int32_t si = 0;
        for (int j = 0; j < TN_Q8_0_BLOCK; j++) si += qw[j] * a[b].qs[j];
        sum += dw * a[b].d * (float)si;
    }
    return sum;
#endif
}

/* ── Whole-matrix GEMV over the thread pool ──────────────────────────────── */

typedef struct {
    float *out;
    const uint8_t *w;
    const TnQ8_0ActBlock *acts;
    size_t row_bytes;
    int n_blocks, type, vnni;
} Q8AArgs;

static void q8a_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const Q8AArgs *a = (const Q8AArgs *)arg;
    for (int r = start; r < end; r++) {
        const uint8_t *row = a->w + (size_t)r * a->row_bytes;
        float v;
        switch (a->type) {
        case GGUF_TYPE_Q5_0: v = tn_dot_q5_0_q8_0(row, a->acts, a->n_blocks, a->vnni); break;
        case GGUF_TYPE_Q5_1: v = tn_dot_q5_1_q8_0(row, a->acts, a->n_blocks, a->vnni); break;
        default:             v = tn_dot_q8_0_q8_0(row, a->acts, a->n_blocks, a->vnni); break;
        }
        a->out[r] = v;
    }
}

static TnQ8_0ActBlock *s_q8a_buf = NULL;
static int             s_q8a_cap = 0;

int parallel_matmul_q8a(float *out, const float *x, const uint8_t *w,
                        int gguf_type, int n, int d, ThreadPool *tp) {
    size_t blk;
    switch (gguf_type) {
    case GGUF_TYPE_Q5_0: blk = Q5_0_BYTES; break;
    case GGUF_TYPE_Q5_1: blk = Q5_1_BYTES; break;
    case GGUF_TYPE_Q8_0: blk = Q8_0_BYTES; break;
    default: return -1;
    }
    if (n <= 0 || n % TN_Q8_0_BLOCK != 0 || d <= 0) return -1;
    const int nb = n / TN_Q8_0_BLOCK;
    if (s_q8a_cap < nb) {
        TnQ8_0ActBlock *nbuf = (TnQ8_0ActBlock *)malloc((size_t)nb * sizeof(TnQ8_0ActBlock));
        if (!nbuf) return -1;
        free(s_q8a_buf);
        s_q8a_buf = nbuf;
        s_q8a_cap = nb;
    }
    tn_quantize_q8_0_act(s_q8a_buf, x, nb);
    Q8AArgs a = { out, w, s_q8a_buf, (size_t)nb * blk, nb, gguf_type, tn_q8_0_act_vnni_ok() };
    if (tp) threadpool_dispatch(tp, q8a_task, &a, d);
    else    q8a_task(&a, 0, 0, d);
    return 0;
}
