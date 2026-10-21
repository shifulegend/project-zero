#include "math/matmul_q6k.h"
#include "core/platform.h"
#include <stdlib.h>
#include <string.h>
#if TN_HAS_AVX2
#include <immintrin.h>
#endif

/* Block layout (ggml block_q6_K): ql[128] low nibbles, qh[64] high 2-bit
 * pairs, scales[16] int8, d fp16. Element e of half h (0..127):
 *   l = e % 32, part = e / 32 (0..3)
 *   low  = part&1 ? ql[64h + l + 32] : ql[64h + l],  shifted >> 4 when part >= 2
 *   high = (qh[32h + l] >> (2*part)) & 3
 *   q = (low | high << 4) - 32,  scale = scales[8h + 2*part + l/16]. */

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

float tn_dot_q6k_row_q8k_scalar(const uint8_t *row, const TnQ8KActBlock *acts, int n_blocks) {
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + (size_t)b * TN_Q6K_BYTES;
        const uint8_t *ql0 = blk, *qh0 = blk + 128;
        const int8_t *sc0 = (const int8_t *)(blk + 192);
        uint16_t dh; memcpy(&dh, blk + 208, 2);
        const int8_t *a = acts[b].qs;
        int32_t isum = 0;
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = ql0 + 64 * h, *qh = qh0 + 32 * h;
            const int8_t *sc = sc0 + 8 * h;
            const int8_t *ah = a + 128 * h;
            for (int g = 0; g < 8; g++) {          /* 16-element sub-blocks */
                int part = g / 2, base = (g % 2) * 16;
                int32_t s = 0;
                for (int k = 0; k < 16; k++) {
                    int l = base + k;
                    int low = (part & 1) ? ql[l + 32] : ql[l];
                    low = (part >= 2) ? (low >> 4) : (low & 0xF);
                    int hi = (qh[l] >> (2 * part)) & 3;
                    int q = (low | (hi << 4)) - 32;
                    s += q * ah[32 * part + l];
                }
                isum += s * sc[2 * part + (g % 2)];
            }
        }
        dot += fp16_to_f32(dh) * acts[b].d * (float)isum;
    }
    return dot;
}

#if TN_HAS_AVX2
static inline int32_t hsum_i32_8(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* 2026-10-21: per-block scale vectors come from one byte shuffle of the
 * loaded scales (was 8 x _mm256_setr_epi16 of 16 scalars each — the bulk of
 * the kernel's instructions), the -32 bias is one vpmaddwd of scales x
 * bsums (was a 16-iteration scalar loop), and the fp16 scale uses F16C.
 * The integer block sum is unchanged, so the result stays bit-identical to
 * tn_dot_q6k_row_q8k_scalar() (tests/test_q6k_matmul.c). The LM head of
 * DeepSeek-V2-Lite (Q6_K, 172 MB) ran at ~15 GB/s on 4 threads. */
static inline float q6k_fp16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_f32(h);
#endif
}

float tn_dot_q6k_row_q8k(const uint8_t *row, const TnQ8KActBlock *acts, int n_blocks) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(0x03);
    /* low 128-bit lane takes int16 scale 2p, high lane scale 2p+1 */
    static const int8_t k_scale_shuf[4][32] = {
        { 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,  2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3 },
        { 4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,  6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7 },
        { 8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,  10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11 },
        { 12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,  14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15 },
    };
    float dot = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + (size_t)b * TN_Q6K_BYTES;
        const int8_t *sc = (const int8_t *)(blk + 192);
        uint16_t dh; memcpy(&dh, blk + 208, 2);
        const TnQ8KActBlock *act = &acts[b];
        const __m128i sc8 = _mm_loadu_si128((const __m128i *)sc);
        /* bias term: sum_j sc[j] * bsums[j] (scale index j covers bsums[j]) */
        const __m256i sc16_all = _mm256_cvtepi8_epi16(sc8);
        __m256i acc = _mm256_slli_epi32(
            _mm256_madd_epi16(sc16_all, _mm256_loadu_si256((const __m256i *)act->bsums)), 5);
        acc = _mm256_sub_epi32(_mm256_setzero_si256(), acc);
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
            const int8_t *a = act->qs + 128 * h;
            const __m128i sh8 = _mm_cvtepi8_epi16(_mm_srli_si128(sc8, 8 * h));
            const __m256i sh  = _mm256_broadcastsi128_si256(sh8);
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
            __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hh = _mm256_loadu_si256((const __m256i *)qh);
            /* unsigned 6-bit codes 0..63 for the four 32-element parts */
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(hh, m3), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 2), m3), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 4), m3), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 6), m3), 4));
            for (int p = 0; p < 4; p++) {
                __m256i av  = _mm256_loadu_si256((const __m256i *)(a + 32 * p));
                __m256i p16 = _mm256_maddubs_epi16(q[p], av);   /* <= 2*63*127 */
                __m256i scv = _mm256_shuffle_epi8(sh, _mm256_loadu_si256((const __m256i *)k_scale_shuf[p]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, scv));
            }
        }
        int32_t isum = hsum_i32_8(acc);
        dot += q6k_fp16(dh) * act->d * (float)isum;
    }
    return dot;
}
#else
float tn_dot_q6k_row_q8k(const uint8_t *row, const TnQ8KActBlock *acts, int n_blocks) {
    return tn_dot_q6k_row_q8k_scalar(row, acts, n_blocks);
}
#endif

typedef struct {
    float *out;
    const uint8_t *w;
    const TnQ8KActBlock *acts;   /* [n_tokens][n_blocks] */
    int n_blocks, d, n_tokens;
} MatmulQ6KArgs;

static void matmul_q6k_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MatmulQ6KArgs *a = (const MatmulQ6KArgs *)arg;
    size_t row_bytes = (size_t)a->n_blocks * TN_Q6K_BYTES;
    for (int i = start; i < end; i++) {
        const uint8_t *row = a->w + (size_t)i * row_bytes;
        for (int t = 0; t < a->n_tokens; t++)
            a->out[(size_t)t * a->d + i] =
                tn_dot_q6k_row_q8k(row, a->acts + (size_t)t * a->n_blocks, a->n_blocks);
    }
}

void parallel_matmul_q6k_tokens(float *out, const float *x, const uint8_t *w_q6k,
                                int n, int d, int n_tokens, ThreadPool *tp) {
    int n_blocks = n / 256;
    if (n_blocks <= 0 || n_tokens <= 0) return;
    TnQ8KActBlock stack_acts[16];
    size_t need = (size_t)n_tokens * n_blocks;
    TnQ8KActBlock *acts = need <= 16 ? stack_acts
                        : (TnQ8KActBlock *)malloc(need * sizeof(TnQ8KActBlock));
    if (!acts) return;   /* OOM: leave out untouched (caller-visible zeros/garbage) */
    for (int t = 0; t < n_tokens; t++)
        tn_quantize_q8k(acts + (size_t)t * n_blocks, x + (size_t)t * n, n_blocks);
    MatmulQ6KArgs args = { out, w_q6k, acts, n_blocks, d, n_tokens };
    if (!tp) matmul_q6k_task(&args, 0, 0, d);
    else threadpool_dispatch(tp, matmul_q6k_task, &args, d);
    if (acts != stack_acts) free(acts);
}

void parallel_matmul_q6k(float *out, const float *x, const uint8_t *w_q6k,
                         int n, int d, ThreadPool *tp) {
    parallel_matmul_q6k_tokens(out, x, w_q6k, n, d, 1, tp);
}
