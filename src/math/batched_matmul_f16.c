#include "math/batched_matmul.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if TN_HAS_AVX512 || TN_HAS_AVX2
#  include <immintrin.h>
#endif

/**
 * F16 dense batched matmul. See include/math/batched_matmul.h for the
 * layout convention and the bandwidth-reuse rationale.
 *
 * 2026-09-24 rewrite: the original v1 scalar-accumulator implementation
 * (`val += f16_to_f32_scalar(row[j]) * xk[j]`, re-decoding + re-accumulating
 * from scratch for every one of the n_tokens candidates) delivered the
 * documented RAM-bandwidth win -- each row's 2 bytes/element are read from
 * RAM once -- but did the actual FMA work at scalar speed, on hardware
 * (AVX-512) whose single-token sibling (matmul_f16.c) does the same work
 * ~16-64x wider per instruction. Measured on a real SmolLM2-360M verifier:
 * a 5-token batched call cost ~790ms vs. ~150ms for 5 sequential
 * SIMD-accelerated single-token calls covering the same work -- i.e. this
 * kernel made speculative decoding's "verify" step *slower* than plain
 * generation, the opposite of Phase 18's whole point (see docs/ai/mistakes.md).
 * Fixed by decoding each row's F16 values to F32 ONCE (SIMD-accelerated
 * where available), then reusing the project's existing SIMD-dispatched
 * tn_vec_dot() (math/simd_dispatch.h -- the same kernel the single-token
 * dense/GQA attention path's own dot products already go through) for each
 * of the n_tokens dot products against that decoded row. This pays the
 * F16->F32 decode cost once per row (not once per candidate token) on top
 * of the RAM-read savings, and gets full SIMD width on the O(n*n_tokens)
 * dot-product work that actually dominates.
 */

/* Mirrors matmul_f16.c's own static f16_to_f32_scalar() -- this codebase's
 * established convention is a small static copy per matmul kernel TU
 * (see matmul_q4k.c/matmul_q8_0.c/matmul_q5_1.c/etc.), not a shared header. */
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

/* Decode `count` contiguous F16 values to F32, SIMD-accelerated where
 * available (hardware cvtph_ps, matching matmul_f16.c's own conversion
 * instructions), scalar tail/fallback otherwise. */
static void f16_row_to_f32(float *out, const tn_u16 *row, int count) {
    int j = 0;
#if TN_HAS_AVX512
    for (; j + 15 < count; j += 16) {
        __m512 wv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)&row[j]));
        _mm512_storeu_ps(&out[j], wv);
    }
#elif TN_HAS_AVX2
    for (; j + 7 < count; j += 8) {
        __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)&row[j]));
        _mm256_storeu_ps(&out[j], wv);
    }
#endif
    for (; j < count; j++) out[j] = f16_to_f32_scalar(row[j]);
}

typedef struct {
    float        *out;
    const float  *x;
    const tn_u16 *w;
    int n, d, n_tokens;
} F16BatchArgs;

static void matmul_f16_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    F16BatchArgs *a = (F16BatchArgs *)arg;

    float *row_f32 = (float *)malloc((size_t)a->n * sizeof(float));
    if (!row_f32) {
        /* OOM fallback: original scalar-per-token path, still correct. */
        for (int i = start; i < end; i++) {
            const tn_u16 *row = a->w + (size_t)i * a->n;
            for (int k = 0; k < a->n_tokens; k++) {
                const float *xk = a->x + (size_t)k * a->n;
                float val = 0.0f;
                for (int j = 0; j < a->n; j++) {
                    val += f16_to_f32_scalar(row[j]) * xk[j];
                }
                a->out[(size_t)k * a->d + i] = val;
            }
        }
        return;
    }

    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        /* `row` (2 bytes/element) is read from RAM and decoded to F32 ONCE
         * here, then reused across the n_tokens loop below -- both the RAM
         * read and the decode cost are amortized, not paid n_tokens times. */
        f16_row_to_f32(row_f32, row, a->n);
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            a->out[(size_t)k * a->d + i] = tn_vec_dot(row_f32, xk, a->n);
        }
    }
    free(row_f32);
}

void tn_matmul_f16_batch(float *out, const float *x, const tn_u16 *w,
                          int n, int d, int n_tokens, ThreadPool *tp) {
    F16BatchArgs args = { .out = out, .x = x, .w = w, .n = n, .d = d, .n_tokens = n_tokens };
    if (!tp) {
        matmul_f16_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_f16_batch_task, &args, d);
}
