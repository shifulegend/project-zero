/*
 * parallel_matmul_q2_0 — fused Q2_0 × F32 activation matmul (gemv).
 *
 * Q2_0 block (18 bytes / 64 elements): [d: fp16 2B] [qs: u8×16 16B].
 * Decode is bit-exact with mainline ggml's dequantize_row_q2_0:
 *   code = (qs[j/4] >> ((j%4)*2)) & 0x3;  value = (code - 1) * d
 *
 * Each block is decoded into a small stack buffer (cheap: 64 shift/mask ops,
 * dwarfed by the 64-wide FMA reduction that follows), then accumulated with
 * SIMD FMA against the matching slice of x. This keeps the hot loop's
 * correctness easy to verify while still vectorizing the memory-bandwidth-
 * bound accumulation.
 *
 * This is now the PORTABLE FALLBACK path: on an AVX-512 VNNI host,
 * parallel_matmul_q2_0() dispatches to matmul_q2_0_vnni.c's int8 VNNI
 * kernel instead (same "w_enc bias trick" as ternary_matmul_packed_vnni.c —
 * this decode-then-FMA path is ~1% of this host's own measured bandwidth
 * ceiling, confirmed on the real Ternary-Bonsai-27B model; see
 * docs/ai/mistakes.md). This scalar/AVX2 path still runs for non-VNNI
 * hosts and any input shape the VNNI kernel declines (n not a multiple of
 * 128, or larger than its stack-buffer ceiling).
 */

#include "math/matmul_q2_0.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX2
#include <immintrin.h>
#endif

/* ── Q2_0 constants ──────────────────────────────────────────────────────────
 * NOTE: this is PrismML's "group-128" Q2_0 packing (34 bytes / 128 elements =
 * 2.125 bits/weight), which predates and differs from mainline ggml's own
 * canonical block_q2_0 (group-64, 18 bytes/64 elements = 2.25 bits/weight,
 * shipped as the separate "*_g64.gguf" release). Both files' tensors are
 * tagged GGUF type 42 — the group size is NOT recoverable from the type ID
 * alone; confirmed empirically from this project's actual downloaded
 * Ternary-Bonsai-27B-Q2_0.gguf by computing bytes-per-tensor from adjacent
 * tensor offsets (2.125 bits/elem exactly, matching group-128, not 2.25).
 * The per-value decode formula is unchanged (see gguf_quant.c's
 * gguf_dequant_q2_0): only the group size and per-block byte count differ. */
#define Q2_0_BLOCK 128
#define Q2_0_BYTES 34   /* bytes per 128-element block: 2 (fp16 scale) + 32 (2-bit codes) */

static inline float q2_0_f16_to_f32(uint16_t h) {
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
 * dot_q2_0_row: fused dot product of one Q2_0-encoded row with float32 x.
 */
static float dot_q2_0_row(const uint8_t *row_q2_0, const float *x, int n) {
    int n_blocks = n / Q2_0_BLOCK;
    float total = 0.0f;

#if TN_HAS_AVX2
    __m256 acc = _mm256_setzero_ps();
#endif

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q2_0 + (size_t)b * Q2_0_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = q2_0_f16_to_f32(d_bits);
        const uint8_t *qs = blk + 2;   /* 16 bytes, 64 × 2-bit codes */
        const float *xb = x + (size_t)b * Q2_0_BLOCK;

        float decoded[Q2_0_BLOCK];
        for (int j = 0; j < Q2_0_BLOCK; j++) {
            int byte_index = j >> 2;
            int bit_offset = (j & 3) * 2;
            int code = (qs[byte_index] >> bit_offset) & 0x3;
            decoded[j] = (float)(code - 1);
        }

#if TN_HAS_AVX2
        __m256 dv = _mm256_set1_ps(d);
        for (int k = 0; k < Q2_0_BLOCK; k += 8) {
            __m256 wv = _mm256_loadu_ps(&decoded[k]);
            __m256 xv = _mm256_loadu_ps(&xb[k]);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(dv, wv), xv, acc);
        }
#else
        for (int j = 0; j < Q2_0_BLOCK; j++) total += d * decoded[j] * xb[j];
#endif
    }

#if TN_HAS_AVX2
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    total = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
#endif
    return total;
}

/* ── Single-matrix thread pool dispatch ─────────────────────────────────── */

typedef struct {
    float         *out;
    const float   *x;
    const uint8_t *w;
    int n, d;
    size_t row_bytes;
} MatmulQ2_0Args;

static void matmul_q2_0_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ2_0Args *a = (MatmulQ2_0Args *)arg;
    for (int i = start; i < end; i++)
        a->out[i] = dot_q2_0_row(a->w + (size_t)i * a->row_bytes, a->x, a->n);
}

#if TN_HAS_AVX512VNNI
/* matmul_q2_0_vnni.c — same translation-unit-local extern pattern
 * ternary_matmul_packed_vnni.c's own AVX-512 fallback uses. Returns 1 if it
 * handled the call, 0 if the caller should use the portable path below
 * (n not a multiple of 128, or larger than its stack-buffer ceiling —
 * never happens for this model's actual dims). */
extern int parallel_matmul_q2_0_vnni(float *out, const float *x, const uint8_t *w_q2_0,
                                      int n, int d, ThreadPool *tp);
#endif

void parallel_matmul_q2_0(float *out, const float *x, const uint8_t *w_q2_0,
                           int n, int d, ThreadPool *tp) {
#if TN_HAS_AVX512VNNI
    if (parallel_matmul_q2_0_vnni(out, x, w_q2_0, n, d, tp)) return;
#endif
    size_t row_bytes = (size_t)(n / Q2_0_BLOCK) * Q2_0_BYTES;
    MatmulQ2_0Args args = { .out = out, .x = x, .w = w_q2_0,
                             .n = n, .d = d, .row_bytes = row_bytes };
    if (!tp) { matmul_q2_0_task(&args, 0, 0, d); return; }
    threadpool_dispatch(tp, matmul_q2_0_task, &args, d);
}

/* ── Batched: k weight matrices, per-expert inputs (MoE) ─────────────────── *
 * Same VNNI-first / portable-fallback dispatch as the single-matrix path
 * above: matmul_q2_0_vnni.c's parallel_matmul_q2_0_batch_vnni() applies the
 * identical "w_enc bias trick" per-expert (each expert's row is independent,
 * so the single-matrix kernel's math carries over unchanged). No Q2_0-MoE
 * model has been wired into this engine yet — parallel_matmul_q2_0_batch is
 * currently unreachable from any real caller (moe_ffn.c only dispatches
 * q4k/q5_1/q5_0/q8_0 batch variants) — so this is verified via direct
 * synthetic unit tests (tests/test_q2_0_matmul.c) rather than an end-to-end
 * model run. Correctness parity with the portable path is what matters here,
 * not measured speedup, since there's no real hot-path call site today. */

#if TN_HAS_AVX512VNNI
extern int parallel_matmul_q2_0_batch_vnni(float **outs, float **xs,
                                            const uint8_t * const *ws,
                                            int n, int d, int k, ThreadPool *tp);
#endif

typedef struct {
    float        **outs;
    float        **xs;
    const uint8_t *const *ws;
    int n_in, n_out, k;
    size_t row_bytes;
} MatmulQ2_0BatchArgs;

static void matmul_q2_0_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ2_0BatchArgs *a = (MatmulQ2_0BatchArgs *)arg;
    int n_out = a->n_out;
    for (int r = start; r < end; r++) {
        int ei = r / n_out;
        int ri = r % n_out;
        a->outs[ei][ri] = dot_q2_0_row(a->ws[ei] + (size_t)ri * a->row_bytes,
                                        a->xs[ei], a->n_in);
    }
}

void parallel_matmul_q2_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp) {
#if TN_HAS_AVX512VNNI
    if (parallel_matmul_q2_0_batch_vnni(outs, xs, ws, n, d, k, tp)) return;
#endif
    size_t row_bytes = (size_t)(n / Q2_0_BLOCK) * Q2_0_BYTES;
    MatmulQ2_0BatchArgs args = {
        .outs = outs, .xs = xs, .ws = ws,
        .n_in = n, .n_out = d, .k = k,
        .row_bytes = row_bytes,
    };
    if (!tp) { matmul_q2_0_batch_task(&args, 0, 0, k * d); return; }
    threadpool_dispatch(tp, matmul_q2_0_batch_task, &args, k * d);
}
