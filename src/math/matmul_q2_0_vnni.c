/*
 * parallel_matmul_q2_0_vnni — AVX-512 VNNI accelerated Q2_0 x F32 matmul.
 *
 * matmul_q2_0.c's fallback kernel decodes each 128-element block into a
 * stack float buffer, then does an AVX2 FMA reduction — correct, but far
 * from this host's own measured bandwidth ceiling (confirmed benchmarking
 * the real Ternary-Bonsai-27B model: ~0.12 tok/s achieved vs. a ~10 tok/s
 * ceiling the engine's own hardware profiler computes for this host/model).
 * This file replaces that decode-then-FMA path with int8 VNNI dot products,
 * mirroring math/ternary_matmul_packed_vnni.c's proven "w_enc bias trick"
 * exactly — GGUF Q2_0's code->value mapping (code=0/1/2 -> -1/0/+1, i.e.
 * w_enc = w+1) is bit-for-bit identical to this project's native packed-
 * ternary format; only the block framing differs (Q2_0 packs a per-128-
 * element FP16 scale directly into the row bytes instead of a separate
 * external scales[] array, i.e. it's structurally the packed kernel's
 * "per-group scale" mode with group_size=128, scale read inline).
 *
 * Algorithm (see ternary_matmul_packed_vnni.c's header comment for the full
 * derivation of the bias trick):
 *   dpbusds(w_enc_u8, q_x_i8) = dot(w, q_x) + sum(q_x)
 *   true_dot = dpbusds_result - sum_qx_of_this_block
 *   partial  = true_dot * block_scale
 *   out[i]   = (sum over blocks of partial) * act_scale
 *
 * Saturation: w_enc in {0,1,2}, q_x in [-127,127] -> per-group-of-4 max
 * = 4*2*127 = 1016; a 128-element block's full accumulation
 * = 128/4 * 1016 = 32,512 << INT32_MAX — no saturation risk even summed
 * across the largest n used by this model (17408, the FFN down-projection).
 */

#include "core/platform.h"

#if TN_HAS_AVX512VNNI

#include "math/matmul_q2_0.h"
#include "math/quantize_i8.h"
#include "math/bitunpack2_vnni.h"
#include "threading/thread_pool.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

/* Same 128-element/34-byte framing as matmul_q2_0.c — see that file's header
 * comment for why this is PrismML's group-128 packing, not mainline ggml's
 * group-64 block_q2_0. */
#define Q2_0V_BLOCK  128
#define Q2_0V_BYTES  34

/* Largest n across every parallel_matmul_q2_0 call site in this model
 * (FFN down-projection, hidden_dim=17408) rounded up generously; n larger
 * than this (or not a multiple of 128) falls back to matmul_q2_0.c's
 * portable decode+FMA path — deterministic, same policy as
 * ternary_matmul_packed_vnni.c's own TN_VNNI_MAX_N fallback. */
#define TN_Q2_0V_MAX_N      20480
#define TN_Q2_0V_MAX_BLOCKS (TN_Q2_0V_MAX_N / Q2_0V_BLOCK)

/* NOTE (2026-07-17, negative result): a "run inline below 64 output rows"
 * dispatch threshold was tried here (targeting DeltaNet's 48-row beta/alpha
 * projections, 96 pool dispatches/token) and REVERTED — an interleaved
 * end-to-end A/B measured it slightly SLOWER both rounds (3.24/3.31 vs
 * 3.30/3.45 tok/s): this pool's dispatch is cheap enough that even a 48-row
 * gemv benefits from the 4-way split. Don't re-add without measuring; see
 * docs/architecture/CEILING_CALCULATION.md §7. */

#define TN_PREFETCH_ROWS 8

/* Prefetch every 64-byte cache line of the target row — mirrors
 * ternary_matmul_packed_vnni.c's TN_PREFETCH_ROW_ALL (a Q2_0 row is
 * n_blocks*34 bytes, several cache lines; one prefetch per row leaves most
 * of it un-prefetched). */
#define Q2_0V_PREFETCH_ROW_ALL(ptr, row_bytes, hint)                       \
    do {                                                                    \
        const char *_p = (const char *)(ptr);                              \
        size_t _rb = (row_bytes);                                          \
        for (size_t _off = 0; _off < _rb; _off += 64)                     \
            _mm_prefetch(_p + _off, (hint));                               \
    } while (0)

static inline float q2_0v_f16_to_f32(uint16_t h) {
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
 * Dot product of one Q2_0 row (n_blocks * 34 bytes) against pre-quantised
 * int8 activations, using two 64-lane VNNI dot products per 128-element
 * block (one per unpack64_to_wenc_u8 call — that helper unpacks exactly 64
 * codes from 16 bytes). Returns the un-descaled-by-act_scale total; the
 * caller multiplies by act_scale once, since it's the same constant for
 * every block of every row in this matmul call.
 *
 * Reference variant (2026-07-17): this was the production kernel until the
 * optimized variant below replaced it. Kept compiled (a) as the baseline
 * leg of tools/bench_q2_0's in-binary A/B, so kernel comparisons are immune
 * to run-to-run host drift on virtualized benchmark hosts, and (b) as an
 * independent same-math implementation for correctness cross-checks. Its
 * two known inefficiencies, deliberately preserved: a full cross-lane
 * _mm512_reduce_add_epi32 per 128-element block, and a branchy scalar
 * fp16->f32 scale conversion per block.
 */
static float dot_q2_0_row_vnni_ref(const uint8_t *row_q2_0, const int8_t *q_x,
                                    const int32_t *sum_qx_blocks, int n_blocks) {
    float total = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q2_0 + (size_t)b * Q2_0V_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = q2_0v_f16_to_f32(d_bits);
        const uint8_t *qs = blk + 2;                       /* 32 bytes, 128 codes */
        const int8_t *qxb = q_x + (size_t)b * Q2_0V_BLOCK;

        __m512i wenc0 = tn_unpack64_to_wenc_u8(qs);
        __m512i wenc1 = tn_unpack64_to_wenc_u8(qs + 16);
        __m512i qxv0  = _mm512_loadu_si512((const void *)qxb);
        __m512i qxv1  = _mm512_loadu_si512((const void *)(qxb + 64));

        __m512i acc = _mm512_setzero_si512();
        acc = _mm512_dpbusds_epi32(acc, wenc0, qxv0);
        acc = _mm512_dpbusds_epi32(acc, wenc1, qxv1);
        int32_t vnni_result = _mm512_reduce_add_epi32(acc);

        int32_t true_dot = vnni_result - sum_qx_blocks[b];
        total += (float)true_dot * d;
    }

    return total;
}

/*
 * Optimized row dot (2026-07-17): same math as the _ref variant, two
 * structural fixes (docs/architecture/CEILING_CALCULATION.md §7 option A):
 *
 * A1 — no per-block horizontal reduce. The 16 int32 lane-partials from the
 *   two dpbusds are exact (each lane accumulates 8 products bounded by
 *   3*127, far below float's 2^24 integer-exact range), so scaling them by
 *   the block's d and FMA-ing into a per-row float vector accumulator gives
 *   sum(acc)*d without ever collapsing lanes; one _mm512_reduce_add_ps per
 *   ROW replaces one _mm512_reduce_add_epi32 per BLOCK (40 for dim=5120).
 *   The w_enc bias correction (-sum_qx*d per block) cannot ride in the
 *   vector accumulator (it's a per-block scalar), so it accumulates in a
 *   separate scalar FMA chain and is subtracted after the final reduce.
 *
 * A2 — F16C hardware fp16->f32 for the block scale instead of the branchy
 *   scalar bit-twiddle (identical IEEE semantics incl. subnormals/inf).
 */
static float dot_q2_0_row_vnni(const uint8_t *row_q2_0, const int8_t *q_x,
                                const int32_t *sum_qx_blocks, int n_blocks) {
    __m512 accf = _mm512_setzero_ps();
    float corr = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row_q2_0 + (size_t)b * Q2_0V_BYTES;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)d_bits)));
        const uint8_t *qs = blk + 2;                       /* 32 bytes, 128 codes */
        const int8_t *qxb = q_x + (size_t)b * Q2_0V_BLOCK;

        __m512i wenc0 = tn_unpack64_to_wenc_u8(qs);
        __m512i wenc1 = tn_unpack64_to_wenc_u8(qs + 16);
        __m512i qxv0  = _mm512_loadu_si512((const void *)qxb);
        __m512i qxv1  = _mm512_loadu_si512((const void *)(qxb + 64));

        __m512i acc = _mm512_dpbusds_epi32(_mm512_setzero_si512(), wenc0, qxv0);
        acc = _mm512_dpbusds_epi32(acc, wenc1, qxv1);

        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(d), accf);
        corr += (float)sum_qx_blocks[b] * d;
    }

    return _mm512_reduce_add_ps(accf) - corr;
}

typedef struct {
    float         *out;
    const int8_t  *q_x;
    const int32_t *sum_qx_blocks;
    float          act_scale;
    const uint8_t *w;
    int            n_blocks;
    size_t         row_bytes;
    int            use_ref;   /* 1 = benchmark-baseline row dot (see above) */
} MatmulQ2_0VnniArgs;

static void matmul_q2_0_vnni_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ2_0VnniArgs *a = (MatmulQ2_0VnniArgs *)arg;
    /* Branch once per task, not per row — and keep two separate loops rather
     * than a function pointer, so each (static, single-call-site) row dot
     * still inlines into its loop. An indirect call would deoptimize the
     * exact code path this dual-variant setup exists to measure. */
    if (a->use_ref) {
        for (int i = start; i < end; i++) {
            if (i + TN_PREFETCH_ROWS < end)
                Q2_0V_PREFETCH_ROW_ALL(a->w + (size_t)(i + TN_PREFETCH_ROWS) * a->row_bytes,
                                        a->row_bytes, _MM_HINT_T1);
            float total = dot_q2_0_row_vnni_ref(a->w + (size_t)i * a->row_bytes, a->q_x,
                                                 a->sum_qx_blocks, a->n_blocks);
            a->out[i] = total * a->act_scale;
        }
    } else {
        for (int i = start; i < end; i++) {
            if (i + TN_PREFETCH_ROWS < end)
                Q2_0V_PREFETCH_ROW_ALL(a->w + (size_t)(i + TN_PREFETCH_ROWS) * a->row_bytes,
                                        a->row_bytes, _MM_HINT_T1);
            float total = dot_q2_0_row_vnni(a->w + (size_t)i * a->row_bytes, a->q_x,
                                             a->sum_qx_blocks, a->n_blocks);
            a->out[i] = total * a->act_scale;
        }
    }
}

/*
 * Returns 1 if handled (VNNI path taken), 0 if the caller (matmul_q2_0.c)
 * should fall back to the portable decode+FMA kernel — n not a multiple of
 * 128, or larger than this file's stack-buffer ceiling (never happens for
 * this model's actual dims; see TN_Q2_0V_MAX_N above).
 */
static int matmul_q2_0_vnni_run(float *out, const float *x, const uint8_t *w_q2_0,
                                 int n, int d, ThreadPool *tp, int use_ref) {
    if (n <= 0 || n % Q2_0V_BLOCK != 0 || n > TN_Q2_0V_MAX_N)
        return 0;

    int n_blocks = n / Q2_0V_BLOCK;
    size_t row_bytes = (size_t)n_blocks * Q2_0V_BYTES;

    int8_t q_x[TN_Q2_0V_MAX_N];
    float act_scale = quantize_row_to_i8_avx512(x, q_x, n);
    if (act_scale == 0.0f) {
        memset(out, 0, (size_t)d * sizeof(float));
        return 1;
    }

    int32_t sum_qx_blocks[TN_Q2_0V_MAX_BLOCKS];
    for (int b = 0; b < n_blocks; b++)
        sum_qx_blocks[b] = sum_i8(q_x + (size_t)b * Q2_0V_BLOCK, Q2_0V_BLOCK);

    MatmulQ2_0VnniArgs args = {
        .out = out, .q_x = q_x, .sum_qx_blocks = sum_qx_blocks,
        .act_scale = act_scale, .w = w_q2_0, .n_blocks = n_blocks,
        .row_bytes = row_bytes, .use_ref = use_ref,
    };
    if (!tp) { matmul_q2_0_vnni_task(&args, 0, 0, d); return 1; }
    threadpool_dispatch(tp, matmul_q2_0_vnni_task, &args, d);
    return 1;
}

int parallel_matmul_q2_0_vnni(float *out, const float *x, const uint8_t *w_q2_0,
                               int n, int d, ThreadPool *tp) {
    return matmul_q2_0_vnni_run(out, x, w_q2_0, n, d, tp, /*use_ref=*/0);
}

/* Benchmark baseline: the pre-2026-07-17 row dot, exported ONLY for
 * tools/bench_q2_0's in-binary A/B and correctness cross-checks. Not part of
 * the engine's dispatch — production always takes the optimized variant. */
int parallel_matmul_q2_0_vnni_ref(float *out, const float *x, const uint8_t *w_q2_0,
                                   int n, int d, ThreadPool *tp) {
    return matmul_q2_0_vnni_run(out, x, w_q2_0, n, d, tp, /*use_ref=*/1);
}

/* ── Batched: k weight matrices, per-expert inputs (MoE) ─────────────────
 * Same VNNI kernel as the single-matrix path above, applied per expert.
 * Each expert has its own activation vector, so each needs its own
 * quantization (act_scale, per-block sum_qx) — unlike the single-matrix
 * path, k is a caller-supplied model dimension (expert count), not a
 * compile-time bound, so these are heap-allocated for this call rather
 * than fixed-size stack arrays. */
typedef struct {
    float        **outs;
    const int8_t **q_xs;
    const float   *act_scales;
    const int32_t **sum_qx_blocks_per_expert;
    const uint8_t *const *ws;
    int            n_out, n_blocks;
    size_t         row_bytes;
} MatmulQ2_0BatchVnniArgs;

static void matmul_q2_0_batch_vnni_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ2_0BatchVnniArgs *a = (MatmulQ2_0BatchVnniArgs *)arg;
    int n_out = a->n_out;
    for (int r = start; r < end; r++) {
        int ei = r / n_out;
        int ri = r % n_out;
        if (a->act_scales[ei] == 0.0f) { a->outs[ei][ri] = 0.0f; continue; }
        float total = dot_q2_0_row_vnni(a->ws[ei] + (size_t)ri * a->row_bytes,
                                         a->q_xs[ei], a->sum_qx_blocks_per_expert[ei],
                                         a->n_blocks);
        a->outs[ei][ri] = total * a->act_scales[ei];
    }
}

/* Returns 1 if handled, 0 if the caller should fall back (same conditions
 * as parallel_matmul_q2_0_vnni, checked against the shared n across every
 * expert — all experts share one weight-matrix shape). */
int parallel_matmul_q2_0_batch_vnni(float **outs, float **xs,
                                     const uint8_t * const *ws,
                                     int n, int d, int k, ThreadPool *tp) {
    if (n <= 0 || n % Q2_0V_BLOCK != 0 || n > TN_Q2_0V_MAX_N || k <= 0)
        return 0;

    int n_blocks = n / Q2_0V_BLOCK;
    size_t row_bytes = (size_t)n_blocks * Q2_0V_BYTES;

    int8_t   *q_x_storage        = (int8_t  *)malloc((size_t)k * (size_t)n * sizeof(int8_t));
    int32_t  *sum_qx_storage     = (int32_t *)malloc((size_t)k * (size_t)n_blocks * sizeof(int32_t));
    float    *act_scales         = (float   *)malloc((size_t)k * sizeof(float));
    const int8_t  **q_xs         = (const int8_t  **)malloc((size_t)k * sizeof(int8_t *));
    const int32_t **sum_qx_ptrs  = (const int32_t **)malloc((size_t)k * sizeof(int32_t *));
    if (!q_x_storage || !sum_qx_storage || !act_scales || !q_xs || !sum_qx_ptrs) {
        free(q_x_storage); free(sum_qx_storage); free(act_scales); free(q_xs); free(sum_qx_ptrs);
        return 0; /* caller falls back to the portable path on OOM too */
    }

    for (int e = 0; e < k; e++) {
        int8_t  *qx = q_x_storage    + (size_t)e * (size_t)n;
        int32_t *sq = sum_qx_storage + (size_t)e * (size_t)n_blocks;
        q_xs[e] = qx;
        sum_qx_ptrs[e] = sq;
        act_scales[e] = quantize_row_to_i8_avx512(xs[e], qx, n);
        if (act_scales[e] == 0.0f) { memset(outs[e], 0, (size_t)d * sizeof(float)); continue; }
        for (int b = 0; b < n_blocks; b++)
            sq[b] = sum_i8(qx + (size_t)b * Q2_0V_BLOCK, Q2_0V_BLOCK);
    }

    MatmulQ2_0BatchVnniArgs args = {
        .outs = outs, .q_xs = q_xs, .act_scales = act_scales,
        .sum_qx_blocks_per_expert = sum_qx_ptrs, .ws = ws,
        .n_out = d, .n_blocks = n_blocks, .row_bytes = row_bytes,
    };
    if (!tp) matmul_q2_0_batch_vnni_task(&args, 0, 0, k * d);
    else     threadpool_dispatch(tp, matmul_q2_0_batch_vnni_task, &args, k * d);

    free(q_x_storage); free(sum_qx_storage); free(act_scales); free(q_xs); free(sum_qx_ptrs);
    return 1;
}

#endif /* TN_HAS_AVX512VNNI */
