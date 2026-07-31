#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
#include "core/weights.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if TN_HAS_AVX2
#include <immintrin.h>
#endif

/* K-4 R-3: Pre-quantised dispatch path for VNNI backend */
#if TN_HAS_AVX512VNNI
#include "math/quantize_i8.h"
#include "math/cpu_features.h"

void ternary_matmul_packed_vnni_preq(float *out,
                                      const int8_t *q_x, float act_scale,
                                      int32_t sum_qx,
                                      const tn_u8 *packed_w,
                                      int n, int d,
                                      const float *scales, int group_size);
extern void ternary_matmul_packed_vnni(float *out, const float *x, const tn_u8 *packed_w,
                                        int n, int d, const float *scales, int group_size);
#endif

/**
 * Argument struct passed to each worker thread via the thread pool.
 */
typedef struct {
    float *out;
    const float *x;
    const tn_i8 *w;
    int n;      /* input dimension (columns) */
    int d;      /* output dimension (rows) — used for bounds check */
    float scale;
} ParallelMatmulArgs;

/**
 * Worker task: compute a slice of output rows [start, end).
 *
 * Each thread calls the SIMD-dispatched matmul on its row range.
 * The weight pointer is offset to the start row, and we compute
 * (end - start) rows into the corresponding output slice.
 */
static void matmul_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulArgs *a = (ParallelMatmulArgs *)arg;

    int slice_rows = end - start;
    if (slice_rows <= 0) return;

    /*
     * w is row-major: row i starts at w[i * n].
     * We pass the sub-matrix starting at row 'start' with 'slice_rows' rows,
     * writing into out[start..end).
     */
    tn_ternary_matmul(
        a->out + start,
        a->x,
        a->w + (size_t)start * (size_t)a->n,
        a->n,
        slice_rows,
        a->scale
    );
}

void parallel_ternary_matmul(float *out, const float *x, const tn_i8 *w,
                              int n, int d, float scale, ThreadPool *tp) {
    if (!tp) {
        tn_ternary_matmul(out, x, w, n, d, scale);
        return;
    }

    ParallelMatmulArgs args = {
        .out   = out,
        .x     = x,
        .w     = w,
        .n     = n,
        .d     = d,
        .scale = scale
    };

    threadpool_dispatch(tp, matmul_task, &args, d);
}

typedef struct {
    float *out;
    const float *x;
    const tn_u8 *w;
    int n, d;
    float scale;
#if TN_HAS_AVX512VNNI
    /* K-4 R-3: pre-quantised activations (non-NULL when VNNI path is active) */
    const int8_t *q_x;
    float         act_scale;
    int32_t       sum_qx;
#endif
} ParallelMatmulPackedArgs;

static void matmul_packed_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulPackedArgs *a = (ParallelMatmulPackedArgs *)arg;
    int slice_rows = end - start;
    if (slice_rows <= 0) return;

    size_t row_bytes = ((size_t)a->n + 3) >> 2;

#if TN_HAS_AVX512VNNI
    /* K-4 R-3: use pre-quantised path when activations have been quantised
     * once by the dispatcher before thread launch (VNNI backend only). */
    if (a->q_x && a->act_scale > 0.0f) {
        ternary_matmul_packed_vnni_preq(
            a->out + start,
            a->q_x, a->act_scale, a->sum_qx,
            a->w + (size_t)start * row_bytes,
            a->n, slice_rows,
            &a->scale, 0
        );
        return;
    }
#endif

    tn_ternary_matmul_packed(
        a->out + start,
        a->x,
        a->w + (size_t)start * row_bytes,
        a->n,
        slice_rows,
        &a->scale,
        0 /* group_size 0 = per-matrix scales */
    );
}

void parallel_ternary_matmul_packed(float *out, const float *x, const tn_u8 *w,
                                     int n, int d, float scale, ThreadPool *tp) {
    if (!tp) {
        tn_ternary_matmul_packed(out, x, w, n, d, &scale, 0);
        return;
    }

#if TN_HAS_AVX512VNNI
    /*
     * Pre-quantize activations ONCE in the dispatcher, then pass to all
     * workers.  Without this, each of T workers independently quantizes
     * the same input vector — 4× redundant at T=4, wasting ~0.8 ms/token.
     *
     * This also enables the _preq kernel path which skips per-worker
     * quantization entirely, saving stack allocation of q_x[16384] per
     * worker and improving instruction cache locality.
     */
    int8_t q_x_buf[16384];
    int8_t *q_x = NULL;
    float act_scale = 0.0f;
    int32_t sum_qx = 0;
    if (n <= 16384) {
        act_scale = quantize_row_to_i8_avx512(x, q_x_buf, n);
        if (act_scale > 0.0f) {
            sum_qx = sum_i8_avx512(q_x_buf, n);
            q_x = q_x_buf;
        }
    }
#endif

    ParallelMatmulPackedArgs args = {
        .out = out,
        .x = x,
        .w = w,
        .n = n,
        .d = d,
        .scale = scale,
#if TN_HAS_AVX512VNNI
        .q_x = q_x,
        .act_scale = act_scale,
        .sum_qx = sum_qx
#endif
    };

    threadpool_dispatch(tp, matmul_packed_task, &args, d);
}

/* ── Layer-level pre-quantisation API ──────────────────────────────────── */

int tn_preq_prepare(TnPreqActivation *preq, int8_t *buf, const float *x, int n) {
    preq->valid = 0;
    preq->q_x = NULL;
    preq->act_scale = 0.0f;
    preq->sum_qx = 0;
#if TN_HAS_AVX512VNNI
    if (n <= 0 || n > 16384) return 0;
    float scale = quantize_row_to_i8_avx512(x, buf, n);
    if (scale <= 0.0f) return 0;
    preq->q_x      = buf;
    preq->act_scale = scale;
    preq->sum_qx    = sum_i8_avx512(buf, n);
    preq->valid     = 1;
    return 1;
#else
    (void)buf; (void)x; (void)n;
    return 0;
#endif
}

void parallel_ternary_matmul_packed_preq(float *out, const float *x, const tn_u8 *w,
                                          int n, int d, float scale,
                                          const TnPreqActivation *preq,
                                          ThreadPool *tp) {
    if (!tp) {
        tn_ternary_matmul_packed(out, x, w, n, d, &scale, 0);
        return;
    }

#if TN_HAS_AVX512VNNI
    if (preq && preq->valid) {
        /* Fast path: activations already quantised by caller — no re-quantisation */
        ParallelMatmulPackedArgs args = {
            .out       = out,
            .x         = x,
            .w         = w,
            .n         = n,
            .d         = d,
            .scale     = scale,
            .q_x       = preq->q_x,
            .act_scale = preq->act_scale,
            .sum_qx    = preq->sum_qx
        };
        threadpool_dispatch(tp, matmul_packed_task, &args, d);
        return;
    }
#else
    (void)preq;
#endif

    /* Fallback: standard quantise-per-call path */
    parallel_ternary_matmul_packed(out, x, w, n, d, scale, tp);
}


typedef struct {
    float *out;
    const float *x;
    const float *w;
    int n, d;
} ParallelMatmulF32Args;

static void matmul_f32_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulF32Args *a = (ParallelMatmulF32Args *)arg;
    for (int i = start; i < end; i++) {
        a->out[i] = tn_vec_dot(a->x, a->w + (size_t)i * a->n, a->n);
    }
}

void parallel_matmul_float32(float *out, const float *x, const float *w,
                             int n, int d, ThreadPool *tp) {
    if (!tp) {
        for (int i = 0; i < d; i++) {
            out[i] = tn_vec_dot(x, w + (size_t)i * n, n);
        }
        return;
    }

    ParallelMatmulF32Args args = {
        .out = out,
        .x = x,
        .w = w,
        .n = n,
        .d = d
    };

    threadpool_dispatch(tp, matmul_f32_task, &args, d);
}

typedef struct {
    float       *out;
    const float *x;
    const tn_u16 *w;
    int n, d;
} ParallelMatmulBF16Args;

static void matmul_bf16_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulBF16Args *a = (ParallelMatmulBF16Args *)arg;

#if TN_HAS_AVX512
    /*
     * AVX-512 BF16 → float32 matmul: 4 independent 16-wide accumulators
     * (64 elements/iter), then a single-accumulator 16-wide loop for the
     * remainder. BF16 = upper 16 bits of float32; shift-left-16 reinterprets
     * as float. Multi-accumulator rationale: see matmul_f16.c's header
     * comment (2026-07-24 RCA vs llama.cpp/ggml) — this is the LM-head
     * classifier kernel, the single largest matmul per token for most
     * models (dim × vocab_size), so it matters most here.
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        int j = 0;
        for (; j + 63 < a->n; j += 64) {
            acc0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)&row[j])), 16)),
                                    _mm512_loadu_ps(&a->x[j]), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)&row[j + 16])), 16)),
                                    _mm512_loadu_ps(&a->x[j + 16]), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)&row[j + 32])), 16)),
                                    _mm512_loadu_ps(&a->x[j + 32]), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)&row[j + 48])), 16)),
                                    _mm512_loadu_ps(&a->x[j + 48]), acc3);
        }
        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        for (; j + 15 < a->n; j += 16) {
            __m256i bf16_16 = _mm256_loadu_si256((const __m256i *)&row[j]);
            __m512i i32_16  = _mm512_cvtepu16_epi32(bf16_16);
            __m512  wv      = _mm512_castsi512_ps(_mm512_slli_epi32(i32_16, 16));
            __m512  xv      = _mm512_loadu_ps(&a->x[j]);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        float val = _mm512_reduce_add_ps(acc);
        /* AVX2 tail (8-wide) */
        if (j + 7 < a->n) {
            __m128i bf16_8 = _mm_loadu_si128((const __m128i *)&row[j]);
            __m256i i32_8  = _mm256_cvtepu16_epi32(bf16_8);
            __m256  wv8    = _mm256_castsi256_ps(_mm256_slli_epi32(i32_8, 16));
            __m256  xv8    = _mm256_loadu_ps(&a->x[j]);
            __m256  p8     = _mm256_mul_ps(wv8, xv8);
            __m128  h = _mm256_extractf128_ps(p8,1), l = _mm256_castps256_ps128(p8);
            __m128  s4 = _mm_add_ps(l,h);
            __m128  sh = _mm_movehdup_ps(s4);
            val += _mm_cvtss_f32(_mm_add_ss(_mm_add_ps(s4,sh), _mm_movehl_ps(sh,sh)));
            j += 8;
        }
        for (; j < a->n; j++) {
            tn_u32 bits = (tn_u32)row[j] << 16;
            float wval; __builtin_memcpy(&wval, &bits, sizeof(wval));
            val += wval * a->x[j];
        }
        a->out[i] = val;
    }
#elif TN_HAS_AVX2
    /*
     * AVX2 BF16 → float32 matmul: 4 independent 8-wide accumulators
     * (32 elements/iter — matches llama.cpp/ggml's GGML_F16_STEP=32/EPR=8
     * layout on this ISA), then a single-accumulator 8-wide loop for the
     * remainder.
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 31 < a->n; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&row[j])), 16)),
                                    _mm256_loadu_ps(&a->x[j]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&row[j + 8])), 16)),
                                    _mm256_loadu_ps(&a->x[j + 8]), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&row[j + 16])), 16)),
                                    _mm256_loadu_ps(&a->x[j + 16]), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&row[j + 24])), 16)),
                                    _mm256_loadu_ps(&a->x[j + 24]), acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; j + 7 < a->n; j += 8) {
            __m128i  bf16_8  = _mm_loadu_si128((const __m128i *)&row[j]);
            __m256i  i32_8   = _mm256_cvtepu16_epi32(bf16_8);
            __m256   wv      = _mm256_castsi256_ps(_mm256_slli_epi32(i32_8, 16));
            __m256   xv      = _mm256_loadu_ps(&a->x[j]);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        __m128 hi   = _mm256_extractf128_ps(acc, 1);
        __m128 lo   = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        float  val  = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2)));
        for (; j < a->n; j++) {
            tn_u32 bits = (tn_u32)row[j] << 16;
            float wval; __builtin_memcpy(&wval, &bits, sizeof(wval));
            val += wval * a->x[j];
        }
        a->out[i] = val;
    }
#else
    /* Scalar fallback */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        float acc = 0.0f;
        for (int j = 0; j < a->n; j++) {
            tn_u32 bits = (tn_u32)row[j] << 16;
            float wval; __builtin_memcpy(&wval, &bits, sizeof(wval));
            acc += wval * a->x[j];
        }
        a->out[i] = acc;
    }
#endif
}

void parallel_matmul_bf16(float *out, const float *x, const tn_u16 *w,
                           int n, int d, ThreadPool *tp) {
    if (!tp) {
        matmul_bf16_task(&(ParallelMatmulBF16Args){out, x, w, n, d}, 0, 0, d);
        return;
    }

    ParallelMatmulBF16Args args = { .out = out, .x = x, .w = w, .n = n, .d = d };
    threadpool_dispatch(tp, matmul_bf16_task, &args, d);
}

/* ── INT8 classifier matmul ─────────────────────────────────────────────────
 *
 * Block-wise INT8 weights: TN_CLS_QUANT_BLOCK-element blocks, each with its
 * own float scale (weights_build_classifier_quant, weights.c).
 * out[i] = sum_b (sum_{j in block b} w[i*n+j] * x[j]) * scales[i*n_blocks+b]
 *
 * Reads 1 byte per weight vs 2 bytes for BF16, halving the 656 MB LM head
 * bandwidth to 328 MB.
 *
 * 2026-07-31: switched from one scale per row to one scale per
 * TN_CLS_QUANT_BLOCK-element block (see weights.h/weights.c — a real
 * quantization-quality gap reported in GitHub issue #27). This drops the
 * VNNI dpbusds fast path: its bias correction (true_dot = dpbusds_result -
 * 128*sum_qx) is only valid when one scale applies to the WHOLE row's raw
 * dot product, which no longer holds once every block has its own scale.
 * The portable FMA path below (previously the fallback) is now the only
 * path, applied per block instead of per row — see docs/ai/mistakes.md for
 * the measured throughput tradeoff.
 *
 * SIMD strategy per block: load up to TN_CLS_QUANT_BLOCK INT8 values,
 * sign-extend to 32-bit, convert to float32, then FMA with activations.
 */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;        /* unsigned uint8 weights (original + 128 bias) */
    const float *scales;    /* d * n_blocks block scales */
    int n, d, n_blocks;
} ParallelMatmulI8Args;

static void matmul_i8_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const ParallelMatmulI8Args *a = (const ParallelMatmulI8Args *)arg;
    const int nb = a->n_blocks;

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * a->n;
        const float *row_scales = a->scales + (size_t)i * nb;
        float total = 0.0f;

        for (int b = 0; b < nb; b++) {
            int blk_start = b * TN_CLS_QUANT_BLOCK;
            int blk_len = a->n - blk_start;
            if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
            const tn_u8 *brow = row + blk_start;
            const float *bx = a->x + blk_start;
            int j = 0;
            float val;

#if TN_HAS_AVX512
            __m512 acc512 = _mm512_setzero_ps();
            for (; j + 15 < blk_len; j += 16) {
                __m128i u8_16  = _mm_loadu_si128((const __m128i *)&brow[j]);
                __m512i u32_16 = _mm512_cvtepu8_epi32(u8_16);
                __m512i s32_16 = _mm512_sub_epi32(u32_16, _mm512_set1_epi32(128));
                __m512  wv     = _mm512_cvtepi32_ps(s32_16);
                __m512  xv     = _mm512_loadu_ps(&bx[j]);
                acc512 = _mm512_fmadd_ps(wv, xv, acc512);
            }
            val = _mm512_reduce_add_ps(acc512);
#elif TN_HAS_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (; j + 7 < blk_len; j += 8) {
                __m128i u8_8  = _mm_loadl_epi64((const __m128i *)&brow[j]);
                __m256i u32_8 = _mm256_cvtepu8_epi32(u8_8);
                __m256i s32_8 = _mm256_sub_epi32(u32_8, _mm256_set1_epi32(128));
                __m256  wv    = _mm256_cvtepi32_ps(s32_8);
                __m256  xv    = _mm256_loadu_ps(&bx[j]);
                acc = _mm256_fmadd_ps(wv, xv, acc);
            }
            __m128 hi   = _mm256_extractf128_ps(acc, 1);
            __m128 lo   = _mm256_castps256_ps128(acc);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sum2 = _mm_add_ps(sum4, shuf);
            val = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2)));
#else
            val = 0.0f;
#endif
            for (; j < blk_len; j++) {
                val += (float)((int)brow[j] - 128) * bx[j];
            }
            total += val * row_scales[b];
        }
        a->out[i] = total;
    }
}

/* ── INT4 classifier matmul ─────────────────────────────────────────────────
 *
 * INT4 packed weights: 2 values per byte (low nibble = w[2k], high = w[2k+1]).
 * Unsigned storage with +8 bias: signed value + 8 → [1, 15].
 * Block-wise scales (TN_CLS_QUANT_BLOCK elements/block) — same 2026-07-31
 * change and rationale as the INT8 matmul above (GitHub issue #27); the
 * VNNI dpbusds fast path was dropped for the same reason (its single
 * row-wide bias correction doesn't hold once every block has its own
 * scale) in favor of a portable per-block FMA path.
 */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;        /* packed INT4 weights */
    const float *scales;    /* d * n_blocks block scales */
    int n, d, n_blocks;
} ParallelMatmulI4Args;

static void matmul_i4_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const ParallelMatmulI4Args *a = (const ParallelMatmulI4Args *)arg;
    size_t row_bytes = ((size_t)a->n + 1) / 2;
    const int nb = a->n_blocks;

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * row_bytes;
        const float *row_scales = a->scales + (size_t)i * nb;
        float total = 0.0f;

        for (int b = 0; b < nb; b++) {
            int blk_start = b * TN_CLS_QUANT_BLOCK;
            int blk_len = a->n - blk_start;
            if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
            const float *bx = a->x + blk_start;
            /* blk_start is always even (TN_CLS_QUANT_BLOCK=32), so
             * blk_start/2 lands exactly on a packed-byte boundary. */
            const tn_u8 *brow = row + blk_start / 2;
            int j = 0;
            float val;

#if TN_HAS_AVX512
            __m512 acc512 = _mm512_setzero_ps();
            for (; j + 15 < blk_len; j += 16) {
                __m128i packed8 = _mm_loadl_epi64((const __m128i *)(brow + j/2));
                __m128i mask4 = _mm_set1_epi8(0x0F);
                __m128i lo_8 = _mm_and_si128(packed8, mask4);
                __m128i hi_8 = _mm_and_si128(_mm_srli_epi16(packed8, 4), mask4);
                __m128i interleaved = _mm_unpacklo_epi8(lo_8, hi_8);
                __m512i u32_16 = _mm512_cvtepu8_epi32(interleaved);
                __m512i s32_16 = _mm512_sub_epi32(u32_16, _mm512_set1_epi32(8));
                __m512  wv     = _mm512_cvtepi32_ps(s32_16);
                __m512  xv     = _mm512_loadu_ps(&bx[j]);
                acc512 = _mm512_fmadd_ps(wv, xv, acc512);
            }
            val = _mm512_reduce_add_ps(acc512);
#elif TN_HAS_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (; j + 7 < blk_len; j += 8) {
                /* Unpack 8 INT4 values from 4 packed bytes */
                tn_u32 packed4;
                memcpy(&packed4, brow + j/2, 4);
                __m128i packed8 = _mm_cvtsi32_si128((int)packed4);
                __m128i mask4 = _mm_set1_epi8(0x0F);
                __m128i lo_8 = _mm_and_si128(packed8, mask4);
                __m128i hi_8 = _mm_and_si128(_mm_srli_epi16(packed8, 4), mask4);
                __m128i interleaved = _mm_unpacklo_epi8(lo_8, hi_8); /* 8 uint4 as uint8 (low 8 bytes) */
                __m256i u32_8 = _mm256_cvtepu8_epi32(interleaved);
                __m256i s32_8 = _mm256_sub_epi32(u32_8, _mm256_set1_epi32(8));
                __m256  wv    = _mm256_cvtepi32_ps(s32_8);
                __m256  xv    = _mm256_loadu_ps(&bx[j]);
                acc = _mm256_fmadd_ps(wv, xv, acc);
            }
            __m128 hi   = _mm256_extractf128_ps(acc, 1);
            __m128 lo   = _mm256_castps256_ps128(acc);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sum2 = _mm_add_ps(sum4, shuf);
            val = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2)));
#else
            val = 0.0f;
#endif
            for (; j < blk_len; j++) {
                int abs_j = blk_start + j;
                int nibble;
                if (abs_j & 1) nibble = (row[abs_j/2] >> 4) & 0x0F;
                else nibble = row[abs_j/2] & 0x0F;
                val += (float)(nibble - 8) * a->x[abs_j];
            }
            total += val * row_scales[b];
        }
        a->out[i] = total;
    }
}

void parallel_matmul_i4(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp) {
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    ParallelMatmulI4Args args = {
        .out = out, .x = x, .w = w, .scales = scales, .n = n, .d = d, .n_blocks = n_blocks,
    };

    if (!tp) {
        matmul_i4_task(&args, 0, 0, d);
        return;
    }

    threadpool_dispatch(tp, matmul_i4_task, &args, d);
}

void parallel_matmul_i8(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp) {
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    ParallelMatmulI8Args args = {
        .out = out, .x = x, .w = w, .scales = scales, .n = n, .d = d, .n_blocks = n_blocks,
    };

    if (!tp) {
        matmul_i8_task(&args, 0, 0, d);
        return;
    }

    threadpool_dispatch(tp, matmul_i8_task, &args, d);
}
