#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "threading/thread_pool.h"
#include "core/platform.h"
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
    if (!preq) return 0;
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
     * AVX-512 BF16 → float32 matmul (16-wide).
     * BF16 = upper 16 bits of float32; shift-left-16 reinterprets as float.
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;

        __m512 acc = _mm512_setzero_ps();
        int j = 0;
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
            __m128  s2 = _mm_add_ps(s4, sh);
            val += _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehl_ps(s2, s2)));
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
     * AVX2 BF16 → float32 matmul (8-wide).
     */
    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
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
 * Per-row INT8 weights with per-row float scales.
 * out[i] = (sum_j w[i*n+j] * x[j]) * scales[i]
 *
 * Reads 1 byte per weight vs 2 bytes for BF16, halving the 656 MB LM head
 * bandwidth to 328 MB.  On bandwidth-bound classifiers (21 ms out of 61 ms
 * total), this yields a significant speedup.
 *
 * SIMD strategy: load 16 INT8 values, sign-extend to 32-bit, convert to
 * float32, then FMA with activations.  Same throughput as BF16 per element,
 * but half the memory traffic.
 */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;        /* unsigned uint8 weights (original + 128 bias) */
    const float *scales;    /* per-row dequantization scales */
    int n, d;
#if TN_HAS_AVX512VNNI
    const int8_t *q_x;     /* pre-quantized activations (signed int8) */
    float         act_scale;
    int32_t       sum_qx;
#endif
} ParallelMatmulI8Args;

static void matmul_i8_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulI8Args *a = (ParallelMatmulI8Args *)arg;

#if TN_HAS_AVX512VNNI
    /*
     * VNNI dpbusds path: 64 elements per iteration (4x faster than FMA).
     *
     * Weights stored as unsigned uint8 (w_signed + 128).
     * Activations quantized to signed int8 (q_x).
     * dpbusds(w_u8, q_x_s8) = Σ w_u8[j] × q_x[j]
     *   = Σ (w_signed[j]+128) × q_x[j]
     *   = dot(w_signed, q_x) + 128 × sum_qx
     * true_dot = dpbusds_result - 128 × sum_qx
     * out[i] = true_dot × act_scale × w_scale[i]
     */
    if (a->q_x && a->act_scale > 0.0f) {
        int32_t bias_128 = 128 * a->sum_qx;
        for (int i = start; i < end; i++) {
            const tn_u8 *row = a->w + (size_t)i * a->n;

            __m512i acc = _mm512_setzero_si512();
            int j = 0;
            for (; j + 63 < a->n; j += 64) {
                __m512i wv  = _mm512_loadu_si512((const __m512i *)&row[j]);
                __m512i qxv = _mm512_loadu_si512((const __m512i *)&a->q_x[j]);
                acc = _mm512_dpbusds_epi32(acc, wv, qxv);
            }

            int32_t result = _mm512_reduce_add_epi32(acc);

            /* Scalar tail (< 64 elements) */
            for (; j < a->n; j++) {
                result += (int32_t)row[j] * (int32_t)a->q_x[j];
            }

            int32_t true_dot = result - bias_128;
            a->out[i] = (float)true_dot * a->act_scale * a->scales[i];
        }
        return;
    }
#endif

#if TN_HAS_AVX512
    /* Float FMA fallback: 16 elements per iteration.
     * Used when activations aren't pre-quantized or VNNI isn't available. */
    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * a->n;

        __m512 acc512 = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < a->n; j += 16) {
            /* Load 16 uint8, convert to signed by subtracting 128 bias, then to float */
            __m128i u8_16  = _mm_loadu_si128((const __m128i *)&row[j]);
            __m512i u32_16 = _mm512_cvtepu8_epi32(u8_16);
            __m512i s32_16 = _mm512_sub_epi32(u32_16, _mm512_set1_epi32(128));
            __m512  wv     = _mm512_cvtepi32_ps(s32_16);
            __m512  xv     = _mm512_loadu_ps(&a->x[j]);
            acc512 = _mm512_fmadd_ps(wv, xv, acc512);
        }
        float val = _mm512_reduce_add_ps(acc512);
        for (; j < a->n; j++) {
            val += (float)((int)row[j] - 128) * a->x[j];
        }
        a->out[i] = val * a->scales[i];
    }
#elif TN_HAS_AVX2
    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * a->n;

        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < a->n; j += 8) {
            __m128i u8_8  = _mm_loadl_epi64((const __m128i *)&row[j]);
            __m256i u32_8 = _mm256_cvtepu8_epi32(u8_8);
            __m256i s32_8 = _mm256_sub_epi32(u32_8, _mm256_set1_epi32(128));
            __m256  wv    = _mm256_cvtepi32_ps(s32_8);
            __m256  xv    = _mm256_loadu_ps(&a->x[j]);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        __m128 hi   = _mm256_extractf128_ps(acc, 1);
        __m128 lo   = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        float  val  = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2)));
        for (; j < a->n; j++) {
            val += (float)((int)row[j] - 128) * a->x[j];
        }
        a->out[i] = val * a->scales[i];
    }
#else
    /* Scalar fallback */
    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * a->n;
        float acc = 0.0f;
        for (int j = 0; j < a->n; j++) {
            acc += (float)((int)row[j] - 128) * a->x[j];
        }
        a->out[i] = acc * a->scales[i];
    }
#endif
}

/* ── INT4 classifier matmul ─────────────────────────────────────────────────
 *
 * INT4 packed weights: 2 values per byte (low nibble = w[2k], high = w[2k+1]).
 * Unsigned storage with +8 bias: signed value + 8 → [1, 15].
 * At runtime, unpack nibbles to uint8, use VNNI dpbusds, then correct:
 *   dpbusds(w_u4_as_u8, q_x) = dot(w_signed, q_x) + 8 * sum_qx
 *   true_dot = result - 8 * sum_qx
 *
 * Processes 64 weights from 32 packed bytes per VNNI iteration.
 * Half the memory traffic of INT8 for the same compute throughput.
 */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;        /* packed INT4 weights */
    const float *scales;
    int n, d;
#if TN_HAS_AVX512VNNI
    const int8_t *q_x;
    float         act_scale;
    int32_t       sum_qx;
#endif
} ParallelMatmulI4Args;

static void matmul_i4_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    ParallelMatmulI4Args *a = (ParallelMatmulI4Args *)arg;
    size_t row_bytes = ((size_t)a->n + 1) / 2;

#if TN_HAS_AVX512VNNI
    /*
     * VNNI dpbusds path for INT4: unpack nibbles → uint8, then dpbusds.
     *
     * For 64 weights we load 32 packed bytes:
     *   byte[k] = (w[2k+1] << 4) | w[2k]
     * Unpack:
     *   low nibbles  = byte & 0x0F → w[0], w[2], w[4], ...  (32 values)
     *   high nibbles = byte >> 4   → w[1], w[3], w[5], ...  (32 values)
     * Interleave to restore sequential order, then dpbusds.
     *
     * Bias correction: values are unsigned +8, so
     *   true_dot = dpbusds_result - 8 * sum_qx
     */
    if (a->q_x && a->act_scale > 0.0f) {
        int32_t bias_8 = 8 * a->sum_qx;

#if TN_HAS_AVX512VBMI
        /*
         * VBMI fast path: 3-instruction INT4 unpack (vs ~15 with SSE interleave).
         * vpermb replicates each byte 2×, multishift extracts nibbles, mask 0x0F.
         *
         * Gated by tn_cpu_features_detect()->avx512vbmi, NOT just this
         * compile-time macro: a hypervisor can advertise AVX-512VBMI via
         * CPUID (matching what -march=native detects at compile time too)
         * while real execution faults with SIGILL — confirmed on a real
         * Firecracker microVM host (see docs/ai/mistakes.md and
         * bitunpack2_vnni.h, which hit the identical issue). Both unpack
         * paths are always compiled in when this file has AVX-512VBMI
         * available; the choice defers to the execution-verified flag.
         */
        bool use_vbmi_i4 = tn_cpu_features_detect()->avx512vbmi;
#endif

        for (int i = start; i < end; i++) {
            const tn_u8 *row = a->w + (size_t)i * row_bytes;

            __m512i acc = _mm512_setzero_si512();
            int j = 0;

#if TN_HAS_AVX512VBMI
            if (use_vbmi_i4) {
                /* VBMI INT4 unpack: load 32 bytes, replicate each 2×, multishift nibbles */
                static const __m512i perm_i4 = {
                    (long long)0x0303020201010000LL, (long long)0x0707060605050404LL,
                    (long long)0x0b0b0a0a09090808LL, (long long)0x0f0f0e0e0d0d0c0cLL,
                    (long long)0x1313121211111010LL, (long long)0x1717161615151414LL,
                    (long long)0x1b1b1a1a19191818LL, (long long)0x1f1f1e1e1d1d1c1cLL
                };
                /* Shift per qword: {0,4,16,20,32,36,48,52} extracts nibbles
                 * from pairs of replicated bytes at positions 0,8,16,24,32,40,48,56 */
                static const __m512i shift_i4 = {
                    (long long)0x3430242014100400LL, (long long)0x3430242014100400LL,
                    (long long)0x3430242014100400LL, (long long)0x3430242014100400LL,
                    (long long)0x3430242014100400LL, (long long)0x3430242014100400LL,
                    (long long)0x3430242014100400LL, (long long)0x3430242014100400LL
                };
                const __m512i mask_nibble = _mm512_set1_epi8(0x0F);

                for (; j + 63 < a->n; j += 64) {
                    __m256i packed = _mm256_loadu_si256((const __m256i *)(row + j/2));
                    __m512i expanded = _mm512_permutexvar_epi8(perm_i4,
                                            _mm512_castsi256_si512(packed));
                    __m512i shifted = _mm512_multishift_epi64_epi8(shift_i4, expanded);
                    __m512i wenc = _mm512_and_si512(shifted, mask_nibble);
                    __m512i qxv = _mm512_loadu_si512((const __m512i *)&a->q_x[j]);
                    acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
                }
            } else
#endif
            {
                const __m256i mask_lo = _mm256_set1_epi8(0x0F);
                for (; j + 63 < a->n; j += 64) {
                    __m256i packed = _mm256_loadu_si256((const __m256i *)(row + j/2));
                    __m256i lo = _mm256_and_si256(packed, mask_lo);
                    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_lo);
                    __m128i lo_lo = _mm256_castsi256_si128(lo);
                    __m128i lo_hi = _mm256_extracti128_si256(lo, 1);
                    __m128i hi_lo = _mm256_castsi256_si128(hi);
                    __m128i hi_hi = _mm256_extracti128_si256(hi, 1);
                    __m128i a0 = _mm_unpacklo_epi8(lo_lo, hi_lo);
                    __m128i b0 = _mm_unpackhi_epi8(lo_lo, hi_lo);
                    __m128i c0 = _mm_unpacklo_epi8(lo_hi, hi_hi);
                    __m128i d0 = _mm_unpackhi_epi8(lo_hi, hi_hi);
                    __m512i wenc = _mm512_castsi128_si512(a0);
                    wenc = _mm512_inserti32x4(wenc, b0, 1);
                    wenc = _mm512_inserti32x4(wenc, c0, 2);
                    wenc = _mm512_inserti32x4(wenc, d0, 3);
                    __m512i qxv = _mm512_loadu_si512((const __m512i *)&a->q_x[j]);
                    acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
                }
            }

            int32_t result = _mm512_reduce_add_epi32(acc);

            /* Scalar tail */
            for (; j < a->n; j++) {
                int nibble;
                if (j & 1)
                    nibble = (row[j/2] >> 4) & 0x0F;
                else
                    nibble = row[j/2] & 0x0F;
                result += nibble * (int32_t)a->q_x[j];
            }

            int32_t true_dot = result - bias_8;
            a->out[i] = (float)true_dot * a->act_scale * a->scales[i];
        }
        return;
    }
#endif

#if TN_HAS_AVX512
    /* Float FMA fallback for INT4 */
    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * row_bytes;
        __m512 acc512 = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < a->n; j += 16) {
            /* Unpack 16 INT4 values from 8 bytes */
            __m128i packed8 = _mm_loadl_epi64((const __m128i *)(row + j/2));
            __m128i mask4 = _mm_set1_epi8(0x0F);
            __m128i lo_8 = _mm_and_si128(packed8, mask4);
            __m128i hi_8 = _mm_and_si128(_mm_srli_epi16(packed8, 4), mask4);
            __m128i interleaved = _mm_unpacklo_epi8(lo_8, hi_8); /* 16 uint4 as uint8 */
            /* Convert to signed: subtract 8 bias */
            __m512i u32_16 = _mm512_cvtepu8_epi32(interleaved);
            __m512i s32_16 = _mm512_sub_epi32(u32_16, _mm512_set1_epi32(8));
            __m512  wv     = _mm512_cvtepi32_ps(s32_16);
            __m512  xv     = _mm512_loadu_ps(&a->x[j]);
            acc512 = _mm512_fmadd_ps(wv, xv, acc512);
        }
        float val = _mm512_reduce_add_ps(acc512);
        for (; j < a->n; j++) {
            int nibble;
            if (j & 1) nibble = (row[j/2] >> 4) & 0x0F;
            else nibble = row[j/2] & 0x0F;
            val += (float)(nibble - 8) * a->x[j];
        }
        a->out[i] = val * a->scales[i];
    }
#else
    /* Scalar fallback */
    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * row_bytes;
        float acc = 0.0f;
        for (int j = 0; j < a->n; j++) {
            int nibble;
            if (j & 1) nibble = (row[j/2] >> 4) & 0x0F;
            else nibble = row[j/2] & 0x0F;
            acc += (float)(nibble - 8) * a->x[j];
        }
        a->out[i] = acc * a->scales[i];
    }
#endif
}

void parallel_matmul_i4(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp) {
#if TN_HAS_AVX512VNNI
    int8_t q_x_buf[16384];
    int8_t *q_x = q_x_buf;
    float act_scale = 0.0f;
    int32_t sum_qx = 0;

    if (n <= 16384) {
        act_scale = quantize_row_to_i8_avx512(x, q_x, n);
        if (act_scale > 0.0f) {
            sum_qx = sum_i8_avx512(q_x, n);
        }
    }
#endif

    ParallelMatmulI4Args args = {
        .out = out, .x = x, .w = w, .scales = scales, .n = n, .d = d,
#if TN_HAS_AVX512VNNI
        .q_x = (act_scale > 0.0f) ? q_x : NULL,
        .act_scale = act_scale,
        .sum_qx = sum_qx
#endif
    };

    if (!tp) {
        matmul_i4_task(&args, 0, 0, d);
        return;
    }

    threadpool_dispatch(tp, matmul_i4_task, &args, d);
}

void parallel_matmul_i8(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp) {
#if TN_HAS_AVX512VNNI
    /*
     * Quantize activations once (shared across all worker threads).
     * Uses the same quantize_row_to_i8_avx512 as the ternary VNNI path.
     */
    int8_t q_x_buf[16384]; /* max dim supported */
    int8_t *q_x = q_x_buf;
    float act_scale = 0.0f;
    int32_t sum_qx = 0;

    if (n <= 16384) {
        act_scale = quantize_row_to_i8_avx512(x, q_x, n);
        if (act_scale > 0.0f) {
            sum_qx = sum_i8_avx512(q_x, n);
        }
    }
#endif

    ParallelMatmulI8Args args = {
        .out = out, .x = x, .w = w, .scales = scales, .n = n, .d = d,
#if TN_HAS_AVX512VNNI
        .q_x = (act_scale > 0.0f) ? q_x : NULL,
        .act_scale = act_scale,
        .sum_qx = sum_qx
#endif
    };

    if (!tp) {
        matmul_i8_task(&args, 0, 0, d);
        return;
    }

    threadpool_dispatch(tp, matmul_i8_task, &args, d);
}
