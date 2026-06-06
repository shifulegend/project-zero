#ifndef TN_PARALLEL_MATMUL_H
#define TN_PARALLEL_MATMUL_H

#include "core/platform.h"
#include "threading/thread_pool.h"

/**
 * Parallel ternary matrix-vector multiplication.
 *
 * Distributes output rows across the thread pool, with each worker
 * calling the SIMD-dispatched ternary_matmul kernel on its slice.
 *
 * Falls back to single-threaded dispatch if tp is NULL.
 *
 * @param out   Output vector of size d (must be pre-allocated)
 * @param x     Input vector of size n
 * @param w     Weight matrix in row-major order, size d * n, values in {-1, 0, 1}
 * @param n     Input dimension (columns)
 * @param d     Output dimension (rows)
 * @param scale Per-matrix scale factor
 * @param tp    Thread pool (NULL for single-threaded fallback)
 */
void parallel_ternary_matmul(float *out, const float *x, const tn_i8 *w,
                              int n, int d, float scale, ThreadPool *tp);

/**
 * Parallel packed ternary matrix-vector multiplication.
 * Uses 2-bit packing (4 weights per byte).
 */
void parallel_ternary_matmul_packed(float *out, const float *x, const tn_u8 *w,
                                     int n, int d, float scale, ThreadPool *tp);

/**
 * Layer-level pre-quantised activation state.
 *
 * Quantise an input vector ONCE before multiple matmul calls that share it
 * (e.g. Q+K+V projections all read s->xb; gate+up projections both read s->xb).
 * Pass the same TnPreqActivation to each call to skip redundant quantisation.
 *
 * Usage:
 *   int8_t buf[MAX_DIM];           // stack buffer, caller owns
 *   TnPreqActivation preq;
 *   tn_preq_prepare(&preq, buf, s->xb, dim);
 *   parallel_ternary_matmul_packed_preq(out_q, wq, dim, dim, sq, &preq, tp);
 *   parallel_ternary_matmul_packed_preq(out_k, wk, dim, kv_dim, sk, &preq, tp);
 *   parallel_ternary_matmul_packed_preq(out_v, wv, dim, kv_dim, sv, &preq, tp);
 *
 * On non-AVX-512VNNI platforms valid=0; _preq functions fall back to
 * the standard quantise-per-call path transparently.
 */
typedef struct {
    const int8_t *q_x;       /* points into caller's buffer (not owned) */
    float         act_scale;
    int32_t       sum_qx;
    int           valid;     /* 1 if successfully quantised, 0 = fallback */
} TnPreqActivation;

/**
 * Quantise x[n] into buf[n] (caller-allocated int8 array, must hold n bytes).
 * Fills preq. Returns 1 on success, 0 if vector is zero or platform lacks VNNI.
 * buf must remain live until all _preq matmul calls using it are complete.
 */
int tn_preq_prepare(TnPreqActivation *preq, int8_t *buf, const float *x, int n);

/**
 * Like parallel_ternary_matmul_packed() but accepts a pre-quantised activation.
 * If preq->valid, skips quantisation step entirely (layer-level preq).
 * Falls back to standard quantise-per-call if preq is NULL or preq->valid == 0.
 */
void parallel_ternary_matmul_packed_preq(float *out, const float *x, const tn_u8 *w,
                                          int n, int d, float scale,
                                          const TnPreqActivation *preq,
                                          ThreadPool *tp);
/**
 * Parallel bf16 matrix-vector multiplication.
 * Used for the output classifier when the embedding table is stored as bfloat16.
 * Converts bf16 → float32 on the fly (single bit-shift per value, no overhead).
 */
void parallel_matmul_bf16(float *out, const float *x, const tn_u16 *w,
                           int n, int d, ThreadPool *tp);

/**
 * Parallel float32 matrix-vector multiplication.
 * Used when weights are stored as regular float32 (non-ternary fallback path).
 */
void parallel_matmul_float32(float *out, const float *x, const float *w,
                              int n, int d, ThreadPool *tp);

/**
 * Parallel INT8 matrix-vector multiplication with per-row scales.
 * Used for the INT8-quantized output classifier (LM head).
 * Reads 1 byte per weight (vs 2 for BF16), halving bandwidth.
 *
 * out[i] = (sum_j w_i8[i][j] * x[j]) * scales[i]
 */
void parallel_matmul_i8(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp);

/**
 * Parallel INT4 matrix-vector multiplication with per-row scales.
 * Used for the INT4-quantized output classifier (LM head).
 * Reads 0.5 bytes per weight (vs 1 for INT8, 2 for BF16).
 * Weights packed as 2 per byte: low nibble = w[2k], high nibble = w[2k+1].
 * Unsigned storage with +8 bias; runtime correction: true_dot - 8*sum_qx.
 *
 * out[i] = (sum_j w_i4[i][j] * x[j]) * scales[i]
 */
void parallel_matmul_i4(float *out, const float *x, const tn_u8 *w,
                          const float *scales, int n, int d, ThreadPool *tp);

#endif /* TN_PARALLEL_MATMUL_H */
