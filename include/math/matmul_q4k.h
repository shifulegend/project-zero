#ifndef TN_MATMUL_Q4K_H
#define TN_MATMUL_Q4K_H

/*
 * Fused Q4_K × F32 activation matmul (gemv).
 *
 * For each output row, Q4_K super-blocks are decoded on-the-fly during the
 * dot product — no intermediate F32 buffer is ever written.  Bandwidth is
 * reduced from ~68.5 bits/weight (dequant+float32 path) to ~4.5 bits/weight
 * (read Q4_K data only), a ~15× reduction.
 *
 * Used for DeepSeek-V2 routed expert weights (w1, w2, w3) which are stored
 * as raw Q4_K bytes (mmap pointer) in moe_w{1,2,3}[layer][expert].
 *
 * parallel_matmul_q4k(out, x, w_q4k, n, d, tp):
 *   out:    float[d]        — output vector
 *   x:      float[n]        — activation input (F32)
 *   w_q4k:  uint8[d * ...]  — Q4_K weight matrix, d rows × n columns
 *   n:      inner dimension (must be multiple of 256 = Q4_K super-block size)
 *   d:      output dimension (number of rows)
 *   tp:     thread pool (NULL = single-threaded)
 *
 * Quant type guard: only call with GGUF_TYPE_Q4_K (= 12) weights.
 */

#include "core/platform.h"
#include "threading/thread_pool.h"
#include <stdint.h>

/* ── Q8K activation block — 292 bytes per 256-element block ─────────────── */
#define TN_Q8K_BLOCK 256
typedef struct {
    float   d;           /* scale = abs_max / 127 */
    int8_t  qs[256];     /* quantized int8 activations */
    int16_t bsums[16];   /* bsums[j] = sum(qs[j*16 .. j*16+15]) */
} TnQ8KActBlock;

/*
 * tn_quantize_q8k — quantize n_blocks×256 floats into Q8K activation blocks.
 * Call once per unique input vector; pass the result to the _preq variants
 * to avoid redundant re-quantisation when the same vector drives multiple
 * Q4K matmuls (e.g., shared-expert w1 and w3, both reading s->xb).
 */
void tn_quantize_q8k(TnQ8KActBlock *out, const float *x, int n_blocks);
TnQ8KActBlock *q8k_buf_ensure(int n_blocks);

void parallel_matmul_q4k(float *out, const float *x, const uint8_t *w_q4k,
                          int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q4k_preq — same as parallel_matmul_q4k but accepts
 * pre-quantized Q8K activation blocks instead of a raw float* input.
 * Use when the same activation vector drives more than one Q4K matmul.
 */
void parallel_matmul_q4k_preq(float *out, const TnQ8KActBlock *acts,
                               const uint8_t *w_q4k,
                               int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q4k_pf — same as parallel_matmul_q4k but accepts
 * an optional w_next pointer (next expert's Q4_K weight matrix) to issue
 * interleaved software prefetch hints for corresponding row offsets.
 */
void parallel_matmul_q4k_pf(float *out, const float *x, const uint8_t *w_q4k,
                           const uint8_t *w_next, int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q4k_preq_pf — same as parallel_matmul_q4k_preq but accepts
 * an optional w_next pointer (next expert's Q4_K weight matrix) to issue
 * interleaved software prefetch hints for corresponding row offsets.
 */
void parallel_matmul_q4k_preq_pf(float *out, const TnQ8KActBlock *acts,
                                  const uint8_t *w_q4k, const uint8_t *w_next,
                                  int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q4k_batch — batched Q4_K × F32 GEMV for k weight matrices
 * sharing the same input vector x.
 *
 * Processes all k expert gate/up projections in a single thread-pool dispatch,
 * reducing dispatch overhead from k separate calls to 1.  Thread distribution:
 * total_rows = k × d rows split across T threads.
 *
 *   outs:  float*[k]          — k output arrays, each d floats
 *   x:     float[n]           — shared activation input (same for all k experts)
 *   ws:    uint8_t*[k]        — k Q4_K weight arrays, each d rows × n cols
 *   n:     inner dimension (multiple of 256)
 *   d:     output rows per expert
 *   k:     number of experts (batch count)
 *   tp:    thread pool (NULL = single-threaded)
 */
void parallel_matmul_q4k_batch(float * const *outs, const float *x,
                                const uint8_t * const *ws,
                                int n, int d, int k, ThreadPool *tp);

/*
 * parallel_matmul_q4k_batch_preq — same as parallel_matmul_q4k_batch but
 * accepts pre-quantized Q8K activation blocks.  Use when the same input
 * drives both gate and up projections (saves one quantize_to_q8k call).
 */
void parallel_matmul_q4k_batch_preq(float * const *outs,
                                     const TnQ8KActBlock *acts,
                                     const uint8_t * const *ws,
                                     int n, int d, int k, ThreadPool *tp);

/*
 * Fused Row-Split GEMV dispatches: 1 threadpool dispatch for all routed experts,
 * splitting total_rows = top_k * d across T threads without inter-expert barriers.
 */
void parallel_matmul_q4k_fused_rowsplit_w13(float *hb_out, float *hb2_out,
                                             const TnQ8KActBlock *acts,
                                             const uint8_t * const *w1_ptrs,
                                             const uint8_t * const *w3_ptrs,
                                             int n, int d, int k, ThreadPool *tp);

void parallel_matmul_q4k_fused_rowsplit_w2(float *q_out,
                                            const TnQ8KActBlock * const *acts_array,
                                            const uint8_t * const *w2_ptrs,
                                            int n, int d, int k, ThreadPool *tp);

#endif /* TN_MATMUL_Q4K_H */
