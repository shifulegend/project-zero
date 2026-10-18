#ifndef TN_MATMUL_Q6K_H
#define TN_MATMUL_Q6K_H

/*
 * Fused Q6_K × F32 activation matmul (gemv), zero-copy from the GGUF mmap.
 *
 * Q4_K_M files store attn_v / ffn_down (and often output.weight) as Q6_K
 * (210 bytes per 256 weights). The loader used to dequantise those to F32
 * (4 bytes per weight, ~4.9x the bytes); this kernel decodes 6-bit codes on
 * the fly against Q8K-quantised activations, like matmul_q4k.c.
 *
 *   n: inner dimension, multiple of 256;  d: output rows.
 */

#include "math/matmul_q4k.h"   /* TnQ8KActBlock, tn_quantize_q8k */

#define TN_Q6K_BYTES 210

/* One row (n_blocks super-blocks) · pre-quantised activations. Exposed for tests. */
float tn_dot_q6k_row_q8k(const uint8_t *row_q6k, const TnQ8KActBlock *acts, int n_blocks);
/* Portable reference of the same dot (no SIMD). */
float tn_dot_q6k_row_q8k_scalar(const uint8_t *row_q6k, const TnQ8KActBlock *acts, int n_blocks);

void parallel_matmul_q6k(float *out, const float *x, const uint8_t *w_q6k,
                         int n, int d, ThreadPool *tp);

/* n_tokens inputs (x: [n_tokens][n], out: [n_tokens][d]); each row read once.
 * Bit-identical to n_tokens parallel_matmul_q6k() calls. */
void parallel_matmul_q6k_tokens(float *out, const float *x, const uint8_t *w_q6k,
                                int n, int d, int n_tokens, ThreadPool *tp);

#endif /* TN_MATMUL_Q6K_H */
