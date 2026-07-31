#ifndef TN_MATMUL_Q4K_X8_H
#define TN_MATMUL_Q4K_X8_H

/*
 * Q4_K x8-interleaved repack + multi-row GEMV — a direct, byte-exact port of
 * llama.cpp's block_q4_Kx8 / ggml_gemv_q4_K_8x8_q8_K (ggml/src/ggml-cpu/
 * repack.cpp + arch/x86/repack.cpp), added 2026-07-31 after two lighter-touch
 * attempts (a "grouped8" memory-only repack, and a macro-duplicated
 * independent-accumulator kernel with no repack) each failed in isolation —
 * see docs/ai/mistakes.md for the full attempt history.
 *
 * Unlike either prior attempt, this combines BOTH of llama.cpp's techniques
 * at once: weight rows are physically repacked at load time into an
 * interleaved layout where 8 rows' nibbles for the same column live next to
 * each other, and the GEMV kernel uses that layout to compute 8 output rows
 * per SIMD lane (via _mm256_blend_epi32 + one shared broadcast activation
 * value), not just via 8 independent register chains — genuine SIMD-width
 * parallelism across output rows, matching llama.cpp's actual mechanism.
 *
 * Only used when a projection's output dimension is a multiple of 8 (true
 * for all of Qwen3-8B's dense projections); other shapes stay on the
 * existing single-row WEIGHT_TYPE_Q4K path.
 */

#include "core/platform.h"
#include "math/matmul_q4k.h"
#include "threading/thread_pool.h"
#include <stdint.h>

/* One repacked super-block covering 8 rows' worth of 256 columns each.
 * Byte-exact port of llama.cpp's block_q4_Kx8 (ggml-cpu/repack.h):
 *   d[8] + dmin[8] (fp16 scale/min per row) + scales[96] (repacked 6-bit
 *   scale/min pairs) + qs[1024] (interleaved 4-bit quants, 8 rows x 128B).
 * sizeof == 16 + 16 + 96 + 1024 == 1152 bytes (144 bytes/row average,
 * matching Q4_K's per-row-per-superblock size). */
typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[96];
    uint8_t  qs[1024];
} TnQ4KX8Block;

/*
 * tn_q4k_repack_x8 — repack 8 weight rows' Q4_K bytes into n_blocks
 * TnQ4KX8Block structures (one per 256-column super-block).
 *
 *   rows:    8 pointers, each to a row's raw Q4_K bytes (n_blocks * 144B)
 *   out:     n_blocks TnQ4KX8Block structures (caller-allocated)
 *   n_blocks: number of 256-column super-blocks per row (row_bytes / 144)
 *
 * Call once per group of 8 rows at model-load time — not per token.
 */
void tn_q4k_repack_x8(TnQ4KX8Block *out, const uint8_t *rows[8], int n_blocks);

/*
 * parallel_matmul_q4k_x8_preq — GEMV against x8-repacked Q4_K weights.
 *
 *   out:      float[d]              — output vector (d must be a multiple of 8)
 *   acts:     pre-quantized Q8K activation blocks (see matmul_q4k.h)
 *   repacked: d/8 groups of n_blocks TnQ4KX8Block each, contiguous
 *             (group g's blocks start at repacked + g * n_blocks)
 *   n_blocks: number of 256-column super-blocks (n / 256)
 *   d:        output dimension, must be a multiple of 8
 *   tp:       thread pool (NULL = single-threaded)
 */
void parallel_matmul_q4k_x8_preq(float *out, const TnQ8KActBlock *acts,
                                  const TnQ4KX8Block *repacked,
                                  int n_blocks, int d, ThreadPool *tp);

#endif /* TN_MATMUL_Q4K_X8_H */
