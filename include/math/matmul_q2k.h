#pragma once
/*
 * matmul_q2k.h — Fused Q2_K × float32 activation matrix-vector product.
 *
 * Key property: dequantizes Q2K blocks on-the-fly while accumulating the
 * dot product, completely eliminating the 11.5 MB intermediate float32
 * buffer that the dequant-then-matmul path requires.
 *
 * Bandwidth comparison (dim=2048, expert_hdim=1408 example):
 *   Old path: read 924 KB Q2K + write 11.5 MB f32 + read 11.5 MB f32 ≈ 24 MB
 *   New path: read 924 KB Q2K + read 8 KB input (L1-resident)           ≈ 924 KB
 *   Improvement: ~25× lower memory traffic per matvec.
 *
 * Q2_K super-block layout (84 bytes / 256 elements):
 *   [scales: 16 bytes]  — 16 sub-block scales, packed as (4-bit lo, 4-bit hi)
 *   [qs:     64 bytes]  — 256 × 2-bit values, 4 per byte
 *   [d:      fp16, 2B]  — super-block scale  (offset 80)
 *   [dmin:   fp16, 2B]  — super-block min    (offset 82)
 *
 * Each 256-element block is split into two 128-element halves. Each half
 * has 4 shift-groups (j=0..3, shift=0/2/4/6) of 32 elements each, with
 * two 16-element sub-blocks per group (each with its own scale pair).
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * parallel_matvec_q2k — compute out[0..out_dim) = W × inp[0..in_dim)
 *   where W is a Q2_K-quantized weight matrix of shape [out_dim × in_dim].
 *
 * Parameters:
 *   out     — output vector, float32, length out_dim
 *   inp     — input vector,  float32, length in_dim
 *   w       — raw Q2_K bytes: out_dim × (in_dim/256) × 84 bytes
 *   in_dim  — input (inner) dimension, must be a multiple of 256
 *   out_dim — output dimension
 *   tp      — thread pool (NULL → single-threaded)
 */
void parallel_matvec_q2k(float *out, const float *inp, const uint8_t *w,
                          int in_dim, int out_dim, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
