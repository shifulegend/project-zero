#pragma once
/*
 * parallel_matmul_q2_0 — fused Q2_0 × F32 activation matmul (gemv).
 *
 * Q2_0 block layout — PrismML's GROUP-128 packing (34 bytes / 128 elements,
 * 2.125 bits/weight), NOT mainline ggml's canonical block_q2_0 (18 B / 64
 * elements); both share GGUF type ID 42, and the group size was confirmed
 * empirically from the real Ternary-Bonsai-27B file (see matmul_q2_0.c's
 * header comment and docs/ai/decision-log.md 2026-07-16). [This comment
 * previously described the 18B/64 ggml layout — stale, fixed 2026-07-17.]
 *   [d:  fp16  2B] — scale
 *   [qs: u8[32]  ] — 128 × 2-bit codes, 4 per byte
 *
 * Element j: code = (qs[j/4] >> ((j%4)*2)) & 0x3   (code in {0,1,2,3})
 * Value = (code - 1) * d      (00=-1, 01=0, 10=+1, 11=+2, all scaled by d)
 *
 * Used by PrismML's Qwen3.6-based "Bonsai" ternary GGUF releases, where
 * embeddings/attention/MLP/LM-head projections are all stored as Q2_0.
 */

#include "threading/thread_pool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void parallel_matmul_q2_0(float *out, const float *x, const uint8_t *w_q2_0,
                           int n, int d, ThreadPool *tp);

/*
 * parallel_matmul_q2_0_batch — batched Q2_0 × F32 GEMV for k weight matrices.
 */
void parallel_matmul_q2_0_batch(float **outs, float **xs,
                                 const uint8_t * const *ws,
                                 int n, int d, int k, ThreadPool *tp);

/*
 * AVX-512 VNNI fast-path entries (only compiled/linked when the build has
 * TN_HAS_AVX512VNNI — matmul_q2_0_vnni.c). Return 1 if handled, 0 if the
 * caller should fall back to the portable kernel. parallel_matmul_q2_0()
 * dispatches to the first automatically; the _ref variant is the frozen
 * pre-2026-07-17 row-dot kept exclusively for tools/bench_q2_0's in-binary
 * A/B baseline and correctness cross-checks — never call it from engine code.
 */
int parallel_matmul_q2_0_vnni(float *out, const float *x, const uint8_t *w_q2_0,
                               int n, int d, ThreadPool *tp);
int parallel_matmul_q2_0_vnni_ref(float *out, const float *x, const uint8_t *w_q2_0,
                                   int n, int d, ThreadPool *tp);

#ifdef __cplusplus
}
#endif
