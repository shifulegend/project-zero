#ifndef TN_DENSE_MATMUL_DISPATCH_H
#define TN_DENSE_MATMUL_DISPATCH_H

/* Shared per-projection matmul dispatch for the generic dense forward path
 * (attention.c / ffn.c). Each projection (wq/wk/wv/wo/w1/w2/w3) carries its
 * own WEIGHT_TYPE_* (see weights.h) since mixed-precision GGUF schemes such
 * as Q4_K_M assign different quant types to different tensor roles within
 * the same model. Centralized here so the dispatch logic exists in exactly
 * one place, not duplicated (and liable to drift) across both call sites. */

#include "core/weights.h"
#include "math/parallel_matmul.h"
#include "math/matmul_f16.h"
#include "math/matmul_q2_0.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q4k_x8.h"
#include "threading/thread_pool.h"

/* Bound for the stack Q8K activation buffer below — matches
 * ATTN_PREQ_BUF_SIZE/FFN_PREQ_BUF_SIZE's 16384-element cap (attention.c,
 * ffn.c) expressed in 256-element Q8K blocks. */
#define TN_DENSE_DISPATCH_MAX_Q8K_BLOCKS 64

static inline void tn_dense_matmul_dispatch(float *out, const float *x,
                                             const tn_i8 *w, int wtype,
                                             int n, int d, ThreadPool *tp) {
    switch (wtype) {
    case WEIGHT_TYPE_F16:
        parallel_matmul_f16(out, x, (const tn_u16 *)w, n, d, tp);
        break;
    case WEIGHT_TYPE_Q4K:
        parallel_matmul_q4k(out, x, (const uint8_t *)w, n, d, tp);
        break;
    case WEIGHT_TYPE_Q4K_X8: {
        int n_blocks = n / 256;
        TnQ8KActBlock acts[TN_DENSE_DISPATCH_MAX_Q8K_BLOCKS];
        tn_quantize_q8k(acts, x, n_blocks);
        parallel_matmul_q4k_x8_preq(out, acts, (const TnQ4KX8Block *)w, n_blocks, d, tp);
        break;
    }
    case WEIGHT_TYPE_Q2_0:
        parallel_matmul_q2_0(out, x, (const uint8_t *)w, n, d, tp);
        break;
    default: /* WEIGHT_TYPE_F32 */
        parallel_matmul_float32(out, x, (const float *)w, n, d, tp);
        break;
    }
}

/* True for weight types that consume pre-quantized Q8K activation blocks
 * (Q4K and its x8-repacked variant) — used by attention.c/ffn.c to decide
 * whether Q/K/V (or gate/up) share a single quantize-once Q8K buffer
 * instead of each independently re-quantizing the same input. */
static inline int tn_is_q4k_family(int wtype) {
    return wtype == WEIGHT_TYPE_Q4K || wtype == WEIGHT_TYPE_Q4K_X8;
}

/* Like tn_dense_matmul_dispatch but takes pre-quantized Q8K activation
 * blocks — caller must have already checked tn_is_q4k_family(wtype). */
static inline void tn_dense_matmul_dispatch_preq(float *out, const TnQ8KActBlock *acts,
                                                  const tn_i8 *w, int wtype,
                                                  int n, int d, ThreadPool *tp) {
    if (wtype == WEIGHT_TYPE_Q4K_X8) {
        parallel_matmul_q4k_x8_preq(out, acts, (const TnQ4KX8Block *)w, n / 256, d, tp);
    } else {
        parallel_matmul_q4k_preq(out, acts, (const uint8_t *)w, n, d, tp);
    }
}

#endif /* TN_DENSE_MATMUL_DISPATCH_H */
