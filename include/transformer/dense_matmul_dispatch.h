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
#include "threading/thread_pool.h"

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
    case WEIGHT_TYPE_Q2_0:
        parallel_matmul_q2_0(out, x, (const uint8_t *)w, n, d, tp);
        break;
    default: /* WEIGHT_TYPE_F32 */
        parallel_matmul_float32(out, x, (const float *)w, n, d, tp);
        break;
    }
}

#endif /* TN_DENSE_MATMUL_DISPATCH_H */
