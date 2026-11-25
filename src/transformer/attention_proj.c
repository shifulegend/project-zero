/*
 * transformer/attention_proj — the Q, K and V projections of one token's attention layer
 * (split out of attention.c).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "core/platform.h"
#include "core/weights.h"
#include "math/batched_matmul.h"
#include "math/matmul_q4k.h"
#include "math/parallel_matmul.h"
#include "runtime/spec_scratch.h"
#include "transformer/attention.h"
#include "transformer/dense_matmul_dispatch.h"

/* Maximum input dimension supported for layer-level preq stack buffer.
 * BitNet-b1.58-2B-4T: dim=2048. Guard against future larger models. */
#define ATTN_PREQ_BUF_SIZE 16384

void attention_project_qkv(RunState *s, const WeightsAttn *wa, bool ternary, int layer, int dim,
                           int q_width, int kv_width, float *q_buf, float *k_buf, float *v_buf,
                           ThreadPool *tp) {
    if (ternary) {
        int8_t preq_buf[ATTN_PREQ_BUF_SIZE];
        TnPreqActivation preq;
        tn_preq_prepare(&preq, preq_buf, s->xb, dim);
        parallel_ternary_matmul_packed_preq(q_buf, s->xb, (const tn_u8 *)wa->wq[layer], dim,
                                            q_width, wa->sq[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(k_buf, s->xb, (const tn_u8 *)wa->wk[layer], dim,
                                            kv_width, wa->sk[layer], &preq, tp);
        parallel_ternary_matmul_packed_preq(v_buf, s->xb, (const tn_u8 *)wa->wv[layer], dim,
                                            kv_width, wa->sv[layer], &preq, tp);
    } else if (tn_is_q4k_family(wa->wq_type[layer]) && tn_is_q4k_family(wa->wk_type[layer]) &&
               tn_is_q4k_family(wa->wv_type[layer])) {
        /* All three read the same s->xb — quantize to Q8K once and reuse, same
     * layer-level-preq idea as the ternary branch above. Bug fixed 2026-07-31:
     * this branch previously called tn_dense_matmul_dispatch() three times,
     * each of which (via parallel_matmul_q4k) independently re-quantized
     * s->xb, despite the comment above already claiming the shared-quantize
     * optimization — it had only ever been wired up for the ternary path.
     * Extended 2026-07-31 to also cover WEIGHT_TYPE_Q4K_X8 (the repacked
     * multi-row-GEMV variant, matmul_q4k_x8.c) — both consume the same Q8K
     * activation format, so the shared-quantize win applies to either. */
        TnQ8KActBlock acts[ATTN_PREQ_BUF_SIZE / TN_Q8K_BLOCK];
        int n_blocks = dim / TN_Q8K_BLOCK;
        tn_quantize_q8k(acts, s->xb, n_blocks);
        tn_dense_matmul_dispatch_preq(q_buf, acts, wa->wq[layer], wa->wq_type[layer], dim, q_width,
                                      tp);
        tn_dense_matmul_dispatch_preq(k_buf, acts, wa->wk[layer], wa->wk_type[layer], dim, kv_width,
                                      tp);
        tn_dense_matmul_dispatch_preq(v_buf, acts, wa->wv[layer], wa->wv_type[layer], dim, kv_width,
                                      tp);
    } else {
        tn_dense_matmul_dispatch(q_buf, s->xb, wa->wq[layer], wa->wq_type[layer], dim, q_width, tp);
        tn_dense_matmul_dispatch(k_buf, s->xb, wa->wk[layer], wa->wk_type[layer], dim, kv_width,
                                 tp);
        tn_dense_matmul_dispatch(v_buf, s->xb, wa->wv[layer], wa->wv_type[layer], dim, kv_width,
                                 tp);
    }
}

void attention_project_qkv_rows(SpecBatchScratch *sb, const TransformerWeights *w,
                                const Config *cfg, int layer, int n_tokens, float *k_buf_all,
                                float *v_buf_all, ThreadPool *tp) {
    const WeightsAttn *wa = &w->attn;
    int dim               = cfg->dim;
    int kv_dim            = config_kv_dim(cfg);
    if (w->layers_are_ternary) {
        tn_ternary_matmul_packed_batch(sb->q, sb->xb, (const tn_u8 *)wa->wq[layer], dim, dim,
                                       wa->sq[layer], n_tokens, tp);
        tn_ternary_matmul_packed_batch(k_buf_all, sb->xb, (const tn_u8 *)wa->wk[layer], dim, kv_dim,
                                       wa->sk[layer], n_tokens, tp);
        tn_ternary_matmul_packed_batch(v_buf_all, sb->xb, (const tn_u8 *)wa->wv[layer], dim, kv_dim,
                                       wa->sv[layer], n_tokens, tp);
    } else {
        tn_dense_matmul_dispatch_batch(sb->q, sb->xb, wa->wq[layer], wa->wq_type[layer], dim, dim,
                                       n_tokens, tp);
        tn_dense_matmul_dispatch_batch(k_buf_all, sb->xb, wa->wk[layer], wa->wk_type[layer], dim,
                                       kv_dim, n_tokens, tp);
        tn_dense_matmul_dispatch_batch(v_buf_all, sb->xb, wa->wv[layer], wa->wv_type[layer], dim,
                                       kv_dim, n_tokens, tp);
    }
}
