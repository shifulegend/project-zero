#ifndef TN_SPEC_SCRATCH_H
#define TN_SPEC_SCRATCH_H

#include "core/config.h"
#include "core/error.h"

/**
 * Phase 18 (speculative decoding): N-token-wide scratch buffers for
 * transformer_forward_batch(). Owned entirely outside RunState so every
 * existing single-token caller (vision_bridge.c, agent/output_inject.c,
 * every pre-existing test, plain generate()) is completely unaffected —
 * RunState's struct layout and memory footprint do not change. Allocated
 * once, sized by --spec-length, only when --draft-model is given.
 *
 * Layout: each buffer is row-major [n_tokens][width] (token k's row starts
 * at buf + k*width) — matches include/math/batched_matmul.h's convention,
 * so these buffers can be passed directly as batched-matmul inputs/outputs.
 *
 * k/v projection scratch reuses xb2/hb2 exactly like the single-token path
 * does (RunState.xb2 doubles as k_buf, RunState.hb2 as v_buf in
 * attention_forward() — kv_dim <= dim <= width(xb2), kv_dim <= hidden_dim
 * <= width(hb2)); the same reuse trick applies here, just N-wide.
 */
typedef struct {
    int n_tokens;
    int dim;
    int hidden_dim;
    int vocab_size;

    float *x;      /* [n_tokens][dim]        post-embedding / residual stream */
    float *xb;     /* [n_tokens][dim]        rmsnorm / attention-out scratch */
    float *xb2;     /* [n_tokens][dim]        residual branch scratch; doubles as k_buf */
    float *hb;      /* [n_tokens][hidden_dim] FFN gate; doubles as MLA kv_full/q_full */
    float *hb2;      /* [n_tokens][hidden_dim] FFN up; doubles as v_buf / MLA kv-compress */
    float *q;       /* [n_tokens][dim]        query projections (all heads) */
    float *logits;   /* [n_tokens][vocab_size] one row per candidate position */
} SpecBatchScratch;

/**
 * Allocate all buffers for a batch width of `n_tokens`. Returns TN_ERR_OOM
 * on allocation failure (all-or-nothing: any partial allocation is freed
 * before returning).
 */
TernaryError spec_batch_scratch_alloc(SpecBatchScratch *sb, const Config *cfg, int n_tokens);

/**
 * Free all buffers. Safe to call on a zero-initialized or already-freed
 * SpecBatchScratch (idempotent).
 */
void spec_batch_scratch_free(SpecBatchScratch *sb);

#endif /* TN_SPEC_SCRATCH_H */
