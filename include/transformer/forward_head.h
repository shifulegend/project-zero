#ifndef TN_FORWARD_HEAD_H
#define TN_FORWARD_HEAD_H

/* Classifier of the forward pass (forward.c): logits from the final hidden state. */

#include "core/batch_scratch_fwd.h"
#include "core/config.h"
#include "core/weights.h"
#include "runtime/run_state.h"
#include "threading/thread_pool.h"

/* s->x (already final-normed) -> s->logits: Q2_0, ternary or the hardware-selected generic head. */
void forward_head_classify(const Config *cfg, const TransformerWeights *w, RunState *s,
                           ThreadPool *tp);

/* Final RMSNorm + classifier over n_tokens rows of sb->x -> logits_out ([n_tokens][vocab]); shared
 * by the speculative batch and multi-sequence steps. */
void forward_head_rows(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                       int n_tokens, ThreadPool *tp, float *logits_out);

#endif /* TN_FORWARD_HEAD_H */
