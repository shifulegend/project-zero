#ifndef TN_SPECULATIVE_SPEC_DECODE_H
#define TN_SPECULATIVE_SPEC_DECODE_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "speculative/draft_model.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"
#include "transformer/generate.h"

/**
 * Phase 18 (speculative decoding), Stage 5: draft/verify/accept-reject
 * generation loop. Signatures mirror generate()/generate_with_callback()
 * (transformer/generate.h) plus a DraftModel and spec_length.
 *
 * Per round: the draft model proposes spec_length candidate tokens one at
 * a time (sequential transformer_forward() calls, sampled with the same
 * temperature/top_p as plain generation), the verifier checks all of them
 * in a single transformer_forward_batch() call, accept_reject_round()
 * (speculative/accept_reject.h) decides how many to keep, the accepted
 * tokens plus one resampled/bonus token are emitted, and both models
 * commit that bonus token with one corrective transformer_forward() call
 * each before drafting the next round -- see spec_decode.c's header
 * comment for why this is the exact same "commit a token" mechanism
 * generate_with_callback() already uses, not a special rollback path.
 *
 * Known limitations, both refused explicitly rather than silently
 * mishandled:
 *   - mc->has_linear_attn models (Qwen3.5/3.6 hybrid): refused at startup,
 *     matching transformer_forward_batch()'s own refusal.
 *   - A prompt continuing from a nonzero s->current_pos (e.g. vision
 *     prefill via vision_pipeline_run()): the draft model has no matching
 *     prefix in its own KV cache, so the two models' position spaces would
 *     diverge. Refused at startup; use plain generate_with_callback()
 *     instead for that case.
 *   - json_mode: grammar-constrained decoding is not wired into the
 *     draft/accept-reject loop this pass -- passing json_mode=true prints
 *     a warning and proceeds with plain temperature/top_p sampling.
 */
void speculative_generate(const Config *cfg, const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                           const char *prompt, int max_tokens, float temperature,
                           float top_p, bool json_mode,
                           DraftModel *draft, int spec_length);

void speculative_generate_with_callback(const Config *cfg, const TransformerWeights *w,
                                         RunState *s, const MoEConfig *mc,
                                         Tokenizer *tok, ThreadPool *tp, const char *prompt,
                                         int max_tokens, float temperature, float top_p,
                                         bool json_mode, DraftModel *draft, int spec_length,
                                         TokenCallback callback, void *userdata);

#endif /* TN_SPECULATIVE_SPEC_DECODE_H */
