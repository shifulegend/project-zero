#ifndef TN_CONSTRAINED_SAMPLE_H
#define TN_CONSTRAINED_SAMPLE_H

/*
 * constrained_sample.h — Phase 20: grammar-constrained sampling.
 *
 * Same sampling cascade as the main generate() loop (argmax when
 * temperature<=0, else temperature+top-p, else temperature+plain-CDF) with
 * one addition: illegal tokens are masked to -infinity first, so the
 * result is guaranteed to be a token that can extend the in-progress JSON
 * document (or legally terminate it, for EOS). This is what turns "the
 * model was asked nicely for JSON" into "the model is physically incapable
 * of emitting invalid JSON".
 */

#include "sampling/fsm.h"
#include "sampling/sampler_chain.h"

/*
 * Masks `logits` in place via the grammar, samples one token from the
 * masked distribution, advances `fsm`'s grammar state by that token, and
 * returns it. `prev_token` is the token before this one (-1 if none) --
 * needed for tokenizer_decode's BOS-adjacent leading-space rule to see the
 * same text real generation would emit.
 */
int sample_constrained(float *logits, int vocab_size, FSMState *fsm,
                        Tokenizer *t, int prev_token,
                        float temperature, float top_p,
                        unsigned long long *rng_state);

/*
 * Full-settings variant: penalties + watermark bias -> grammar mask ->
 * top-k/min-p/temperature/top-p (sampler_chain.h). `history`/`wm` may be NULL.
 */
int sample_constrained_chain(float *logits, int vocab_size, FSMState *fsm,
                             Tokenizer *t, int prev_token,
                             const SamplerSettings *ss,
                             const int *history, int n_history,
                             const WatermarkConfig *wm,
                             unsigned long long *rng_state);

#endif /* TN_CONSTRAINED_SAMPLE_H */
