#include "sampling/constrained_sample.h"
#include "sampling/sampler_chain.h"

int sample_constrained_chain(float *logits, int vocab_size, FSMState *fsm,
                             Tokenizer *t, int prev_token,
                             const SamplerSettings *ss,
                             const int *history, int n_history,
                             const WatermarkConfig *wm,
                             unsigned long long *rng_state) {
    /* Penalties and watermark bias first, then the grammar mask, then the
     * truncation/temperature/sampling tail. Top-k/min-p run AFTER the grammar
     * mask so they can never cut away every grammatically legal token. */
    sampler_apply_penalties(logits, vocab_size, ss, history, n_history);
    watermark_bias_logits(wm, logits, vocab_size, prev_token);
    fsm_compute_token_mask(fsm, t, prev_token, logits, vocab_size);
    int next = sampler_chain_finish(logits, vocab_size, ss, rng_state);
    fsm_advance(fsm, t, prev_token, next);
    return next;
}

int sample_constrained(float *logits, int vocab_size, FSMState *fsm,
                        Tokenizer *t, int prev_token,
                        float temperature, float top_p,
                        unsigned long long *rng_state) {
    SamplerSettings ss;
    sampler_settings_default(&ss);
    ss.temperature = temperature;
    ss.top_p = top_p;
    return sample_constrained_chain(logits, vocab_size, fsm, t, prev_token, &ss,
                                    NULL, 0, NULL, rng_state);
}
