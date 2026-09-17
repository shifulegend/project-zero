#include "sampling/constrained_sample.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "math/simd_dispatch.h"

int sample_constrained(float *logits, int vocab_size, FSMState *fsm,
                        Tokenizer *t, int prev_token,
                        float temperature, float top_p,
                        unsigned long long *rng_state) {
    fsm_compute_token_mask(fsm, t, prev_token, logits, vocab_size);

    int next;
    if (temperature <= 0.0f || temperature < 1e-6f) {
        next = sample_argmax(logits, vocab_size);
    } else {
        apply_temperature(logits, vocab_size, temperature);
        if (top_p < 1.0f && top_p > 0.0f) {
            next = sample_top_p(logits, vocab_size, top_p, rng_state);
        } else {
            tn_softmax(logits, vocab_size);
            float r = rng_float(rng_state);
            float cdf = 0.0f;
            next = vocab_size - 1; /* fallback */
            for (int i = 0; i < vocab_size; i++) {
                cdf += logits[i];
                if (r < cdf) { next = i; break; }
            }
        }
    }

    fsm_advance(fsm, t, prev_token, next);
    return next;
}
