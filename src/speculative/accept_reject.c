#include "speculative/accept_reject.h"

#include <stdlib.h>
#include <string.h>

#include "math/simd_dispatch.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"

static int cdf_sample(const float *probs, int vocab_size, unsigned long long *rng_state) {
    float r = rng_float(rng_state);
    float cdf = 0.0f;
    for (int j = 0; j < vocab_size; j++) {
        cdf += probs[j];
        if (r < cdf) return j;
    }
    return vocab_size - 1; /* fallback for float rounding */
}

static void greedy_round(const int *draft_tokens, const float *verify_check_logits,
                          const float *bonus_logits, int spec_length, int vocab_size,
                          int *n_accept, int *emit_token) {
    for (int i = 0; i < spec_length; i++) {
        const float *vlog = verify_check_logits + (size_t)i * vocab_size;
        int verifier_pick = sample_argmax(vlog, vocab_size);
        if (draft_tokens[i] != verifier_pick) {
            *n_accept = i;
            *emit_token = verifier_pick;
            return;
        }
    }
    *n_accept = spec_length;
    *emit_token = sample_argmax(bonus_logits, vocab_size);
}

void accept_reject_round(const int *draft_tokens, const float *draft_logits,
                          const float *verify_check_logits, const float *bonus_logits,
                          int spec_length, int vocab_size, float temperature,
                          unsigned long long *rng_state,
                          int *n_accept, int *emit_token) {
    if (temperature <= 0.0f || temperature < 1e-6f) {
        greedy_round(draft_tokens, verify_check_logits, bonus_logits, spec_length, vocab_size,
                     n_accept, emit_token);
        return;
    }

    float *p_draft = (float *)malloc((size_t)vocab_size * sizeof(float));
    float *p_verifier = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!p_draft || !p_verifier) {
        /* OOM: fall back to greedy rather than crash -- deterministic degradation. */
        free(p_draft);
        free(p_verifier);
        greedy_round(draft_tokens, verify_check_logits, bonus_logits, spec_length, vocab_size,
                     n_accept, emit_token);
        return;
    }

    int accept_count = 0;
    for (int i = 0; i < spec_length; i++) {
        memcpy(p_draft, draft_logits + (size_t)i * vocab_size, (size_t)vocab_size * sizeof(float));
        memcpy(p_verifier, verify_check_logits + (size_t)i * vocab_size,
               (size_t)vocab_size * sizeof(float));
        apply_temperature(p_draft, vocab_size, temperature);
        apply_temperature(p_verifier, vocab_size, temperature);
        tn_softmax(p_draft, vocab_size);
        tn_softmax(p_verifier, vocab_size);

        int x = draft_tokens[i];
        float pd = p_draft[x];
        float pv = p_verifier[x];
        float accept_prob = (pd > 0.0f) ? (pv / pd) : 1.0f;
        if (accept_prob > 1.0f) accept_prob = 1.0f;

        if (rng_float(rng_state) < accept_prob) {
            accept_count++;
            continue;
        }

        /* Reject: resample from the renormalized residual max(0, p_verifier - p_draft),
         * reusing p_verifier as the residual buffer. */
        float sum = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            float v = p_verifier[j] - p_draft[j];
            if (v < 0.0f) v = 0.0f;
            p_verifier[j] = v;
            sum += v;
        }
        int resampled;
        if (sum <= 0.0f) {
            /* Degenerate (draft and verifier distributions coincide exactly):
             * fall back to the verifier's own argmax at this position. */
            resampled = sample_argmax(verify_check_logits + (size_t)i * vocab_size, vocab_size);
        } else {
            for (int j = 0; j < vocab_size; j++) p_verifier[j] /= sum;
            resampled = cdf_sample(p_verifier, vocab_size, rng_state);
        }
        *n_accept = accept_count;
        *emit_token = resampled;
        free(p_draft);
        free(p_verifier);
        return;
    }

    /* All accepted: sample the bonus token from softmax(bonus_logits/temperature). */
    memcpy(p_verifier, bonus_logits, (size_t)vocab_size * sizeof(float));
    apply_temperature(p_verifier, vocab_size, temperature);
    tn_softmax(p_verifier, vocab_size);
    *n_accept = accept_count;
    *emit_token = cdf_sample(p_verifier, vocab_size, rng_state);
    free(p_draft);
    free(p_verifier);
}
