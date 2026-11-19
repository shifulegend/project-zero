#include <math.h>
#include <stdlib.h>

#include "sampling/rng.h"
#include "sampling/sampling.h"

/* Sort helper: index + probability pair for descending sort */
typedef struct {
    float prob;
    int index;
} ProbIndex;

static int compare_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/*
 * Candidate buffer: the nucleus pre-filter below keeps only tokens within 20 nats of the maximum,
 * which is a few dozen in practice, so a small stack array covers the normal case without any
 * allocation. A flat distribution (every token a candidate) falls back to one malloc for that
 * call. There is no shared state, so concurrent samplers (parallel slots, several connection
 * threads) cannot interfere.
 */
#define TOP_P_STACK_CANDIDATES 4096

int sample_top_p(float *logits, int vocab_size, float top_p, unsigned long long *rng_state) {
    if (vocab_size <= 0) return 0;

    /* Step 1: Find max for numerical stability */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    /*
   * Step 2: Pre-filter tokens unlikely to be in the top-p nucleus.
   *
   * Tokens where exp(logit - max) < exp(-20) ≈ 2e-9 contribute negligibly
   * (< 2e-9 / sum) to any reasonable distribution.  Skipping them avoids
   * ~90-95% of expf() calls and proportionally reduces qsort() cost.
   *
   * The filter is conservative: with top_p=0.9 and temperature=0.7, any token
   * that could ever enter the top-p nucleus must have logit ≥ max - 20.
   */
    float log_threshold = max_val - 20.0f;
    int n_candidates    = 0;
    for (int i = 0; i < vocab_size; i++) n_candidates += (logits[i] >= log_threshold);

    /* Edge case: all logits below threshold (shouldn't happen in practice) */
    if (n_candidates == 0) return sample_argmax(logits, vocab_size);

    ProbIndex stack_candidates[TOP_P_STACK_CANDIDATES];
    ProbIndex *cand = stack_candidates;
    if (n_candidates > TOP_P_STACK_CANDIDATES) {
        cand = (ProbIndex *)malloc((size_t)n_candidates * sizeof(ProbIndex));
        if (!cand) return sample_argmax(logits, vocab_size); /* out of memory: stay deterministic */
    }

    int n     = 0;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= log_threshold) {
            float e       = expf(logits[i] - max_val);
            cand[n].prob  = e;
            cand[n].index = i;
            sum += e;
            n++;
        }
    }

    /* A +inf logit makes exp(logit - max) NaN: no usable distribution, so stay deterministic
     * (2026-11-19). Finite logits always give sum >= 1 (the max itself contributes exp(0)). */
    if (!(sum > 0.0f) || !isfinite(sum)) {
        if (cand != stack_candidates) free(cand);
        return sample_argmax(logits, vocab_size);
    }

    /* Normalize to probabilities */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) cand[i].prob *= inv_sum;

    /* Step 3: Sort only the filtered candidates (n << vocab_size typically) */
    qsort(cand, n, sizeof(ProbIndex), compare_prob_desc);

    /* Step 4: Accumulate until cumulative >= top_p, find cutoff */
    float cumulative = 0.0f;
    int cutoff       = n;
    for (int i = 0; i < n; i++) {
        cumulative += cand[i].prob;
        if (cumulative >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Step 5: Re-normalize the truncated distribution */
    float trunc_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) trunc_sum += cand[i].prob;

    /* Step 6: Sample from the truncated distribution */
    float r    = rng_float(rng_state) * trunc_sum;
    float cdf  = 0.0f;
    int result = cand[0].index; /* default to top */
    for (int i = 0; i < cutoff; i++) {
        cdf += cand[i].prob;
        if (r < cdf) {
            result = cand[i].index;
            break;
        }
    }

    if (cand != stack_candidates) free(cand);
    return result;
}
