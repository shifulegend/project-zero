#include "sampling/rng.h"
#include "sampling/sampling.h"
#include <math.h>
#include <stdlib.h>

/* Sort helper: index + probability pair for descending sort */
typedef struct {
  float prob;
  int index;
} ProbIndex;

static int compare_prob_desc(const void *a, const void *b) {
  float pa = ((const ProbIndex *)a)->prob;
  float pb = ((const ProbIndex *)b)->prob;
  if (pa > pb)
    return -1;
  if (pa < pb)
    return 1;
  return 0;
}

/*
 * Static candidate buffer: avoids per-call malloc/free (which cost ~1 ms at
 * 128K vocab).  Single-threaded only — generation is always called from one
 * thread, so this is safe.
 */
#define MAX_VOCAB_SIZE 200000
static ProbIndex static_candidates[MAX_VOCAB_SIZE];

int sample_top_p(float *logits, int vocab_size, float top_p,
                 unsigned long long *rng_state) {
  /* Guard: if vocab exceeds static buffer, fall back to argmax */
  if (vocab_size > MAX_VOCAB_SIZE)
    return sample_argmax(logits, vocab_size);

  /* Step 1: Find max for numerical stability */
  float max_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val)
      max_val = logits[i];
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
  int n = 0;
  float sum = 0.0f;

  for (int i = 0; i < vocab_size; i++) {
    if (logits[i] >= log_threshold) {
      float e = expf(logits[i] - max_val);
      static_candidates[n].prob  = e;
      static_candidates[n].index = i;
      sum += e;
      n++;
    }
  }

  /* Edge case: all logits below threshold (shouldn't happen in practice) */
  if (n == 0)
    return sample_argmax(logits, vocab_size);

  /* Normalize to probabilities */
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; i++)
    static_candidates[i].prob *= inv_sum;

  /* Step 3: Sort only the filtered candidates (n << vocab_size typically) */
  qsort(static_candidates, n, sizeof(ProbIndex), compare_prob_desc);

  /* Step 4: Accumulate until cumulative >= top_p, find cutoff */
  float cumulative = 0.0f;
  int cutoff = n;
  for (int i = 0; i < n; i++) {
    cumulative += static_candidates[i].prob;
    if (cumulative >= top_p) {
      cutoff = i + 1;
      break;
    }
  }

  /* Step 5: Re-normalize the truncated distribution */
  float trunc_sum = 0.0f;
  for (int i = 0; i < cutoff; i++)
    trunc_sum += static_candidates[i].prob;

  /* Step 6: Sample from the truncated distribution */
  float r = rng_float(rng_state) * trunc_sum;
  float cdf = 0.0f;
  int result = static_candidates[0].index; /* default to top */
  for (int i = 0; i < cutoff; i++) {
    cdf += static_candidates[i].prob;
    if (r < cdf) {
      result = static_candidates[i].index;
      break;
    }
  }

  return result;
}
