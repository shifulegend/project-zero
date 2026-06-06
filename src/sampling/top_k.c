#include "sampling/rng.h"
#include "sampling/sampling.h"
#include <math.h>

/* Maximum k for stack-allocated top-k arrays. Values above this are clamped. */
#define TOP_K_MAX 1024

int sample_top_k(float *logits, int vocab_size, int k,
                 unsigned long long *rng_state) {
  /* Clamp k to vocab_size */
  if (k > vocab_size)
    k = vocab_size;
  if (k <= 0)
    k = 1;
  /* Clamp k to TOP_K_MAX to prevent stack buffer overflow (QA-BUG-001) */
  if (k > TOP_K_MAX)
    k = TOP_K_MAX;

  /* Step 1: Find top-k indices via partial selection.
   * We use a simple O(k*n) approach which is fine for typical vocab sizes
   * and small k values. For production, a partial quickselect would be better.
   */
  int top_indices[TOP_K_MAX]; /* Stack buffer; k is clamped to TOP_K_MAX */
  int *indices = top_indices;
  float top_vals[TOP_K_MAX];
  float *vals = top_vals;

  /* Initialize with -inf */
  for (int i = 0; i < k; i++) {
    indices[i] = -1;
    vals[i] = -1e30f;
  }

  /* Find top-k by maintaining a sorted array of the k best */
  for (int i = 0; i < vocab_size; i++) {
    /* Check if this logit is better than the worst in our top-k */
    if (logits[i] > vals[k - 1]) {
      /* Insert in sorted position */
      int j = k - 1;
      while (j > 0 && logits[i] > vals[j - 1]) {
        vals[j] = vals[j - 1];
        indices[j] = indices[j - 1];
        j--;
      }
      vals[j] = logits[i];
      indices[j] = i;
    }
  }

  /* Step 2: Softmax over the top-k values */
  float max_val = vals[0];
  float sum = 0.0f;
  for (int i = 0; i < k; i++) {
    vals[i] = expf(vals[i] - max_val);
    sum += vals[i];
  }
  for (int i = 0; i < k; i++) {
    vals[i] /= sum;
  }

  /* Step 3: Sample from the distribution */
  float r = rng_float(rng_state);
  float cdf = 0.0f;
  for (int i = 0; i < k; i++) {
    cdf += vals[i];
    if (r < cdf) {
      return indices[i];
    }
  }

  /* Fallback to top-1 */
  return indices[0];
}
