#ifndef TN_SAMPLING_H
#define TN_SAMPLING_H

/**
 * Argmax (greedy) sampler.
 * Returns the index of the maximum logit value. Deterministic.
 *
 * @param logits      Array of logit values
 * @param vocab_size  Number of entries in logits
 * @return            Index of the maximum value
 */
int sample_argmax(const float *logits, int vocab_size);

/**
 * Apply temperature scaling to logits (in-place).
 * Divides all logits by the temperature value before softmax.
 * Temperature > 1.0 = more random, < 1.0 = more focused.
 *
 * @param logits      Array of logit values (modified in-place)
 * @param vocab_size  Number of entries
 * @param temperature Temperature value (must be > 0)
 */
void apply_temperature(float *logits, int vocab_size, float temperature);

/**
 * Top-p (nucleus) sampling.
 * Sorts logits descending, applies softmax, accumulates probabilities
 * until sum >= top_p, then samples from the truncated distribution.
 *
 * @param logits      Array of logit values (modified in-place by softmax)
 * @param vocab_size  Number of entries
 * @param top_p       Cumulative probability threshold (0.0 to 1.0)
 * @param rng_state   Pointer to RNG state
 * @return            Sampled token index
 */
int sample_top_p(float *logits, int vocab_size, float top_p,
                 unsigned long long *rng_state);

/**
 * Top-k sampling.
 * Finds the top-k logits, applies softmax over them, and samples.
 *
 * @param logits      Array of logit values (modified in-place)
 * @param vocab_size  Number of entries
 * @param k           Number of top candidates to consider
 * @param rng_state   Pointer to RNG state
 * @return            Sampled token index
 */
int sample_top_k(float *logits, int vocab_size, int k,
                 unsigned long long *rng_state);

#endif /* TN_SAMPLING_H */
