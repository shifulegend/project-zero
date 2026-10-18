#ifndef TN_LOGPROB_H
#define TN_LOGPROB_H

/*
 * logprob.h — log-probabilities of sampled tokens under the model's raw
 * (temperature-1, unpenalised) distribution. Used to score candidates for
 * Phase 30 best-of-N selection and reported in usage stats.
 */

/* log(sum_i exp(logits[i])), numerically stable. -INFINITY for n <= 0. */
double logprob_logsumexp(const float *logits, int n);

#endif /* TN_LOGPROB_H */
