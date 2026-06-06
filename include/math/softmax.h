#ifndef TN_SOFTMAX_H
#define TN_SOFTMAX_H

/**
 * In-place numerically stable softmax.
 * Finds max, subtracts it, exponentiates, normalizes.
 */
void softmax(float *x, int size);

#endif /* TN_SOFTMAX_H */
