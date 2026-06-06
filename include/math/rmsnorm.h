#ifndef TN_RMSNORM_H
#define TN_RMSNORM_H

/**
 * RMS Normalization.
 * out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
 */
void rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

#endif /* TN_RMSNORM_H */
