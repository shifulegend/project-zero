#ifndef TN_ELEMENTWISE_H
#define TN_ELEMENTWISE_H

/** out[i] = a[i] + b[i] */
void vec_add(float *out, const float *a, const float *b, int n);

/** out[i] = a[i] * b[i]  (Hadamard product, for SwiGLU gate) */
void vec_mul(float *out, const float *a, const float *b, int n);

/** x[i] = x[i] * s */
void vec_scale(float *x, float s, int n);

/** In-place SiLU activation: x[i] = x[i] / (1 + exp(-x[i])) */
void silu(float *x, int n);

/** In-place ReLU² activation: x[i] = max(0, x[i])² */
void relu2_scalar(float *x, int n);

/** Dot product: sum(a[i] * b[i]) */
float vec_dot(const float *a, const float *b, int n);

/** SAXPY: out[i] += scale * v[i] */
void vec_saxpy(float *out, float scale, const float *v, int n);

#endif /* TN_ELEMENTWISE_H */
