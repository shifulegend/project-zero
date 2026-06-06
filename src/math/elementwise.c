#include "math/elementwise.h"
#include <math.h>

void vec_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_mul(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void vec_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) {
        x[i] *= s;
    }
}

void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void relu2_scalar(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] < 0.0f ? 0.0f : x[i];
        x[i] = v * v;
    }
}

float vec_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vec_saxpy(float *out, float scale, const float *v, int n) {
    for (int i = 0; i < n; i++) out[i] += scale * v[i];
}
