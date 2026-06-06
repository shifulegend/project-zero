#include "math/rmsnorm.h"
#include <math.h>

void rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    /* Calculate sum of squares */
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss = 1.0f / sqrtf(ss + eps);

    /* Normalize and scale */
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}
