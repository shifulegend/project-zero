#include "math/softmax.h"
#include <math.h>

void softmax(float *x, int size) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}
