#include "math/ternary_matmul.h"

/**
 * Scalar reference ternary matrix-vector multiplication.
 *
 * This is the baseline implementation all SIMD variants are tested against.
 * No floating-point multiplications in the inner loop — only conditional
 * add/subtract based on ternary weight values {-1, 0, 1}.
 *
 * Scale factor note (Phase 10 forward-looking):
 * Currently, `scale` is applied per-matrix (one scale factor per weight matrix).
 * When we move to 2-bit packed weights (Phase 10), many quantization schemes
 * use per-group scale factors (e.g., one scale per 128 weights) rather than
 * per-matrix. To support this:
 *   - The inner loop will need to apply scale factors at group boundaries
 *   - The signature may change to accept a `const float *scales` array
 *     with `n_groups = ceil(n / group_size)` entries
 *   - The accumulation becomes: out[i] = sum_g( scale[g] * sum_j_in_g(...) )
 * For now, per-matrix scale is correct for unpacked int8 ternary weights.
 */
void ternary_matmul(float *out, const float *x, const tn_i8 *w,
                    int n, int d, float scale) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const tn_i8 *row = w + (size_t)i * n;

        for (int j = 0; j < n; j++) {
            tn_i8 weight = row[j];
            if (weight == 1) {
                val += x[j];
            } else if (weight == -1) {
                val -= x[j];
            }
            /* weight == 0: skip entirely */
        }

        out[i] = val * scale;
    }
}
