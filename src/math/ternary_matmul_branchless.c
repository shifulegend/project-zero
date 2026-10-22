#include "math/ternary_matmul.h"

/**
 * Phase 3.2 — branchless scalar ternary matvec.
 *
 * Same contract as ternary_matmul() (the reference), but the inner loop is
 * `val += (float)w * x` with w ∈ {-1, 0, 1}: no data-dependent branches, so
 * the compiler can auto-vectorise it (int8 -> float convert + FMA) even on
 * the scalar backend, and random sign patterns never mispredict. Four
 * independent accumulators break the FP add dependency chain. Summation
 * order differs from the reference, so results match to rounding, not bits.
 */
void ternary_matmul_branchless(float *out, const float *x, const tn_i8 *w,
                               int n, int d, float scale) {
    for (int i = 0; i < d; i++) {
        const tn_i8 *row = w + (size_t)i * n;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            a0 += (float)row[j]     * x[j];
            a1 += (float)row[j + 1] * x[j + 1];
            a2 += (float)row[j + 2] * x[j + 2];
            a3 += (float)row[j + 3] * x[j + 3];
        }
        for (; j < n; j++) a0 += (float)row[j] * x[j];
        out[i] = ((a0 + a1) + (a2 + a3)) * scale;
    }
}
