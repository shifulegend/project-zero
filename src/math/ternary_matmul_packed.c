#include "math/ternary_matmul_packed.h"
#include "core/unpack.h"

/**
 * Scalar reference: packed ternary matmul with per-matrix or per-group scales.
 *
 * Unpacks each row on-the-fly and accumulates. For per-group mode,
 * scale factors are applied at group boundaries.
 */
void ternary_matmul_packed(float *out, const float *x, const tn_u8 *packed_w,
                           int n, int d, const float *scales, int group_size) {
    size_t row_bytes = packed_bytes(n);

    for (int i = 0; i < d; i++) {
        const tn_u8 *row = packed_w + (size_t)i * row_bytes;

        if (group_size <= 0) {
            /* Per-matrix scale mode: single scale factor */
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                tn_i8 w = unpack_ternary(row, j);
                if (w == 1) {
                    val += x[j];
                } else if (w == -1) {
                    val -= x[j];
                }
            }
            out[i] = val * scales[0];
        } else {
            /* Per-group scale mode: one scale per group */
            float total = 0.0f;
            int n_groups = (n + group_size - 1) / group_size;

            for (int g = 0; g < n_groups; g++) {
                float group_sum = 0.0f;
                int start = g * group_size;
                int end = start + group_size;
                if (end > n) end = n;

                for (int j = start; j < end; j++) {
                    tn_i8 w = unpack_ternary(row, j);
                    if (w == 1) {
                        group_sum += x[j];
                    } else if (w == -1) {
                        group_sum -= x[j];
                    }
                }

                /* Scale index: per-row, per-group */
                int scale_idx = i * n_groups + g;
                total += group_sum * scales[scale_idx];
            }
            out[i] = total;
        }
    }
}
