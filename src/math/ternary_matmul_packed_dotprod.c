#include "core/platform.h"

#if TN_HAS_ARM_DOTPROD

#include "math/ternary_matmul_packed.h"
#include "math/quantize_i8.h"
#include <arm_neon.h>
#include <string.h>
#include <stdint.h>

/*
 * ARM dotprod ternary matmul (Phase 16-S.4 — K-1c)
 *
 * Targets Apple M1/M2/M3/M4, Cortex-A75+, Snapdragon 8 Gen 1+, AWS Graviton 3+.
 * Any ARM CPU with the __ARM_FEATURE_DOTPROD extension (ARMv8.2-A or later).
 *
 * Unlike the x86 dpbusds (unsigned x signed), the ARM vdotq_s32 instruction
 * is SIGNED × SIGNED, which means:
 *   - Weights ∈ {-1, 0, +1} as int8_t — no bias encoding needed
 *   - Activations ∈ [-127, 127] as int8_t — direct use
 *   - No bias correction: vdotq_s32(acc, w_i8, q_x_i8) = exact dot product
 *
 * This is cleaner than the x86 VNNI approach because ARM dotprod supports
 * signed × signed natively.  The w_enc trick and sum_qx correction are NOT
 * needed here.
 *
 * vdotq_s32: processes 16 int8 (4 groups of 4) → 4 int32 accumulators.
 * One call: 16 weights × 16 activations = 16 MACs.
 * Per cycle throughput on Cortex-A75+ / Apple M-series: 1 cycle latency.
 *
 * Weight unpacking:
 *   Scalar per group of 16 (same cost relative to dotprod savings).
 *   Future: replace with NEON VTBL lookup for vectorized unpack.
 */

#define TN_DOTPROD_MAX_N 16384
#define TN_PREFETCH_ROWS 4

/*
 * Unpack 16 packed 2-bit weights (4 bytes) into a signed int8x16_t
 * with values {-1, 0, +1} directly (no w_enc bias needed for ARM).
 */
static inline int8x16_t unpack16_to_i8(const tn_u8 *row, int j)
{
    uint32_t bits;
    memcpy(&bits, row + (j >> 2), sizeof(bits));

    int8_t buf[16];
    for (int k = 0; k < 16; k++) {
        /* 2-bit encoding: 0→-1, 1→0, 2→+1 (same as x86 kernel, subtract 1) */
        buf[k] = (int8_t)(((bits >> (k * 2)) & 3) - 1);
    }
    return vld1q_s8(buf);
}

void ternary_matmul_packed_dotprod(float *out, const float *x, const tn_u8 *packed_w,
                                    int n, int d, const float *scales, int group_size)
{
    if (n > TN_DOTPROD_MAX_N) {
        /* Fallback to portable scalar kernel when n exceeds dotprod buffer.
         * ternary_matmul_packed() handles all input sizes correctly on
         * all platforms (x86, ARM, etc.) — no platform-specific symbol. */
        extern void ternary_matmul_packed(float*, const float*, const tn_u8*,
                                          int, int, const float*, int);
        ternary_matmul_packed(out, x, packed_w, n, d, scales, group_size);
        return;
    }

    /* Quantize activations once (reused for all d rows) */
    int8_t  q_x[TN_DOTPROD_MAX_N];
    float   act_scale = quantize_row_to_i8(x, q_x, n);

    if (act_scale == 0.0f) {
        memset(out, 0, (size_t)d * sizeof(float));
        return;
    }

    /* NOTE: No sum_qx needed — ARM vdotq_s32 is signed×signed, exact dot product. */

    size_t row_bytes = ((size_t)n + 3) >> 2;

    if (group_size <= 0) {
        float w_scale = scales[0];

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            /* ARM prefetch */
            if (i + TN_PREFETCH_ROWS < d) {
                __builtin_prefetch(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes, 0, 1);
            }

            int32x4_t acc = vdupq_n_s32(0);
            int j = 0;

            /* 16-weight dotprod loop */
            for (; j + 15 < n; j += 16) {
                int8x16_t w_i8  = unpack16_to_i8(row, j);
                int8x16_t qx_i8 = vld1q_s8(q_x + j);
                acc = vdotq_s32(acc, w_i8, qx_i8);
            }

            /* Horizontal sum of 4 int32 lanes */
            int32_t result = vaddvq_s32(acc);

            /* Scalar tail */
            for (; j < n; j++) {
                int w = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3) - 1;
                result += w * (int32_t)q_x[j];
            }

            out[i] = (float)result * act_scale * w_scale;
        }

    } else {
        int n_groups = (n + group_size - 1) / group_size;

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d) {
                __builtin_prefetch(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes, 0, 1);
            }

            float total = 0.0f;

            for (int g = 0; g < n_groups; g++) {
                int gs = g * group_size;
                int ge = gs + group_size;
                if (ge > n) ge = n;

                int32x4_t acc = vdupq_n_s32(0);
                int j = gs;

                for (; j + 15 < ge; j += 16) {
                    int8x16_t w_i8  = unpack16_to_i8(row, j);
                    int8x16_t qx_i8 = vld1q_s8(q_x + j);
                    acc = vdotq_s32(acc, w_i8, qx_i8);
                }

                int32_t result = vaddvq_s32(acc);

                for (; j < ge; j++) {
                    int w = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3) - 1;
                    result += w * (int32_t)q_x[j];
                }

                total += (float)result * act_scale * scales[i * n_groups + g];
            }

            out[i] = total;
        }
    }
}

#endif /* TN_HAS_ARM_DOTPROD */
