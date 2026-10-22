#include "core/platform.h"

#if TN_HAS_NEON

#include "math/ternary_matmul_packed.h"
#include "core/unpack.h"
#include <arm_neon.h>
#include <string.h>

/**
 * Plain ARM NEON fused unpack+matmul for 2-bit packed ternary weights.
 *
 * Fills the dispatch tier between ARM dotprod (Cortex-A75+/Apple M1+, needs
 * __ARM_FEATURE_DOTPROD) and the scalar fallback: any ARMv8-A core has NEON
 * unconditionally (it is the AArch64 baseline -- cpu_features.c sets
 * f->neon = true unconditionally there), but plenty of real deployment
 * targets predate the dotprod extension -- Raspberry Pi 4 (Cortex-A72,
 * ARMv8.0), older AWS Graviton/Graviton2, and most ARMv8.0/8.1 embedded
 * boards -- and NEON is unconditionally available (mandatory) on all of
 * them too. Before this file, simd_dispatch.c's own comment described this
 * exact tier ("NEON: 4 fp32 MACs/cycle -- stub, always compiled for ARM")
 * but no kernel backed it, so those CPUs silently fell all the way through
 * to the scalar reference for the hottest path in the engine.
 *
 * Strategy mirrors ternary_matmul_packed_avx2.c: unpack 4 weights at a time
 * from packed bytes via vectorized shift+mask (not scalar extraction), then
 * run the same conditional add/subtract pipeline used by every other packed
 * kernel in this family -- no floating-point multiply anywhere.
 *
 * Horizontal sum uses vadd_f32/vpadd_f32 (portable to both ARMv7-A NEON and
 * AArch64) rather than the AArch64-only vaddvq_f32, since TN_HAS_NEON does
 * not itself distinguish the two (see core/platform.h).
 */

static inline int32x4_t unpack4_to_epi32(const tn_u8 *row, int j) {
    /*
     * 4 weights are in 8 bits (1 byte) at row + j/4. Broadcast that byte to
     * all 4 lanes, shift each lane right by its bit-offset (0,2,4,6), mask
     * the low 2 bits, then subtract 1 to map the stored encoding
     * {0b00->-1, 0b01->0, 0b10->1} (see core/unpack.h).
     */
    uint8_t byte;
    memcpy(&byte, row + (j >> 2), sizeof(byte));

    int32x4_t v = vdupq_n_s32((int32_t)byte);
    static const int32_t shift_vals[4] = {0, -2, -4, -6}; /* negative = right shift */
    int32x4_t shifts = vld1q_s32(shift_vals);

    v = vshlq_s32(v, shifts);
    v = vandq_s32(v, vdupq_n_s32(3));
    v = vsubq_s32(v, vdupq_n_s32(1));
    return v;
}

static inline float horizontal_sum_f32(float32x4_t v) {
    float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

#define TN_PREFETCH_ROWS 8

void ternary_matmul_packed_neon(float *out, const float *x, const tn_u8 *packed_w,
                                 int n, int d, const float *scales, int group_size) {
    size_t row_bytes = ((size_t)n + 3) >> 2;

    const int32x4_t ones = vdupq_n_s32(1);
    const int32x4_t neg1 = vdupq_n_s32(-1);

    for (int i = 0; i < d; i++) {
        const tn_u8 *row = packed_w + (size_t)i * row_bytes;

        if (i + TN_PREFETCH_ROWS < d) {
            __builtin_prefetch(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes, 0, 1);
        }

        if (group_size <= 0) {
            /* Per-matrix scale mode */
            float32x4_t accum = vdupq_n_f32(0.0f);
            int j = 0;

            for (; j + 3 < n; j += 4) {
                float32x4_t x_vec = vld1q_f32(&x[j]);
                int32x4_t w32 = unpack4_to_epi32(row, j);

                uint32x4_t mask_pos = vceqq_s32(w32, ones);
                uint32x4_t mask_neg = vceqq_s32(w32, neg1);

                float32x4_t to_add = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(x_vec), mask_pos));
                float32x4_t to_sub = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(x_vec), mask_neg));

                accum = vaddq_f32(accum, to_add);
                accum = vsubq_f32(accum, to_sub);
            }

            float val = horizontal_sum_f32(accum);

            /* Scalar tail */
            for (; j < n; j++) {
                tn_i8 w = unpack_ternary(row, j);
                if (w == 1) val += x[j];
                else if (w == -1) val -= x[j];
            }

            out[i] = val * scales[0];

        } else {
            /* Per-group scale mode */
            float total = 0.0f;
            int n_groups = (n + group_size - 1) / group_size;

            for (int g = 0; g < n_groups; g++) {
                int start = g * group_size;
                int end = start + group_size;
                if (end > n) end = n;

                float32x4_t accum = vdupq_n_f32(0.0f);
                int j = start;

                for (; j + 3 < end; j += 4) {
                    float32x4_t x_vec = vld1q_f32(&x[j]);
                    int32x4_t w32 = unpack4_to_epi32(row, j);

                    uint32x4_t mask_pos = vceqq_s32(w32, ones);
                    uint32x4_t mask_neg = vceqq_s32(w32, neg1);

                    float32x4_t to_add = vreinterpretq_f32_u32(
                        vandq_u32(vreinterpretq_u32_f32(x_vec), mask_pos));
                    float32x4_t to_sub = vreinterpretq_f32_u32(
                        vandq_u32(vreinterpretq_u32_f32(x_vec), mask_neg));

                    accum = vaddq_f32(accum, to_add);
                    accum = vsubq_f32(accum, to_sub);
                }

                float group_sum = horizontal_sum_f32(accum);

                /* Scalar tail for this group */
                for (; j < end; j++) {
                    tn_i8 w = unpack_ternary(row, j);
                    if (w == 1) group_sum += x[j];
                    else if (w == -1) group_sum -= x[j];
                }

                int scale_idx = i * n_groups + g;
                total += group_sum * scales[scale_idx];
            }
            out[i] = total;
        }
    }
}

#endif /* TN_HAS_NEON */
