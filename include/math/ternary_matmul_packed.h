#ifndef TN_TERNARY_MATMUL_PACKED_H
#define TN_TERNARY_MATMUL_PACKED_H

#include "core/platform.h"

/**
 * Ternary matrix-vector multiplication with 2-bit packed weights.
 *
 * Supports two scale modes:
 *   - Per-matrix: scale is a single float, scale_mode = 0
 *   - Per-group:  scales is an array with ceil(n / group_size) entries per row,
 *                 scale_mode = 1
 *
 * Packed weight format: 4 ternary values per byte, 2 bits each.
 *   -1 -> 0b00, 0 -> 0b01, 1 -> 0b10
 *
 * Phase 10.5 of the implementation plan.
 *
 * @param out        Output vector of size d (must be pre-allocated)
 * @param x          Input vector of size n
 * @param packed_w   Packed weight matrix, row-major, each row has ceil(n/4) bytes
 * @param n          Input dimension (columns, number of ternary weights per row)
 * @param d          Output dimension (rows)
 * @param scales     Scale factor(s): single float if per-matrix, array if per-group
 * @param group_size Group size for per-group scales (0 = per-matrix mode)
 */
void ternary_matmul_packed(float *out, const float *x, const tn_u8 *packed_w,
                           int n, int d, const float *scales, int group_size);

/**
 * AVX2 variant: float32 FMA, 8 MACs/cycle.
 */
void ternary_matmul_packed_avx2(float *out, const float *x, const tn_u8 *packed_w,
                                int n, int d, const float *scales, int group_size);

/**
 * AVX-512F variant: float32 FMA, 16 MACs/cycle.
 */
void ternary_matmul_packed_avx512(float *out, const float *x, const tn_u8 *packed_w,
                                   int n, int d, const float *scales, int group_size);

/**
 * AVX-512 VNNI variant: int8 VNNI, 64 MACs/cycle (K-1a).
 * Targets Ice Lake, Tiger Lake, Zen 4, Sapphire Rapids.
 * Quantizes activations internally; transparent to caller.
 */
void ternary_matmul_packed_vnni(float *out, const float *x, const tn_u8 *packed_w,
                                 int n, int d, const float *scales, int group_size);

/**
 * AVX-512 VNNI pre-quantised variant (K-4 R-3).
 *
 * Same as ternary_matmul_packed_vnni() but accepts activations that have
 * already been quantised to int8 by the caller. Use this when the same
 * activation vector is multiplied against multiple weight matrices (e.g.,
 * Q/K/V projections or FFN gate/up) to avoid redundant quantisation.
 *
 * @param q_x       Pre-quantised int8 activations (length n)
 * @param act_scale Dequantisation scale: true_x ≈ q_x * act_scale
 * @param sum_qx    Sum of q_x[0..n-1] (for VNNI bias correction)
 */
void ternary_matmul_packed_vnni_preq(float *out,
                                      const int8_t *q_x, float act_scale,
                                      int32_t sum_qx,
                                      const tn_u8 *packed_w,
                                      int n, int d,
                                      const float *scales, int group_size);

/**
 * AVX-VNNI 256-bit variant: int8 VNNI, 32 MACs/cycle (K-1b).
 * Targets Alder Lake, Raptor Lake, Arrow Lake, AMD Zen 3.
 * For CPUs with 256-bit VNNI but WITHOUT AVX-512.
 */
void ternary_matmul_packed_avx_vnni(float *out, const float *x, const tn_u8 *packed_w,
                                     int n, int d, const float *scales, int group_size);

/**
 * ARM dotprod variant: SDOT int8, 16 MACs/cycle (K-1c).
 * Targets Apple M1-M4, Cortex-A75+, Snapdragon 8 Gen 1+.
 * Uses vdotq_s32 (signed×signed), no bias correction needed.
 */
void ternary_matmul_packed_dotprod(float *out, const float *x, const tn_u8 *packed_w,
                                    int n, int d, const float *scales, int group_size);

#endif /* TN_TERNARY_MATMUL_PACKED_H */
