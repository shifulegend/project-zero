#ifndef PZ_CLS_KERNELS_H
#define PZ_CLS_KERNELS_H

/*
 * math/cls_kernels — INT8 / INT4 classifier (LM head) row kernels.
 *
 * Layout (weights_build_classifier_quant, core/weights.c): one row per output, blocks of
 * TN_CLS_QUANT_BLOCK (32) elements with one float scale per block and row.
 *   INT8: n bytes per row, value = (byte - 128) * scale
 *   INT4: (n + 1) / 2 bytes per row, low nibble = element 2k, high = 2k+1,
 *         value = (nibble - 8) * scale
 * out[i] = sum over blocks b of scale[i][b] * sum_j value_code[j] * x[j]      (float accumulation)
 *
 * The kernel computes rows [start, end). Two implementations per format:
 *   generic  — AVX2 or scalar, baseline translation unit (cls_kernels.c);
 *   avx512   — AVX-512F, its own translation unit built with the AVX-512 flags in every x86
 *              build (cls_kernels_avx512.c), bound at runtime by tn_simd_init()
 *              (tn_cls_i8_rows / tn_cls_i4_rows in simd_dispatch.h), so the portable
 *              dist binary uses it on any CPU that has the ISA.
 * Both keep the scaled partial sums in vector registers and reduce ONCE PER ROW. The earlier kernel
 * reduced horizontally after every 32-element block (80 reductions per 2560-element row, ~10
 * million per BitNet token), which made the int4 head slower than the BF16 default although it reads
 * ~40% less data (2026-11-18, docs/ai/mistakes.md).
 */

#include "core/platform.h"
#include "core/weights.h"

/* Scalar helpers shared by every implementation (one definition, no ISA code). `from` is the first element of
 * the partial last block (a multiple of TN_CLS_QUANT_BLOCK); the full blocks are the vector kernels' job. */
static inline float tn_cls_i8_row_tail(const tn_u8 *row, const float *x, int from, int n) {
    float v = 0.0f;
    for (int j = from; j < n; j++) v += (float)((int)row[j] - 128) * x[j];
    return v;
}

static inline float tn_cls_i4_row_tail(const tn_u8 *row, const float *x, int from, int n) {
    float v = 0.0f;
    for (int j = from; j < n; j++) {
        int nibble = (j & 1) ? (row[j / 2] >> 4) & 0x0F : row[j / 2] & 0x0F;
        v += (float)(nibble - 8) * x[j];
    }
    return v;
}

typedef void (*tn_cls_rows_fn)(float *out, const float *x, const tn_u8 *w, const float *scales,
                               int n, int n_blocks, int start, int end);

void tn_cls_i8_rows_generic(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                            int n_blocks, int start, int end);
void tn_cls_i4_rows_generic(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                            int n_blocks, int start, int end);

void tn_cls_i8_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end);
void tn_cls_i4_rows_avx512(float *out, const float *x, const tn_u8 *w, const float *scales, int n,
                           int n_blocks, int start, int end);

#endif /* PZ_CLS_KERNELS_H */
