#include "math/batched_matmul.h"
#include "math/simd_dispatch.h"
#include "core/weights.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * Batched classifier-format matmuls (BF16 / block-quantized INT8 / packed
 * INT4). See include/math/batched_matmul.h for the layout convention and
 * the bandwidth-reuse rationale.
 *
 * 2026-09-24 rewrite: see batched_matmul_f16.c's matching header comment for
 * the full root-cause writeup (docs/ai/mistakes.md has the real-model
 * measurement). Same fix applied to all three formats here: dequantize each
 * row to F32 ONCE per row, then reuse the SIMD-dispatched tn_vec_dot() for
 * each of the n_tokens dot products against that row -- the classifier
 * matmul's output width is vocab_size (tens of thousands for real models),
 * making it one of the largest-FLOP calls in the whole forward pass, so a
 * scalar-per-token accumulator here was especially costly.
 */

/* ── BF16 ────────────────────────────────────────────────────────────────── */

typedef struct {
    float        *out;
    const float  *x;
    const tn_u16 *w;
    int n, d, n_tokens;
} Bf16BatchArgs;

static void matmul_bf16_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    Bf16BatchArgs *a = (Bf16BatchArgs *)arg;

    float *row_f32 = (float *)malloc((size_t)a->n * sizeof(float));
    if (!row_f32) {
        for (int i = start; i < end; i++) {
            const tn_u16 *row = a->w + (size_t)i * a->n;
            for (int k = 0; k < a->n_tokens; k++) {
                const float *xk = a->x + (size_t)k * a->n;
                float val = 0.0f;
                for (int j = 0; j < a->n; j++) {
                    tn_u32 bits = (tn_u32)row[j] << 16;
                    float wval; memcpy(&wval, &bits, sizeof(wval));
                    val += wval * xk[j];
                }
                a->out[(size_t)k * a->d + i] = val;
            }
        }
        return;
    }

    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        for (int j = 0; j < a->n; j++) {
            tn_u32 bits = (tn_u32)row[j] << 16;
            memcpy(&row_f32[j], &bits, sizeof(float));
        }
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            a->out[(size_t)k * a->d + i] = tn_vec_dot(row_f32, xk, a->n);
        }
    }
    free(row_f32);
}

void tn_matmul_bf16_batch(float *out, const float *x, const tn_u16 *w,
                           int n, int d, int n_tokens, ThreadPool *tp) {
    Bf16BatchArgs args = { .out = out, .x = x, .w = w, .n = n, .d = d, .n_tokens = n_tokens };
    if (!tp) {
        matmul_bf16_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_bf16_batch_task, &args, d);
}

/* ── INT8, block-quantized (TN_CLS_QUANT_BLOCK elements/block) ────────────── */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;
    const float *scales;
    int n, d, n_tokens, n_blocks;
} I8BatchArgs;

static void matmul_i8_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const I8BatchArgs *a = (const I8BatchArgs *)arg;
    const int nb = a->n_blocks;

    float *row_f32 = (float *)malloc((size_t)a->n * sizeof(float));
    if (!row_f32) {
        for (int i = start; i < end; i++) {
            const tn_u8 *row = a->w + (size_t)i * a->n;
            const float *row_scales = a->scales + (size_t)i * nb;
            for (int k = 0; k < a->n_tokens; k++) {
                const float *xk = a->x + (size_t)k * a->n;
                float total = 0.0f;
                for (int b = 0; b < nb; b++) {
                    int blk_start = b * TN_CLS_QUANT_BLOCK;
                    int blk_len = a->n - blk_start;
                    if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
                    const tn_u8 *brow = row + blk_start;
                    const float *bx = xk + blk_start;
                    float val = 0.0f;
                    for (int j = 0; j < blk_len; j++) {
                        val += (float)((int)brow[j] - 128) * bx[j];
                    }
                    total += val * row_scales[b];
                }
                a->out[(size_t)k * a->d + i] = total;
            }
        }
        return;
    }

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * a->n;
        const float *row_scales = a->scales + (size_t)i * nb;
        /* Dequantize this row to F32 ONCE, folding each block's scale in
         * directly -- turns the per-token cost below into a single plain
         * dot product over the whole row, not a nested block/element loop
         * repeated per token. */
        for (int b = 0; b < nb; b++) {
            int blk_start = b * TN_CLS_QUANT_BLOCK;
            int blk_len = a->n - blk_start;
            if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
            float s = row_scales[b];
            for (int j = 0; j < blk_len; j++) {
                row_f32[blk_start + j] = (float)((int)row[blk_start + j] - 128) * s;
            }
        }
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            a->out[(size_t)k * a->d + i] = tn_vec_dot(row_f32, xk, a->n);
        }
    }
    free(row_f32);
}

void tn_matmul_i8_batch(float *out, const float *x, const tn_u8 *w,
                         const float *scales, int n, int d, int n_tokens, ThreadPool *tp) {
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    I8BatchArgs args = {
        .out = out, .x = x, .w = w, .scales = scales,
        .n = n, .d = d, .n_tokens = n_tokens, .n_blocks = n_blocks
    };
    if (!tp) {
        matmul_i8_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_i8_batch_task, &args, d);
}

/* ── INT4, packed 2/byte, block-quantized ──────────────────────────────────
 * Unsigned storage with +8 bias: signed value + 8 -> [1, 15]. */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *w;
    const float *scales;
    int n, d, n_tokens, n_blocks;
} I4BatchArgs;

static void matmul_i4_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const I4BatchArgs *a = (const I4BatchArgs *)arg;
    size_t row_bytes = ((size_t)a->n + 1) / 2;
    const int nb = a->n_blocks;

    float *row_f32 = (float *)malloc((size_t)a->n * sizeof(float));
    if (!row_f32) {
        for (int i = start; i < end; i++) {
            const tn_u8 *row = a->w + (size_t)i * row_bytes;
            const float *row_scales = a->scales + (size_t)i * nb;
            for (int k = 0; k < a->n_tokens; k++) {
                const float *xk = a->x + (size_t)k * a->n;
                float total = 0.0f;
                for (int b = 0; b < nb; b++) {
                    int blk_start = b * TN_CLS_QUANT_BLOCK;
                    int blk_len = a->n - blk_start;
                    if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
                    const float *bx = xk + blk_start;
                    float val = 0.0f;
                    for (int j = 0; j < blk_len; j++) {
                        int abs_j = blk_start + j;
                        int nibble;
                        if (abs_j & 1) nibble = (row[abs_j / 2] >> 4) & 0x0F;
                        else nibble = row[abs_j / 2] & 0x0F;
                        val += (float)(nibble - 8) * bx[j];
                    }
                    total += val * row_scales[b];
                }
                a->out[(size_t)k * a->d + i] = total;
            }
        }
        return;
    }

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->w + (size_t)i * row_bytes;
        const float *row_scales = a->scales + (size_t)i * nb;
        for (int b = 0; b < nb; b++) {
            int blk_start = b * TN_CLS_QUANT_BLOCK;
            int blk_len = a->n - blk_start;
            if (blk_len > TN_CLS_QUANT_BLOCK) blk_len = TN_CLS_QUANT_BLOCK;
            float s = row_scales[b];
            for (int j = 0; j < blk_len; j++) {
                int abs_j = blk_start + j;
                int nibble;
                if (abs_j & 1) nibble = (row[abs_j / 2] >> 4) & 0x0F;
                else nibble = row[abs_j / 2] & 0x0F;
                row_f32[abs_j] = (float)(nibble - 8) * s;
            }
        }
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            a->out[(size_t)k * a->d + i] = tn_vec_dot(row_f32, xk, a->n);
        }
    }
    free(row_f32);
}

void tn_matmul_i4_batch(float *out, const float *x, const tn_u8 *w,
                         const float *scales, int n, int d, int n_tokens, ThreadPool *tp) {
    int n_blocks = (n + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    I4BatchArgs args = {
        .out = out, .x = x, .w = w, .scales = scales,
        .n = n, .d = d, .n_tokens = n_tokens, .n_blocks = n_blocks
    };
    if (!tp) {
        matmul_i4_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_i4_batch_task, &args, d);
}
