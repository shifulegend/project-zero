#include "math/batched_matmul.h"
#include "core/weights.h"
#include <stdint.h>
#include <string.h>

/**
 * Batched classifier-format matmuls (BF16 / block-quantized INT8 / packed
 * INT4). See include/math/batched_matmul.h for the layout convention and
 * the bandwidth-reuse rationale.
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
