#include "math/batched_matmul.h"
#include "core/unpack.h"
#include <stddef.h>

/**
 * Ternary-packed batched matmul. See include/math/batched_matmul.h for the
 * layout convention and the bandwidth-reuse rationale.
 */

typedef struct {
    float       *out;
    const float *x;
    const tn_u8 *packed_w;
    int n, d, n_tokens;
    float scale;
} TernaryBatchArgs;

static void ternary_matmul_packed_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    TernaryBatchArgs *a = (TernaryBatchArgs *)arg;
    size_t row_bytes = packed_bytes(a->n);

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->packed_w + (size_t)i * row_bytes;
        /* `row` is small (ceil(n/4) bytes) -- it is read from RAM once here
         * and stays resident in L1/L2 across the n_tokens loop below, since
         * row_bytes is tiny relative to cache size. That is what delivers
         * the batched-verification bandwidth win (this row's bytes are
         * fetched from RAM once total, not once per candidate token). */
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            float val = 0.0f;
            for (int j = 0; j < a->n; j++) {
                tn_i8 w = unpack_ternary(row, j);
                if (w == 1) val += xk[j];
                else if (w == -1) val -= xk[j];
            }
            a->out[(size_t)k * a->d + i] = val * a->scale;
        }
    }
}

void tn_ternary_matmul_packed_batch(float *out, const float *x, const tn_u8 *packed_w,
                                     int n, int d, float scale, int n_tokens, ThreadPool *tp) {
    TernaryBatchArgs args = {
        .out = out, .x = x, .packed_w = packed_w,
        .n = n, .d = d, .n_tokens = n_tokens, .scale = scale
    };
    if (!tp) {
        ternary_matmul_packed_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, ternary_matmul_packed_batch_task, &args, d);
}
