#include "math/batched_matmul.h"
#include "core/unpack.h"
#include "math/simd_dispatch.h"
#include "core/platform.h"
#include <stddef.h>
#include <stdlib.h>

/**
 * Ternary-packed batched matmul. See include/math/batched_matmul.h for the
 * layout convention and the bandwidth-reuse rationale.
 *
 * 2026-09-24 rewrite: see batched_matmul_f16.c's matching header comment for
 * the full root-cause writeup (docs/ai/mistakes.md has the real-model
 * measurement) -- the original v1 scalar-accumulator implementation
 * (`unpack_ternary()` called per-element per-token) paid the RAM-bandwidth
 * savings this kernel exists for, but at scalar per-element speed instead of
 * this project's real SIMD ternary dispatch tiers. Fixed the same way:
 * unpack each row's 2-bit ternary weights to {-1,0,1} ONCE per row (via the
 * existing SIMD shuffle-LUT unpacker where available -- unpack_ternary_block_avx2(),
 * core/unpack.h), then reuse the SIMD-dispatched tn_vec_dot() for each of the
 * n_tokens dot products against that decoded row.
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

    tn_i8 *row_i8 = (tn_i8 *)malloc((size_t)a->n * sizeof(tn_i8));
    float *row_f32 = (float *)malloc((size_t)a->n * sizeof(float));
    if (!row_i8 || !row_f32) {
        free(row_i8); free(row_f32);
        /* OOM fallback: original scalar-per-token path, still correct. */
        for (int i = start; i < end; i++) {
            const tn_u8 *row = a->packed_w + (size_t)i * row_bytes;
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
        return;
    }

    for (int i = start; i < end; i++) {
        const tn_u8 *row = a->packed_w + (size_t)i * row_bytes;
        /* `row` is small (ceil(n/4) bytes) -- it is read from RAM and
         * unpacked to {-1,0,1} floats ONCE here, then reused across the
         * n_tokens loop below (both the RAM read and the unpack cost are
         * amortized, not paid n_tokens times). */
#if TN_HAS_AVX2
        unpack_ternary_block_avx2(row_i8, row, a->n);
#else
        unpack_ternary_block(row_i8, row, a->n);
#endif
        for (int j = 0; j < a->n; j++) row_f32[j] = (float)row_i8[j];

        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            a->out[(size_t)k * a->d + i] = tn_vec_dot(row_f32, xk, a->n) * a->scale;
        }
    }
    free(row_i8);
    free(row_f32);
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
