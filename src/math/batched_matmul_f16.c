#include "math/batched_matmul.h"
#include <stdint.h>
#include <string.h>

/**
 * F16 dense batched matmul. See include/math/batched_matmul.h for the
 * layout convention and the bandwidth-reuse rationale.
 */

/* Mirrors matmul_f16.c's own static f16_to_f32_scalar() -- this codebase's
 * established convention is a small static copy per matmul kernel TU
 * (see matmul_q4k.c/matmul_q8_0.c/matmul_q5_1.c/etc.), not a shared header. */
static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

typedef struct {
    float        *out;
    const float  *x;
    const tn_u16 *w;
    int n, d, n_tokens;
} F16BatchArgs;

static void matmul_f16_batch_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    F16BatchArgs *a = (F16BatchArgs *)arg;

    for (int i = start; i < end; i++) {
        const tn_u16 *row = a->w + (size_t)i * a->n;
        /* `row` (2 bytes/element) is read from RAM once here and stays
         * resident in L1/L2 across the n_tokens loop below -- same
         * bandwidth-reuse reasoning as the ternary kernel in
         * batched_matmul.c. */
        for (int k = 0; k < a->n_tokens; k++) {
            const float *xk = a->x + (size_t)k * a->n;
            float val = 0.0f;
            for (int j = 0; j < a->n; j++) {
                val += f16_to_f32_scalar(row[j]) * xk[j];
            }
            a->out[(size_t)k * a->d + i] = val;
        }
    }
}

void tn_matmul_f16_batch(float *out, const float *x, const tn_u16 *w,
                          int n, int d, int n_tokens, ThreadPool *tp) {
    F16BatchArgs args = { .out = out, .x = x, .w = w, .n = n, .d = d, .n_tokens = n_tokens };
    if (!tp) {
        matmul_f16_batch_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_f16_batch_task, &args, d);
}
