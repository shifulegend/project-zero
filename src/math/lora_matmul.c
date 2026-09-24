#include "math/lora_matmul.h"
#include "math/simd_dispatch.h"
#include <stdlib.h>

/*
 * lora_apply() computes out += scale * (B @ (A @ x)) in two small steps:
 *
 *   1. tmp[r] = dot(A_row_r, x)  for r in [0, rank)  -- A is [rank x n]
 *   2. out[i] += scale * dot(B_row_i, tmp)  for i in [0, d) -- B is [d x rank]
 *
 * Both dot products go through tn_vec_dot() (math/simd_dispatch.h), the
 * same SIMD-dispatched kernel the single-token dense/GQA attention path's
 * own dot products already use -- and the exact lesson learned fixing
 * Phase 18's scalar-batched-kernel regression (docs/ai/mistakes.md,
 * 2026-09-24): never hand-roll a scalar accumulator loop on this hardware.
 *
 * Step 1 (rank rows, each an O(n) dot product) is small by construction
 * (rank is typically 8-128) and run directly, not thread-dispatched --
 * dispatching a handful of rows across worker threads would add more
 * overhead than it saves. Step 2 (up to `d` rows, e.g. thousands for a
 * large FFN down-projection) is row-distributed across the thread pool,
 * mirroring src/math/parallel_matmul.c's own task-per-output-row pattern.
 */

typedef struct {
    float       *out;
    const float *tmp;
    const float *B;
    int          rank;
    float        scale;
} LoraBStepArgs;

static void lora_b_step_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    LoraBStepArgs *a = (LoraBStepArgs *)arg;
    for (int i = start; i < end; i++) {
        const float *b_row = a->B + (size_t)i * a->rank;
        a->out[i] += a->scale * tn_vec_dot(b_row, a->tmp, a->rank);
    }
}

void lora_apply(float *out, const float *x, const LoRAModule *mod,
                 int n, int d, ThreadPool *tp) {
    if (!mod || mod->rank == 0) return;

    float *tmp = (float *)malloc((size_t)mod->rank * sizeof(float));
    if (!tmp) return; /* OOM: skip the correction rather than crash generation */

    for (int r = 0; r < mod->rank; r++) {
        const float *a_row = mod->A + (size_t)r * n;
        tmp[r] = tn_vec_dot(a_row, x, n);
    }

    LoraBStepArgs args = { .out = out, .tmp = tmp, .B = mod->B,
                            .rank = mod->rank, .scale = mod->scale };
    if (!tp) {
        lora_b_step_task(&args, 0, 0, d);
    } else {
        threadpool_dispatch(tp, lora_b_step_task, &args, d);
    }

    free(tmp);
}
