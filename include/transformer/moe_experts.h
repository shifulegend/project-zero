#ifndef TN_MOE_EXPERTS_H
#define TN_MOE_EXPERTS_H

/*
 * moe_experts — fused routed + shared expert FFN for one token (2026-10-21).
 *
 * Replaces the per-layer sequence "gate batch → up batch → SiLU×up →
 * down batch → saxpy per expert → shared gate/up → shared down → saxpy"
 * (6 thread-pool dispatches and 8 F32 scratch round-trips) with exactly two
 * dispatches, modelled on llama.cpp's ggml_compute_forward_mul_mat_id:
 *
 *   phase 1  rows of every selected expert's gate AND up matrix (plus the
 *            shared expert's) in one dispatch; each thread computes its rows'
 *            gate and up dots against the activation quantized ONCE, then
 *            applies act(gate) * up in place.
 *   (main)   each expert's hidden vector is quantized once for its down type
 *            (Q8K for K-quants, Q8_0 for Q5_0/Q5_1/Q8_0 — the integer dots
 *            of q8_0_act.c instead of the old F32-activation kernels).
 *   phase 2  one dispatch over the dim output rows; each thread owns a
 *            contiguous row range and sums weight_e * dot(down_e[r], h_e)
 *            over all experts straight into out[r] (no per-expert scratch,
 *            no separate weighted-accumulate pass).
 *
 * Inactive experts cost nothing (only selected jobs are passed in).
 */

#include "math/matmul_q4k.h"
#include "threading/thread_pool.h"
#include <stdint.h>

typedef struct {
    const uint8_t *w_gate;   /* hdim rows x dim cols, type gate_type */
    const uint8_t *w_up;     /* hdim rows x dim cols, type up_type   */
    const uint8_t *w_down;   /* dim rows x hdim cols, type down_type */
    int   gate_type, up_type, down_type;   /* GGUFType values */
    int   hdim;
    float weight;            /* routing weight (1.0 for a shared expert) */
} TnMoEExpertJob;

/* 1 if a weight of GGUF type `type` with `n_inner` columns has a row-dot
 * kernel in this executor (Q4_K, Q6_K: n_inner % 256; Q5_0, Q5_1, Q8_0:
 * n_inner % 32). */
int moe_experts_type_ok(int type, int n_inner);

/* 1 if every matrix of `job` is supported for model width `dim`. */
int moe_experts_job_ok(const TnMoEExpertJob *job, int dim);

/*
 * out[0..dim) += sum_j jobs[j].weight * down_j( act(gate_j x) * (up_j x) ).
 * act_type: 0 = SiLU, 1 = ReLU² (Config.act_type). All jobs must satisfy
 * moe_experts_job_ok(). Returns 0, or -1 on allocation failure (out is then
 * unchanged and the caller should fall back to the legacy path).
 * Not re-entrant (module scratch), like moe_ffn_forward().
 */
int moe_experts_forward(float *out, const float *x,
                        const TnMoEExpertJob *jobs, int n_jobs,
                        int dim, int act_type, ThreadPool *tp);

/* Release the module scratch (optional; called at model unload). */
void moe_experts_free(void);

#endif /* TN_MOE_EXPERTS_H */
