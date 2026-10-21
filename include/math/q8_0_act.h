#ifndef TN_Q8_0_ACT_H
#define TN_Q8_0_ACT_H

/*
 * Q8_0 activation blocks + integer row dots for the 32-element legacy quant
 * formats (Q5_0, Q5_1, Q8_0) — 2026-10-21.
 *
 * Why: the F32-activation kernels in matmul_q5_0.c / matmul_q5_1.c /
 * matmul_q8_0.c decode every weight to float and run FMAs, which made the
 * DeepSeek-V2-Lite routed-expert down projection (Q5_0) compute-bound at
 * ~4 GB/s per core. llama.cpp quantizes the activation once per expert to
 * Q8_0 and runs an integer dot (ggml_vec_dot_q5_0_q8_0 / _q5_1_q8_1 /
 * _q8_0_q8_0); this is the same technique. The activation is quantized ONCE
 * and reused for every weight row, so its cost is negligible.
 *
 * Numerics: the dot of a weight row with a Q8_0-quantized activation differs
 * from the F32-activation dot by the activation rounding (~0.4 % per element,
 * random sign) — the same approximation llama.cpp makes on every one of these
 * matmuls, so outputs move closer to the reference engine, not further.
 */

#include "core/platform.h"
#include "threading/thread_pool.h"
#include <stdint.h>

#define TN_Q8_0_BLOCK 32

/* One 32-element activation block. s = d * sum(qs) feeds the Q5_1 min term
 * (and the unsigned-offset trick of the Q5_0 VNNI path). 40 bytes. */
typedef struct {
    float  d;
    float  s;
    int8_t qs[TN_Q8_0_BLOCK];
} TnQ8_0ActBlock;

/* Quantize n_blocks * 32 floats (round to nearest, d = amax / 127). */
void tn_quantize_q8_0_act(TnQ8_0ActBlock *out, const float *x, int n_blocks);

/* Row dots: row = n_blocks weight blocks of the named GGUF type.
 * vnni != 0 selects the AVX-512 VNNI (256-bit, EVEX) path when it was
 * compiled in; callers pass tn_q8_0_act_vnni_ok() evaluated on the main
 * thread (the runtime-verified CPU flag — see cpu_features.c). */
float tn_dot_q5_0_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int n_blocks, int vnni);
float tn_dot_q5_1_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int n_blocks, int vnni);
float tn_dot_q8_0_q8_0(const uint8_t *row, const TnQ8_0ActBlock *a, int n_blocks, int vnni);

/* 1 when the VNNI paths above are compiled in AND the CPU verified them. */
int tn_q8_0_act_vnni_ok(void);

/* out[d] = W x for a d x n weight of GGUF type gguf_type (GGUF_TYPE_Q5_0,
 * GGUF_TYPE_Q5_1 or GGUF_TYPE_Q8_0; n % 32 == 0). x is quantized to Q8_0
 * once, then rows are split across the pool. Main-thread only (module
 * scratch). Returns 0, or -1 for an unsupported type / OOM (out untouched). */
int parallel_matmul_q8a(float *out, const float *x, const uint8_t *w,
                        int gguf_type, int n, int d, ThreadPool *tp);

#endif /* TN_Q8_0_ACT_H */
