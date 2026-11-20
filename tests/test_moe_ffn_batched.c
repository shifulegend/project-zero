/*
 * test_moe_ffn_batched.c -- moe_ffn_forward() on Q4_K experts through the legacy batched path
 * (PZ_MOE_FUSED=0), from two threads at once with different model widths. Each thread must get the
 * single-thread result exactly. The function's scratch (dequant buffer, batched gate/up/down
 * outputs, the Q8K copy of the activation) used to be file-scope statics shared by every caller.
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"
#include "core/gguf_reader.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "runtime/run_state.h"
#include "transformer/moe_ffn.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

#define NE 4
#define TOPK 2
#define MAX_DIM 512

typedef struct {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    float ref[MAX_DIM];
} Model;

static uint8_t *make_q4k(int rows, int cols, unsigned *seed) {
    size_t n   = (size_t)rows * (size_t)(cols / 256) * 144;
    uint8_t *w = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        w[i]  = (uint8_t)(*seed >> 24);
    }
    for (size_t b = 0; b < n / 144; b++) { /* d = dmin = fp16 2^-14: keeps silu out of saturation */
        w[b * 144 + 0] = 0x00;
        w[b * 144 + 1] = 0x04;
        w[b * 144 + 2] = 0x00;
        w[b * 144 + 3] = 0x04;
    }
    return w;
}

static void build(Model *m, int dim, unsigned seed) {
    memset(m, 0, sizeof(*m));
    m->cfg.dim                = dim;
    m->cfg.hidden_dim         = dim;
    m->cfg.n_layers           = 1;
    m->cfg.n_heads            = 2;
    m->cfg.n_kv_heads         = 2;
    m->cfg.vocab_size         = 8;
    m->cfg.seq_len            = 16;
    m->cfg.rms_norm_eps       = 1e-5f;
    m->mc.is_moe              = true;
    m->mc.num_experts         = NE;
    m->mc.num_experts_per_tok = TOPK;
    m->mc.expert_hidden_dim   = dim;
    weights_alloc_pointers(&m->w, &m->cfg);
    m->w.expert_quant.present  = true;
    m->w.expert_quant.w13_type = GGUF_TYPE_Q4_K;
    m->w.expert_quant.w2_type  = GGUF_TYPE_Q4_K;
    m->w.ffn.norm[0]           = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++) m->w.ffn.norm[0][i] = 1.0f;
    m->w.moe.gate_w = (tn_i8 **)calloc(1, sizeof(tn_i8 *));
    m->w.moe.w1     = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    m->w.moe.w2     = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    m->w.moe.w3     = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    float *gate     = (float *)malloc((size_t)NE * dim * sizeof(float));
    for (int i = 0; i < NE * dim; i++) {
        seed    = seed * 1664525u + 1013904223u;
        gate[i] = ((int)(seed >> 24) - 128) / 256.0f;
    }
    m->w.moe.gate_w[0] = (tn_i8 *)gate;
    m->w.moe.w1[0]     = (tn_i8 **)calloc(NE, sizeof(tn_i8 *));
    m->w.moe.w2[0]     = (tn_i8 **)calloc(NE, sizeof(tn_i8 *));
    m->w.moe.w3[0]     = (tn_i8 **)calloc(NE, sizeof(tn_i8 *));
    for (int e = 0; e < NE; e++) {
        m->w.moe.w1[0][e] = (tn_i8 *)make_q4k(dim, dim, &seed);
        m->w.moe.w3[0][e] = (tn_i8 *)make_q4k(dim, dim, &seed);
        m->w.moe.w2[0][e] = (tn_i8 *)make_q4k(dim, dim, &seed);
    }
}

static void destroy(Model *m) {
    free(m->w.ffn.norm[0]);
    m->w.ffn.norm[0] = NULL;
    free(m->w.moe.gate_w[0]);
    for (int e = 0; e < NE; e++) {
        free(m->w.moe.w1[0][e]);
        free(m->w.moe.w2[0][e]);
        free(m->w.moe.w3[0][e]);
    }
    free(m->w.moe.w1[0]);
    free(m->w.moe.w2[0]);
    free(m->w.moe.w3[0]);
    free(m->w.moe.gate_w);
    free(m->w.moe.w1);
    free(m->w.moe.w2);
    free(m->w.moe.w3);
    m->w.moe.gate_w = NULL;
    m->w.moe.w1 = m->w.moe.w2 = m->w.moe.w3 = NULL;
    weights_free_pointers(&m->w);
}

/* One forward pass on a fresh RunState; returns 0 on success. */
static int run_once(const Model *m, float *out) {
    RunState s;
    if (run_state_alloc(&s, &m->cfg, m->cfg.seq_len) != TN_OK) return -1;
    for (int i = 0; i < m->cfg.dim; i++) s.x[i] = 0.01f * (float)((i * 7) % 29) - 0.1f;
    moe_ffn_forward(&s, &m->w, &m->cfg, &m->mc, 0, NULL);
    memcpy(out, s.x, (size_t)m->cfg.dim * sizeof(float));
    run_state_free(&s);
    return 0;
}

typedef struct {
    const Model *a, *b;
    int ok;
} Job;

static void *run_fn(void *arg) {
    Job *j = (Job *)arg;
    j->ok  = 1;
    for (int rep = 0; rep < 100; rep++) {
        const Model *m = (rep & 1) ? j->b : j->a; /* alternate widths so the scratch regrows */
        float out[MAX_DIM];
        if (run_once(m, out) != 0 || memcmp(out, m->ref, (size_t)m->cfg.dim * sizeof(float)) != 0)
            j->ok = 0;
    }
    return NULL;
}

int main(void) {
    setenv("PZ_MOE_FUSED", "0", 1); /* force the legacy batched path (read once, at first use) */
    tn_simd_init();
    Model small, big;
    build(&small, 256, 11);
    build(&big, 512, 12);
    CHECK(run_once(&small, small.ref) == 0 && run_once(&big, big.ref) == 0,
          "single-thread reference runs");

    int changed = 0, finite = 1;
    for (int i = 0; i < small.cfg.dim; i++) {
        changed |= fabsf(small.ref[i] - (0.01f * (float)((i * 7) % 29) - 0.1f)) > 1e-6f;
        finite &= isfinite(small.ref[i]);
    }
    CHECK(changed && finite, "the reference output is finite and the experts changed it");

    Job a = {&small, &big, 0}, b = {&big, &small, 0}; /* opposite phase */
    pthread_t ta, tb;
    pthread_create(&ta, NULL, run_fn, &a);
    pthread_create(&tb, NULL, run_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    CHECK(a.ok && b.ok, "two threads reproduce the single-thread moe_ffn_forward output exactly");

    destroy(&small);
    destroy(&big);
    printf("=== %s ===\n", g_fail ? "FAILED" : "moe ffn batched tests passed");
    return g_fail ? 1 : 0;
}
