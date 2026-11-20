/*
 * test_qwen35_attention.c -- Qwen3.5 hybrid attention (one Gated-DeltaNet layer and one full-
 * attention layer) on a tiny synthetic Q2_0 model, no GGUF needed.
 *   1. a forward over several positions is deterministic and changes the hidden state;
 *   2. two threads on separate RunStates reproduce the single-thread result exactly (the layer
 *      functions used to keep their scratch in function-local statics shared by every caller; under
 *      make test-tsan the old code is reported as a race).
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "kv_cache/sliding_window.h"
#include "math/simd_dispatch.h"
#include "runtime/run_state.h"
#include "transformer/qwen35_attention.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

#define DIM 128
#define LAYERS 2
#define HEADS 2
#define KV_HEADS 1
#define HEAD_DIM 64
#define ROPE_DIM 32
#define CONV_K 4
#define GROUPS 2
#define STATE 32
#define VHEADS 4
#define INNER (VHEADS * STATE)
#define KEY_DIM (GROUPS * STATE)
#define CONV_DIM (2 * KEY_DIM + INNER)
#define SEQ 16
#define STEPS 6

typedef struct {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    void *owned[64];
    int n_owned;
} Model;

static uint32_t g_seed = 12345;

static uint8_t *q2_0(Model *m, int rows, int cols) { /* rows x cols Q2_0, 34 bytes per 128 */
    size_t n   = (size_t)rows * (size_t)(cols / 128) * 34;
    uint8_t *p = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        g_seed = g_seed * 1664525u + 1013904223u;
        p[i]   = (uint8_t)(g_seed >> 24);
    }
    for (size_t b = 0; b < n / 34; b++) { /* fp16 scale 2^-6 */
        p[b * 34 + 0] = 0x00;
        p[b * 34 + 1] = 0x24;
    }
    m->owned[m->n_owned++] = p;
    return p;
}

static float *vec(Model *m, int n, float base, float spread) {
    float *p = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        g_seed = g_seed * 1664525u + 1013904223u;
        p[i]   = base + spread * (((int)(g_seed >> 24) - 128) / 128.0f);
    }
    m->owned[m->n_owned++] = p;
    return p;
}

static void build(Model *m) {
    memset(m, 0, sizeof(*m));
    m->cfg.dim                    = DIM;
    m->cfg.hidden_dim             = DIM;
    m->cfg.n_layers               = LAYERS;
    m->cfg.n_heads                = HEADS;
    m->cfg.n_kv_heads             = KV_HEADS;
    m->cfg.vocab_size             = 8;
    m->cfg.seq_len                = SEQ;
    m->cfg.rms_norm_eps           = 1e-5f;
    m->cfg.rope_theta             = 10000.0f;
    m->mc.has_linear_attn         = 1;
    m->mc.full_attention_interval = 2; /* layer 0 linear, layer 1 full attention */
    m->mc.attn_head_dim           = HEAD_DIM;
    m->mc.attn_rope_dim           = ROPE_DIM;
    m->mc.ssm_conv_kernel         = CONV_K;
    m->mc.ssm_group_count         = GROUPS;
    m->mc.ssm_inner_size          = INNER;
    m->mc.ssm_state_size          = STATE;
    m->mc.ssm_time_step_rank      = VHEADS;
    weights_alloc_pointers(&m->w, &m->cfg);
    TransformerWeights *w = &m->w;
    for (int l = 0; l < LAYERS; l++) w->attn.norm[l] = vec(m, DIM, 1.0f, 0.1f);
    /* layer 0: linear attention */
    w->q35.ssm_qkv[0]     = (tn_i8 *)q2_0(m, CONV_DIM, DIM);
    w->q35.ssm_gate[0]    = (tn_i8 *)q2_0(m, INNER, DIM);
    w->q35.ssm_beta[0]    = (tn_i8 *)q2_0(m, VHEADS, DIM);
    w->q35.ssm_alpha[0]   = (tn_i8 *)q2_0(m, VHEADS, DIM);
    w->q35.ssm_out[0]     = (tn_i8 *)q2_0(m, DIM, INNER);
    w->q35.ssm_conv1d[0]  = vec(m, CONV_DIM * CONV_K, 0.0f, 0.5f);
    w->q35.ssm_dt_bias[0] = vec(m, VHEADS, 0.0f, 0.5f);
    w->q35.ssm_a[0]       = vec(m, VHEADS, -0.5f, 0.2f);
    w->q35.ssm_norm[0]    = vec(m, STATE, 1.0f, 0.1f);
    /* layer 1: full attention (q is doubled for the output gate) */
    w->attn.wq[1]         = (tn_i8 *)q2_0(m, 2 * HEADS * HEAD_DIM, DIM);
    w->attn.wk[1]         = (tn_i8 *)q2_0(m, KV_HEADS * HEAD_DIM, DIM);
    w->attn.wv[1]         = (tn_i8 *)q2_0(m, KV_HEADS * HEAD_DIM, DIM);
    w->attn.wo[1]         = (tn_i8 *)q2_0(m, DIM, HEADS * HEAD_DIM);
    w->q35.attn_q_norm[1] = vec(m, HEAD_DIM, 1.0f, 0.1f);
    w->q35.attn_k_norm[1] = vec(m, HEAD_DIM, 1.0f, 0.1f);
}

static void destroy(Model *m) {
    for (int i = 0; i < m->n_owned; i++) free(m->owned[i]);
    /* the per-layer arrays hold borrowed pointers: clear them so weights_free_pointers only frees the arrays */
    for (int l = 0; l < LAYERS; l++) {
        m->w.attn.norm[l]       = NULL;
        m->w.q35.ssm_qkv[l]     = NULL;
        m->w.q35.ssm_gate[l]    = NULL;
        m->w.q35.ssm_beta[l]    = NULL;
        m->w.q35.ssm_alpha[l]   = NULL;
        m->w.q35.ssm_out[l]     = NULL;
        m->w.q35.ssm_conv1d[l]  = NULL;
        m->w.q35.ssm_dt_bias[l] = NULL;
        m->w.q35.ssm_a[l]       = NULL;
        m->w.q35.ssm_norm[l]    = NULL;
        m->w.attn.wq[l] = m->w.attn.wk[l] = m->w.attn.wv[l] = m->w.attn.wo[l] = NULL;
        m->w.q35.attn_q_norm[l] = m->w.q35.attn_k_norm[l] = NULL;
    }
    weights_free_pointers(&m->w);
}

/* Runs STEPS positions through both layers on a fresh RunState; out gets x after every step. */
static int run(const Model *m, float out[STEPS][DIM]) {
    RunState s;
    if (run_state_alloc_ex(&s, &m->cfg, SEQ, true) != TN_OK) return -1;
    if (q35_run_state_alloc(&s, &m->cfg, &m->mc, SEQ) != TN_OK) {
        run_state_free(&s);
        return -1;
    }
    sw_init(&s.sw, SEQ, 0);
    for (int pos = 0; pos < STEPS; pos++) {
        for (int i = 0; i < DIM; i++) s.x[i] = 0.05f * (float)(((i + 3 * pos) * 7) % 23) - 0.5f;
        for (int l = 0; l < LAYERS; l++)
            qwen35_attention_forward(&s, &m->w, &m->cfg, &m->mc, l, pos, NULL);
        memcpy(out[pos], s.x, DIM * sizeof(float));
    }
    q35_run_state_free(&s, &m->cfg, &m->mc);
    run_state_free(&s);
    return 0;
}

typedef struct {
    const Model *m;
    float out[STEPS][DIM];
    int ok;
} Job;

static void *job_fn(void *arg) {
    Job *j = (Job *)arg;
    j->ok  = 1;
    float first[STEPS][DIM];
    for (int rep = 0; rep < 30; rep++) {
        if (run(j->m, j->out) != 0) {
            j->ok = 0;
            return NULL;
        }
        if (rep == 0)
            memcpy(first, j->out, sizeof(first));
        else if (memcmp(first, j->out, sizeof(first)) != 0)
            j->ok = 0;
    }
    return NULL;
}

int main(void) {
    tn_simd_init();
    Model m;
    build(&m);

    float ref[STEPS][DIM], again[STEPS][DIM];
    CHECK(run(&m, ref) == 0 && run(&m, again) == 0, "single-thread runs");
    CHECK(memcmp(ref, again, sizeof(ref)) == 0, "a forward pass is deterministic");
    double moved = 0, mag = 0;
    for (int i = 0; i < DIM; i++) {
        moved +=
            fabs(ref[STEPS - 1][i] - (0.05f * (float)(((i + 3 * (STEPS - 1)) * 7) % 23) - 0.5f));
        mag += fabs(ref[STEPS - 1][i]);
    }
    int finite = 1;
    for (int p = 0; p < STEPS; p++)
        for (int i = 0; i < DIM; i++) finite &= isfinite(ref[p][i]);
    CHECK(finite && moved > 1e-3 && mag > 1e-3,
          "the layers change the hidden state and stay finite");
    if (getenv("Q35_HASH")) {
        unsigned long long h   = 1469598103934665603ULL;
        const unsigned char *b = (const unsigned char *)ref;
        for (size_t i = 0; i < sizeof(ref); i++) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
        printf("Q35HASH %llx\n", h);
    }

    Job a = {&m, {{0}}, 0}, b = {&m, {{0}}, 0};
    pthread_t ta, tb;
    pthread_create(&ta, NULL, job_fn, &a);
    pthread_create(&tb, NULL, job_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    CHECK(a.ok && b.ok, "two concurrent threads complete and repeat themselves");
    CHECK(memcmp(ref, a.out, sizeof(ref)) == 0 && memcmp(ref, b.out, sizeof(ref)) == 0,
          "two threads on separate RunStates reproduce the single-thread result exactly");

    destroy(&m);
    printf("=== %s ===\n", g_fail ? "FAILED" : "qwen35 attention tests passed");
    return g_fail ? 1 : 0;
}
