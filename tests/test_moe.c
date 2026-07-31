/**
 * test_moe.c — Unit tests for Phase 17: Mixture of Experts Routing
 */

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/unpack.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "test_harness.h"
#include "transformer/ffn.h"
#include "transformer/forward.h"
#include "transformer/moe_ffn.h"
#include "transformer/moe_router.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ================================================================
 * Helpers
 * ================================================================ */

static void fill_f32(float *buf, int n, float val) {
    for (int i = 0; i < n; i++) buf[i] = val;
}

/**
 * Allocate a properly-packed ternary weight buffer of `count` weights.
 * Values cycle through {-1, 0, 1} deterministically based on seed.
 * Returns a calloc'd buffer of packed_bytes(count) bytes.
 */
static tn_u8 *alloc_packed_ternary(size_t count, int seed) {
    size_t pb = packed_bytes((int)count);
    tn_u8 *buf = (tn_u8 *)calloc(pb, 1);
    if (!buf) return NULL;
    for (size_t i = 0; i < count; i++) {
        int v = ((int)(i * 7 + (size_t)seed * 13) % 3) - 1;
        pack_ternary(buf, (int)i, (tn_i8)v);
    }
    return buf;
}

static Config tiny_config(void) {
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim        = 16;
    cfg.hidden_dim = 32;
    cfg.n_layers   = 1;
    cfg.n_heads    = 2;
    cfg.n_kv_heads = 2;
    cfg.vocab_size = 8;
    cfg.seq_len    = 16;
    return cfg;
}

/**
 * Allocate dense FFN weights for layer l in packed ternary format.
 * Requires w->layers_are_ternary == true and weights_alloc_pointers()
 * to have been called first.
 */
static void alloc_dense_layer(TransformerWeights *w, const Config *cfg, int l) {
    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    size_t gate  = (size_t)dim * hidden_dim;  /* number of weights */
    size_t down  = (size_t)hidden_dim * dim;

    w->rms_ffn_weight[l] = (float *)calloc(dim, sizeof(float));
    fill_f32(w->rms_ffn_weight[l], dim, 1.0f);

    /* packed_bytes() buffers — 4 ternary weights per byte */
    w->w1[l] = (tn_i8 *)alloc_packed_ternary(gate, l*100+5); w->s1[l] = 1.0f;
    w->w2[l] = (tn_i8 *)alloc_packed_ternary(down, l*100+6); w->s2[l] = 1.0f;
    w->w3[l] = (tn_i8 *)alloc_packed_ternary(gate, l*100+7); w->s3[l] = 1.0f;
}

/* ================================================================
 * TEST 1: Dense config
 * ================================================================ */

static void test_moe_config_dense(void) {
    MoEConfig mc;
    moe_config_init_dense(&mc);
    TEST_ASSERT(!mc.is_moe, "moe_config_init_dense sets is_moe=false");
    TEST_ASSERT(mc.num_experts == 0, "num_experts=0 for dense");
    TEST_ASSERT(mc.num_experts_per_tok == 0, "top_k=0 for dense");
    TEST_ASSERT(!moe_layer_is_moe(&mc, 0), "moe_layer_is_moe returns false for dense");
}

/* ================================================================
 * TEST 2: Router top-k selection
 * ================================================================ */

static void test_moe_router_topk(void) {
    tn_simd_init();
    int dim = 4, ne = 8, top_k = 2;
    /* gate_w is F32 [ne × dim] — matches moe_router_forward signature.
     * Experts 6 and 2 get positive weights (score > 0 after dot with all-ones x),
     * all others get negative weights, so top-2 must be {6, 2}. */
    float *gate_w = (float *)calloc((size_t)ne * dim, sizeof(float));
    for (int e = 0; e < ne; e++)
        for (int j = 0; j < dim; j++)
            gate_w[e * dim + j] = (float)((e == 6 || e == 2) ? 1.0f : -1.0f);

    float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float expert_scores[8];
    int   selected[2];
    float sel_scores[2];

    moe_router_forward(expert_scores, selected, sel_scores,
                       x, gate_w, dim, ne, top_k, NULL);

    int got6 = (selected[0] == 6 || selected[1] == 6);
    int got2 = (selected[0] == 2 || selected[1] == 2);
    TEST_ASSERT(got6, "Router selects expert 6");
    TEST_ASSERT(got2, "Router selects expert 2");
    free(gate_w);
}

/* ================================================================
 * TEST 3: Router scores sum to 1.0
 * ================================================================ */

static void test_moe_router_scores_sum_1(void) {
    tn_simd_init();
    int dim = 8, ne = 6, top_k = 3;
    /* gate_w is F32 [ne × dim] */
    float *gate_w = (float *)calloc((size_t)ne * dim, sizeof(float));
    for (int e = 0; e < ne; e++)
        for (int j = 0; j < dim; j++)
            gate_w[e * dim + j] = (float)((e % 3 == 0) ? 1.0f : ((e % 3 == 1) ? 0.0f : -1.0f));

    float x[8]; fill_f32(x, dim, 1.0f);
    float expert_scores[6];
    int   selected[3];
    float sel_scores[3];

    moe_router_forward(expert_scores, selected, sel_scores,
                       x, gate_w, dim, ne, top_k, NULL);

    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) sum += sel_scores[i];
    /* sel_scores are a subset of the full softmax over all ne experts.
     * top_k < ne, so the selected scores sum to < 1.0 (not renormalised).
     * Verify: each score is in (0, 1], sum is positive, and sum <= 1.0. */
    TEST_ASSERT(sum > 0.0f, "Scores sum > 0");
    TEST_ASSERT(sum <= 1.0f + 1e-5f, "Scores sum <= 1.0 (subset of softmax)");
    for (int i = 0; i < top_k; i++) {
        TEST_ASSERT(sel_scores[i] > 0.0f, "Each selected score is positive");
        TEST_ASSERT(sel_scores[i] <= 1.0f + 1e-5f, "Each selected score <= 1.0");
    }
    free(gate_w);
}

/* ================================================================
 * TEST 4: FFN output finite + Leak-free cleanup
 * ================================================================ */

static void test_moe_ffn_output_finite(void) {
    tn_simd_init();
    Config cfg = tiny_config();
    MoEConfig mc;
    mc.is_moe = true;
    mc.num_experts = 4;
    mc.num_experts_per_tok = 2;
    mc.expert_hidden_dim = cfg.hidden_dim;
    mc.first_k_dense_replace = 0;

    TransformerWeights w;
    memset(&w, 0, sizeof(w));
    weights_alloc_pointers(&w, &cfg);
    w.layers_are_ternary = true;   /* set after alloc_pointers (modular ordering) */
    alloc_dense_layer(&w, &cfg, 0);

    int nl = cfg.n_layers, ne = mc.num_experts, ehdim = mc.expert_hidden_dim;
    w.moe_gate_w = (tn_i8 **)calloc(nl, sizeof(tn_i8 *));
    w.moe_gate_s = (float *)calloc(nl, sizeof(float));
    w.moe_w1 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w.moe_w2 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w.moe_w3 = (tn_i8 ***)calloc(nl, sizeof(tn_i8 **));
    w.moe_s1 = (float **)calloc(nl, sizeof(float *));
    w.moe_s2 = (float **)calloc(nl, sizeof(float *));
    w.moe_s3 = (float **)calloc(nl, sizeof(float *));

    for (int l = 0; l < nl; l++) {
        /* moe_gate_w is stored as a tn_i8* pointer but holds raw F32 gate
         * weights (moe_ffn.c casts it to `const float *` before calling
         * moe_router_forward — see its comment "Gate weight is
         * w->moe_gate_w[layer] — F32 [num_experts × dim]"). Sizing this
         * allocation with sizeof(tn_i8) under-allocated by 4x, which ASan
         * only caught once make test started actually instrumenting
         * LIB_OBJS with sanitizers (see docs/ai/mistakes.md) — the router's
         * F32 SIMD load read past this buffer's true (undersized) end. */
        w.moe_gate_w[l] = (tn_i8 *)calloc((size_t)cfg.dim * ne, sizeof(float));
        w.moe_gate_s[l] = 1.0f;
        w.moe_w1[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w.moe_w2[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w.moe_w3[l] = (tn_i8 **)calloc(ne, sizeof(tn_i8 *));
        w.moe_s1[l] = (float *)calloc(ne, sizeof(float));
        w.moe_s2[l] = (float *)calloc(ne, sizeof(float));
        w.moe_s3[l] = (float *)calloc(ne, sizeof(float));
        for (int e = 0; e < ne; e++) {
            w.moe_w1[l][e] = (tn_i8 *)calloc((size_t)cfg.dim * ehdim, sizeof(tn_i8));
            w.moe_w2[l][e] = (tn_i8 *)calloc((size_t)ehdim * cfg.dim, sizeof(tn_i8));
            w.moe_w3[l][e] = (tn_i8 *)calloc((size_t)cfg.dim * ehdim, sizeof(tn_i8));
            w.moe_s1[l][e] = 1.0f; w.moe_s2[l][e] = 1.0f; w.moe_s3[l][e] = 1.0f;
        }
    }

    RunState s;
    run_state_alloc(&s, &cfg, cfg.seq_len);
    fill_f32(s.x, cfg.dim, 0.5f);
    moe_ffn_forward(&s, &w, &cfg, &mc, 0, NULL);
    for (int i = 0; i < cfg.dim; i++) TEST_ASSERT(!isnan(s.x[i]), "No NaN");
    run_state_free(&s);

    /* Clean up */
    for (int l = 0; l < nl; l++) {
        free(w.moe_gate_w[l]);
        for (int e = 0; e < ne; e++) {
            free(w.moe_w1[l][e]); free(w.moe_w2[l][e]); free(w.moe_w3[l][e]);
        }
        free(w.moe_w1[l]); free(w.moe_w2[l]); free(w.moe_w3[l]);
        free(w.moe_s1[l]); free(w.moe_s2[l]); free(w.moe_s3[l]);
        free(w.rms_ffn_weight[l]); free(w.w1[l]); free(w.w2[l]); free(w.w3[l]);
    }
    free(w.moe_gate_w); free(w.moe_gate_s);
    free(w.moe_w1); free(w.moe_w2); free(w.moe_w3);
    free(w.moe_s1); free(w.moe_s2); free(w.moe_s3);
    weights_free_pointers(&w);
}

/* ================================================================
 * TEST 5: Dense model regression
 * ================================================================ */

static void test_moe_dense_unchanged(void) {
    tn_simd_init();
    Config cfg = tiny_config();
    TransformerWeights w;
    memset(&w, 0, sizeof(w));
    /* layers_are_ternary MUST be set after weights_alloc_pointers —
     * that function no longer does a memset, so this ordering is safe. */
    weights_alloc_pointers(&w, &cfg);
    w.layers_are_ternary = true;
    alloc_dense_layer(&w, &cfg, 0);  /* allocates proper packed ternary */

    RunState s1, s2;
    run_state_alloc(&s1, &cfg, cfg.seq_len);
    run_state_alloc(&s2, &cfg, cfg.seq_len);
    fill_f32(s1.x, cfg.dim, 0.5f);
    memcpy(s2.x, s1.x, cfg.dim * sizeof(float));

    /* Case 1: mc == NULL  (dense path) */
    ffn_forward(&s1, &w, &cfg, NULL, 0, NULL);
    /* Case 2: mc->is_moe == false  (same dense path) */
    MoEConfig mc_dense; moe_config_init_dense(&mc_dense);
    ffn_forward(&s2, &w, &cfg, &mc_dense, 0, NULL);

    for (int i = 0; i < cfg.dim; i++) TEST_ASSERT_FLOAT_EQ(s1.x[i], s2.x[i], 1e-7f, "Paths match");

    run_state_free(&s1); run_state_free(&s2);
    free(w.rms_ffn_weight[0]); free(w.w1[0]); free(w.w2[0]); free(w.w3[0]);
    weights_free_pointers(&w);
}

int main(void) {
    RUN_TEST(test_moe_config_dense);
    RUN_TEST(test_moe_router_topk);
    RUN_TEST(test_moe_router_scores_sum_1);
    RUN_TEST(test_moe_ffn_output_finite);
    RUN_TEST(test_moe_dense_unchanged);
    TEST_SUMMARY();
}
