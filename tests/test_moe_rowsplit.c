/**
 * test_moe_rowsplit.c — Unit tests for MoE Address-Sorted Sequential Row-Split GEMV
 */

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "core/gguf_reader.h"
#include "math/simd_dispatch.h"
#include "math/matmul_q4k.h"
#include "transformer/moe_ffn.h"
#include "test_harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef Q4K_SUPER
#define Q4K_SUPER 256
#endif
#ifndef Q4K_BYTES
#define Q4K_BYTES 144
#endif

/* ================================================================
 * TEST 1: Expert Address Sorting
 * ================================================================ */

static void test_expert_address_sorting(void) {
    int experts[5]   = { 42, 7, 100, 3, 15 };
    float scores[5]  = { 0.1f, 0.4f, 0.05f, 0.3f, 0.15f };
    int expected_e[5] = { 3, 7, 15, 42, 100 };
    float expected_s[5] = { 0.3f, 0.4f, 0.15f, 0.1f, 0.05f };

    moe_sort_selected_experts(experts, scores, 5);

    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_EQ(experts[i], expected_e[i], "Expert ID sorted in ascending order");
        TEST_ASSERT_FLOAT_EQ(scores[i], expected_s[i], 1e-6f, "Score matched with sorted expert ID");
    }
}

/* ================================================================
 * TEST 2: Threading Mode Toggle
 * ================================================================ */

static void test_moe_threading_mode_switch(void) {
    moe_set_threading_mode("legacy");
    TEST_ASSERT_EQ(moe_get_threading_mode(), TN_MOE_THREADING_LEGACY, "Mode set to legacy");

    moe_set_threading_mode("rowsplit");
    TEST_ASSERT_EQ(moe_get_threading_mode(), TN_MOE_THREADING_ROWSPLIT, "Mode set to rowsplit");
}

/* ================================================================
 * TEST 3: Numerical Equivalence (Legacy vs RowSplit)
 * ================================================================ */

static void test_moe_rowsplit_numerical_equivalence(void) {
    tn_simd_init();

    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim        = 256;
    cfg.hidden_dim = 512;
    cfg.n_layers   = 1;
    cfg.n_heads    = 4;
    cfg.n_kv_heads = 4;
    cfg.vocab_size = 100;
    cfg.seq_len    = 64;

    MoEConfig mc;
    memset(&mc, 0, sizeof(mc));
    mc.is_moe              = true;
    mc.num_experts         = 8;
    mc.num_experts_per_tok = 4;
    mc.expert_hidden_dim   = 512;

    RunState s;
    memset(&s, 0, sizeof(s));
    s.x   = (float *)calloc(cfg.dim, sizeof(float));
    s.xb  = (float *)calloc(cfg.dim, sizeof(float));
    s.xb2 = (float *)calloc(cfg.dim, sizeof(float));
    s.hb  = (float *)calloc(mc.expert_hidden_dim, sizeof(float));
    s.hb2 = (float *)calloc(mc.expert_hidden_dim, sizeof(float));
    s.q   = (float *)calloc(cfg.dim, sizeof(float));

    float *out_legacy   = (float *)calloc(cfg.dim, sizeof(float));
    float *out_rowsplit = (float *)calloc(cfg.dim, sizeof(float));

    /* Populate input vector x */
    for (int i = 0; i < cfg.dim; i++) {
        s.x[i] = (float)((i % 17) - 8) * 0.125f;
    }

    TransformerWeights w;
    memset(&w, 0, sizeof(w));
    w.rms_ffn_weight = (float **)calloc(1, sizeof(float *));
    w.rms_ffn_weight[0] = (float *)calloc(cfg.dim, sizeof(float));
    for (int i = 0; i < cfg.dim; i++) w.rms_ffn_weight[0][i] = 1.0f;

    w.moe_gate_w = (tn_i8 **)calloc(1, sizeof(tn_i8 *));
    w.moe_gate_w[0] = (tn_i8 *)calloc(mc.num_experts * cfg.dim, sizeof(float));
    for (int i = 0; i < mc.num_experts * cfg.dim; i++) {
        ((float *)w.moe_gate_w[0])[i] = (float)((i % 11) - 5) * 0.05f;
    }

    /* Allocate dummy Q4K weight structures */
    int n_blocks = cfg.dim / Q4K_SUPER;
    size_t q4k_bytes_w13 = (size_t)n_blocks * Q4K_BYTES * mc.expert_hidden_dim;
    size_t q4k_bytes_w2  = (size_t)(mc.expert_hidden_dim / Q4K_SUPER) * Q4K_BYTES * cfg.dim;

    w.has_expert_quant = true;
    w.expert_w13_quant_type = GGUF_TYPE_Q4_K;
    w.expert_w2_quant_type  = GGUF_TYPE_Q4_K;

    w.moe_w1 = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    w.moe_w2 = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    w.moe_w3 = (tn_i8 ***)calloc(1, sizeof(tn_i8 **));
    w.moe_w1[0] = (tn_i8 **)calloc(mc.num_experts, sizeof(tn_i8 *));
    w.moe_w2[0] = (tn_i8 **)calloc(mc.num_experts, sizeof(tn_i8 *));
    w.moe_w3[0] = (tn_i8 **)calloc(mc.num_experts, sizeof(tn_i8 *));

    for (int e = 0; e < mc.num_experts; e++) {
        w.moe_w1[0][e] = (tn_i8 *)calloc(q4k_bytes_w13, 1);
        w.moe_w2[0][e] = (tn_i8 *)calloc(q4k_bytes_w2, 1);
        w.moe_w3[0][e] = (tn_i8 *)calloc(q4k_bytes_w13, 1);
    }

    ThreadPool *tp = threadpool_create(4);

    /* Run Legacy mode */
    moe_set_threading_mode("legacy");
    moe_ffn_forward(&s, &w, &cfg, &mc, 0, tp);
    memcpy(out_legacy, s.x, cfg.dim * sizeof(float));

    /* Reset state x vector */
    for (int i = 0; i < cfg.dim; i++) {
        s.x[i] = (float)((i % 17) - 8) * 0.125f;
    }

    /* Run RowSplit mode */
    moe_set_threading_mode("rowsplit");
    moe_ffn_forward(&s, &w, &cfg, &mc, 0, tp);
    memcpy(out_rowsplit, s.x, cfg.dim * sizeof(float));

    /* Verify output vector match within FP32 tolerance */
    for (int i = 0; i < cfg.dim; i++) {
        TEST_ASSERT_FLOAT_EQ(out_legacy[i], out_rowsplit[i], 1e-4f,
                             "Legacy and RowSplit outputs match within FP32 tolerance");
    }

    /* Cleanup */
    if (tp) threadpool_destroy(tp);
    for (int e = 0; e < mc.num_experts; e++) {
        free(w.moe_w1[0][e]);
        free(w.moe_w2[0][e]);
        free(w.moe_w3[0][e]);
    }
    free(w.moe_w1[0]); free(w.moe_w2[0]); free(w.moe_w3[0]);
    free(w.moe_w1); free(w.moe_w2); free(w.moe_w3);
    free(w.moe_gate_w[0]); free(w.moe_gate_w);
    free(w.rms_ffn_weight[0]); free(w.rms_ffn_weight);
    free(out_legacy); free(out_rowsplit);
    free(s.x); free(s.xb); free(s.xb2); free(s.hb); free(s.hb2); free(s.q);
}

int main(void) {
    RUN_TEST(test_expert_address_sorting);
    RUN_TEST(test_moe_threading_mode_switch);
    RUN_TEST(test_moe_rowsplit_numerical_equivalence);
    TEST_SUMMARY();
}
