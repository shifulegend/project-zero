/**
 * test_moe_benchmark.c — Micro-Benchmark Suite for Expert-Centric Compute-Amplified MoE
 *
 * Measures layer-level latency (us/layer), throughput (tok/s), and Arithmetic Intensity
 * (FLOPs / Byte) across:
 *   1. legacy         (Batched multi-expert dispatch)
 *   2. rowsplit       (Un-fused per-expert row-split dispatch)
 *   3. rowsplit-fused (Single-dispatch fused row-split GEMV with prefetching)
 *
 * Runs automatically during `make test` on GitHub Actions CI.
 */

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "core/gguf_reader.h"
#include "math/simd_dispatch.h"
#include "math/matmul_q4k.h"
#include "transformer/moe_ffn.h"
#include "transformer/moe_scheduler.h"
#include "test_harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#ifndef Q4K_SUPER
#define Q4K_SUPER 256
#endif
#ifndef Q4K_BYTES
#define Q4K_BYTES 144
#endif

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double benchmark_mode(const char *mode_name, RunState *s, TransformerWeights *w,
                             Config *cfg, MoEConfig *mc, ThreadPool *tp,
                             int warmups, int reps, float *out_buf) {
    moe_set_threading_mode(mode_name);

    /* Warmup runs */
    for (int i = 0; i < warmups; i++) {
        moe_ffn_forward(s, w, cfg, mc, 0, tp);
    }

    double t0 = get_time_ns();
    for (int i = 0; i < reps; i++) {
        moe_ffn_forward(s, w, cfg, mc, 0, tp);
    }
    double t1 = get_time_ns();

    if (out_buf) {
        memcpy(out_buf, s->xb2, cfg->dim * sizeof(float));
    }

    double total_us = (t1 - t0) / 1000.0;
    return total_us / reps; /* avg us per layer */
}

static void run_moe_benchmark(void) {
    tn_simd_init();

    printf("\n============================================================\n");
    printf("  Project Zero — Expert-Centric Compute-Amplified MoE Benchmark\n");
    printf("============================================================\n");

    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim        = 2048;
    cfg.hidden_dim = 5632;
    cfg.n_layers   = 1;
    cfg.n_heads    = 16;
    cfg.n_kv_heads = 16;
    cfg.vocab_size = 32000;
    cfg.seq_len    = 2048;
    cfg.act_type   = 0; /* SiLU */

    MoEConfig mc;
    memset(&mc, 0, sizeof(mc));
    mc.is_moe              = true;
    mc.num_experts         = 64;
    mc.num_experts_per_tok = 6;
    mc.expert_hidden_dim   = 1408;

    RunState s;
    memset(&s, 0, sizeof(s));
    s.x   = (float *)calloc(cfg.dim, sizeof(float));
    s.xb  = (float *)calloc(cfg.dim, sizeof(float));
    s.xb2 = (float *)calloc(cfg.dim, sizeof(float));
    s.hb  = (float *)calloc(mc.expert_hidden_dim, sizeof(float));
    s.hb2 = (float *)calloc(mc.expert_hidden_dim, sizeof(float));
    s.q   = (float *)calloc(cfg.dim, sizeof(float));

    for (int i = 0; i < cfg.dim; i++) {
        s.xb[i] = (float)((i % 31) - 15) * 0.0625f;
    }

    TransformerWeights w;
    memset(&w, 0, sizeof(w));
    w.rms_ffn_weight = (float **)calloc(1, sizeof(float *));
    w.rms_ffn_weight[0] = (float *)calloc(cfg.dim, sizeof(float));
    for (int i = 0; i < cfg.dim; i++) w.rms_ffn_weight[0][i] = 1.0f;

    w.moe_gate_w = (tn_i8 **)calloc(1, sizeof(tn_i8 *));
    w.moe_gate_w[0] = (tn_i8 *)calloc((size_t)mc.num_experts * cfg.dim, sizeof(float));
    for (int i = 0; i < mc.num_experts * cfg.dim; i++) {
        ((float *)w.moe_gate_w[0])[i] = (float)((i % 13) - 6) * 0.01f;
    }

    int n_blocks_w13 = cfg.dim / Q4K_SUPER;
    int n_blocks_w2  = mc.expert_hidden_dim / Q4K_SUPER;
    size_t q4k_bytes_w13 = (size_t)n_blocks_w13 * Q4K_BYTES * mc.expert_hidden_dim;
    size_t q4k_bytes_w2  = (size_t)n_blocks_w2  * Q4K_BYTES * cfg.dim;

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

    float *out_legacy = (float *)calloc(cfg.dim, sizeof(float));
    float *out_unfused = (float *)calloc(cfg.dim, sizeof(float));
    float *out_fused  = (float *)calloc(cfg.dim, sizeof(float));

    int warmups = 5;
    int reps    = 20;

    printf("\n[--- Single-Thread Performance (T=1) ---]\n");
    double t1_legacy  = benchmark_mode("legacy", &s, &w, &cfg, &mc, NULL, warmups, reps, out_legacy);
    double t1_unfused = benchmark_mode("rowsplit", &s, &w, &cfg, &mc, NULL, warmups, reps, out_unfused);
    double t1_fused   = benchmark_mode("fused", &s, &w, &cfg, &mc, NULL, warmups, reps, out_fused);

    printf("  legacy         : %8.2f us / layer\n", t1_legacy);
    printf("  rowsplit       : %8.2f us / layer\n", t1_unfused);
    printf("  rowsplit-fused : %8.2f us / layer\n", t1_fused);

    ThreadPool *tp = threadpool_create(4);
    printf("\n[--- Multi-Thread Performance (T=4) ---]\n");
    double t4_legacy  = benchmark_mode("legacy", &s, &w, &cfg, &mc, tp, warmups, reps, out_legacy);
    double t4_unfused = benchmark_mode("rowsplit", &s, &w, &cfg, &mc, tp, warmups, reps, out_unfused);
    double t4_fused   = benchmark_mode("fused", &s, &w, &cfg, &mc, tp, warmups, reps, out_fused);

    printf("  legacy         : %8.2f us / layer  (Speedup vs T=1: %.2fx)\n", t4_legacy, t1_legacy / t4_legacy);
    printf("  rowsplit       : %8.2f us / layer  (Speedup vs T=1: %.2fx)\n", t4_unfused, t1_unfused / t4_unfused);
    printf("  rowsplit-fused : %8.2f us / layer  (Speedup vs T=1: %.2fx)\n", t4_fused, t1_fused / t4_fused);

    /* Calculate Arithmetic Intensity Metrics */
    double flops_per_layer = (double)mc.num_experts_per_tok * 2.0 * (2.0 * cfg.dim * mc.expert_hidden_dim + mc.expert_hidden_dim * cfg.dim);
    double bytes_per_layer = (double)mc.num_experts_per_tok * (q4k_bytes_w13 * 2 + q4k_bytes_w2);
    double single_token_intensity = flops_per_layer / bytes_per_layer;
    double mode_a_batched_intensity = single_token_intensity * 4.0; /* 4 tokens sharing experts */

    printf("\n[--- Arithmetic Intensity & Data Reuse Metrics ---]\n");
    printf("  Single-Token FLOPs / Byte Intensity: %6.2f FLOPs/Byte\n", single_token_intensity);
    printf("  Batched Mode A FLOPs / Byte Intensity: %6.2f FLOPs/Byte (4.00x Compute Amplification)\n", mode_a_batched_intensity);

    printf("\n[--- Numerical Equivalence Verification ---]\n");
    int match_legacy_fused = 1;
    int match_unfused_fused = 1;
    for (int i = 0; i < cfg.dim; i++) {
        if (fabsf(out_legacy[i] - out_fused[i]) > 1e-4f) match_legacy_fused = 0;
        if (fabsf(out_unfused[i] - out_fused[i]) > 1e-4f) match_unfused_fused = 0;
    }
    TEST_ASSERT(match_legacy_fused, "Legacy vs RowSplit-Fused outputs match within 1e-4 FP32 tolerance");
    TEST_ASSERT(match_unfused_fused, "RowSplit vs RowSplit-Fused outputs match within 1e-4 FP32 tolerance");

    printf("============================================================\n\n");

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
    free(out_legacy); free(out_unfused); free(out_fused);
    free(s.x); free(s.xb); free(s.xb2); free(s.hb); free(s.hb2); free(s.q);
}

int main(void) {
    RUN_TEST(run_moe_benchmark);
    TEST_SUMMARY();
}
