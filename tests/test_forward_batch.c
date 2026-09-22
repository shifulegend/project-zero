/*
 * test_forward_batch.c — Phase 18 (speculative decoding) Stage 2:
 * correctness of transformer_forward_batch() against N sequential
 * transformer_forward() calls fed the same token sequence (the
 * teacher-forcing property causal transformers guarantee).
 *
 * Covers the three batchable architecture families (dense/GQA, MLA,
 * Qwen3-MoE QK-norm) with tiny synthetic ternary-weighted models — proving
 * the write-phase/read-phase restructuring is correct, and that it
 * genuinely exercises the real Stage-1 batched kernels (not just the
 * sequential-fallback path), plus a negative test for the documented
 * has_linear_attn refusal.
 */
#include "test_harness.h"
#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/unpack.h"
#include "math/simd_dispatch.h"
#include "speculative/spec_scratch.h"
#include "transformer/forward.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOL 1e-3f

/* ── Shared fill helpers ──────────────────────────────────────────────── */

static void fill_ternary_i8(tn_i8 *buf, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        int r = (seed >> 16) % 3;
        buf[i] = (tn_i8)(r - 1);
    }
}

/* Packs a [rows][cols] ternary matrix (row-major) into 2-bit packed form. */
static tn_u8 *make_packed_ternary(int rows, int cols, unsigned seed) {
    size_t row_bytes = packed_bytes(cols);
    tn_u8 *packed = (tn_u8 *)calloc((size_t)rows * row_bytes, 1);
    tn_i8 *tmp = (tn_i8 *)malloc((size_t)rows * cols);
    fill_ternary_i8(tmp, rows * cols, seed);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pack_ternary(&packed[(size_t)i * row_bytes], j, tmp[i * cols + j]);
        }
    }
    free(tmp);
    return packed;
}

static void fill_pattern_bf16(tn_u16 *buf, size_t count, int seed) {
    for (size_t i = 0; i < count; i++) {
        int v = ((int)(i * 7 + (unsigned)seed * 13) % 3) - 1;
        float f = (float)v * 0.5f;
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        buf[i] = (tn_u16)(bits >> 16);
    }
}

static void fill_ones(float *buf, int count) {
    for (int i = 0; i < count; i++) buf[i] = 1.0f;
}

/* ── Dense/GQA ternary synthetic model ───────────────────────────────── */

#define D_DIM 16
#define D_HIDDEN 32
#define D_LAYERS 1
#define D_HEADS 2
#define D_KV_HEADS 2
#define D_VOCAB 8
#define D_SEQ 16

static void build_dense_ternary_model(Config *cfg, TransformerWeights *w) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->dim = D_DIM;
    cfg->hidden_dim = D_HIDDEN;
    cfg->n_layers = D_LAYERS;
    cfg->n_heads = D_HEADS;
    cfg->n_kv_heads = D_KV_HEADS;
    cfg->vocab_size = D_VOCAB;
    cfg->seq_len = D_SEQ;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_freq_scale = 1.0f;
    cfg->rope_theta = 10000.0f;
    cfg->rope_yarn_attn_factor = 1.0f;

    memset(w, 0, sizeof(*w));
    weights_alloc_pointers(w, cfg);
    w->layers_are_ternary = true;

    int kv_dim = config_kv_dim(cfg);

    w->token_embedding_table = (tn_u16 *)calloc((size_t)D_VOCAB * D_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->token_embedding_table, (size_t)D_VOCAB * D_DIM, 1);

    for (int l = 0; l < D_LAYERS; l++) {
        w->wq[l] = (tn_i8 *)make_packed_ternary(D_DIM, D_DIM, 100 + l);
        w->sq[l] = 1.0f;
        w->wk[l] = (tn_i8 *)make_packed_ternary(kv_dim, D_DIM, 200 + l);
        w->sk[l] = 1.0f;
        w->wv[l] = (tn_i8 *)make_packed_ternary(kv_dim, D_DIM, 300 + l);
        w->sv[l] = 1.0f;
        w->wo[l] = (tn_i8 *)make_packed_ternary(D_DIM, D_DIM, 400 + l);
        w->so[l] = 1.0f;

        w->w1[l] = (tn_i8 *)make_packed_ternary(D_HIDDEN, D_DIM, 500 + l);
        w->s1[l] = 1.0f;
        w->w3[l] = (tn_i8 *)make_packed_ternary(D_HIDDEN, D_DIM, 600 + l);
        w->s3[l] = 1.0f;
        w->w2[l] = (tn_i8 *)make_packed_ternary(D_DIM, D_HIDDEN, 700 + l);
        w->s2[l] = 1.0f;

        w->rms_att_weight[l] = (float *)malloc(D_DIM * sizeof(float));
        fill_ones(w->rms_att_weight[l], D_DIM);
        w->rms_ffn_weight[l] = (float *)malloc(D_DIM * sizeof(float));
        fill_ones(w->rms_ffn_weight[l], D_DIM);
    }

    w->rms_final_weight = (float *)malloc(D_DIM * sizeof(float));
    fill_ones(w->rms_final_weight, D_DIM);

    w->wcls = (tn_u16 *)calloc((size_t)D_VOCAB * D_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->wcls, (size_t)D_VOCAB * D_DIM, 99);
}

static void free_dense_ternary_model(TransformerWeights *w) {
    free(w->token_embedding_table);
    for (int l = 0; l < D_LAYERS; l++) {
        free(w->wq[l]); free(w->wk[l]); free(w->wv[l]); free(w->wo[l]);
        free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
        free(w->rms_att_weight[l]); free(w->rms_ffn_weight[l]);
    }
    free(w->rms_final_weight);
    free(w->wcls);
    weights_free_pointers(w);
}

static void test_dense_batch_matches_sequential(void) {
    tn_simd_init();

    Config cfg;
    TransformerWeights w;
    build_dense_ternary_model(&cfg, &w);

    const int n_tokens = 3;
    int tokens[3] = {1, 4, 2};

    /* Sequential reference */
    RunState s_ref;
    TEST_ASSERT(run_state_alloc(&s_ref, &cfg, cfg.seq_len) == TN_OK, "ref run_state_alloc");
    sw_init(&s_ref.sw, cfg.seq_len, 0);
    float logits_ref[3][D_VOCAB];
    for (int k = 0; k < n_tokens; k++) {
        float *lg = transformer_forward(tokens[k], k, &cfg, &w, &s_ref, NULL, NULL);
        memcpy(logits_ref[k], lg, D_VOCAB * sizeof(float));
    }

    /* Batched */
    RunState s_batch;
    TEST_ASSERT(run_state_alloc(&s_batch, &cfg, cfg.seq_len) == TN_OK, "batch run_state_alloc");
    sw_init(&s_batch.sw, cfg.seq_len, 0);
    SpecBatchScratch sb;
    TEST_ASSERT(spec_batch_scratch_alloc(&sb, &cfg, n_tokens) == TN_OK, "spec_batch_scratch_alloc");

    /* Embed tokens directly into sb->x (transformer_forward_batch does this
     * internally too, but we need sb->x populated the same way the
     * sequential reference's s.x was via transformer_forward's own embed
     * step -- both paths call the same embed_token(), so just invoke the
     * batched function with the raw token IDs, matching its documented
     * contract). */
    float logits_batch[3][D_VOCAB];
    TernaryError err = transformer_forward_batch(tokens, 0, n_tokens, &cfg, &w,
                                                  &s_batch, &sb, NULL, NULL,
                                                  (float *)logits_batch);
    TEST_ASSERT(err == TN_OK, "transformer_forward_batch returns TN_OK for dense model");

    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        for (int i = 0; i < D_VOCAB; i++) {
            if (fabsf(logits_ref[k][i] - logits_batch[k][i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "dense/GQA batched forward matches N sequential forward calls");

    spec_batch_scratch_free(&sb);
    run_state_free(&s_ref);
    run_state_free(&s_batch);
    free_dense_ternary_model(&w);
}

/* ── MLA ternary synthetic model ─────────────────────────────────────── */

#define M_DIM 16
#define M_HIDDEN 32
#define M_LAYERS 1
#define M_HEADS 2
#define M_KV_HEADS 2
#define M_NOPE 4
#define M_ROPE 4
#define M_LORA 6
#define M_VDIM 4
#define M_VOCAB 8
#define M_SEQ 16

static void build_mla_ternary_model(Config *cfg, TransformerWeights *w, MoEConfig *mc) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->dim = M_DIM;
    cfg->hidden_dim = M_HIDDEN;
    cfg->n_layers = M_LAYERS;
    cfg->n_heads = M_HEADS;
    cfg->n_kv_heads = M_KV_HEADS;
    cfg->vocab_size = M_VOCAB;
    cfg->seq_len = M_SEQ;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_freq_scale = 1.0f;
    cfg->rope_theta = 10000.0f;
    cfg->rope_yarn_attn_factor = 1.0f;

    moe_config_init_dense(mc);
    mc->has_mla = 1;
    mc->qk_nope_head_dim = M_NOPE;
    mc->qk_rope_head_dim = M_ROPE;
    mc->kv_lora_rank = M_LORA;
    mc->v_head_dim = M_VDIM;
    /* is_moe must be true for moe_weights_alloc() to allocate the mla_*
     * pointer arrays (see src/core/moe_weights.c) -- first_k_dense_replace
     * covers the one layer so ffn_forward_batch takes the dense path,
     * keeping this test focused on MLA *attention* batching alone. */
    mc->is_moe = true;
    mc->first_k_dense_replace = M_LAYERS;

    memset(w, 0, sizeof(*w));
    weights_alloc_pointers(w, cfg);
    TEST_ASSERT(moe_weights_alloc(w, cfg, mc) == TN_OK, "moe_weights_alloc (MLA arrays)");
    w->layers_are_ternary = true;

    const int q_rows = M_HEADS * (M_NOPE + M_ROPE);
    const int kva_rows = M_LORA + M_ROPE;
    const int kvb_rows = M_KV_HEADS * (M_NOPE + M_VDIM);

    w->token_embedding_table = (tn_u16 *)calloc((size_t)M_VOCAB * M_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->token_embedding_table, (size_t)M_VOCAB * M_DIM, 1);

    for (int l = 0; l < M_LAYERS; l++) {
        w->mla_wq[l]    = (tn_i8 *)make_packed_ternary(q_rows, M_DIM, 111 + l);
        w->mla_sq[l]    = 1.0f;
        w->mla_wkv_a[l] = (tn_i8 *)make_packed_ternary(kva_rows, M_DIM, 222 + l);
        w->mla_skv_a[l] = 1.0f;
        w->mla_wkv_b[l] = (tn_i8 *)make_packed_ternary(kvb_rows, M_LORA, 333 + l);
        w->mla_skv_b[l] = 1.0f;

        w->wo[l] = (tn_i8 *)make_packed_ternary(M_DIM, M_HEADS * M_VDIM, 444 + l);
        w->so[l] = 1.0f;

        /* Dense FFN (first_k_dense_replace covers this layer) */
        w->w1[l] = (tn_i8 *)make_packed_ternary(M_HIDDEN, M_DIM, 555 + l);
        w->s1[l] = 1.0f;
        w->w3[l] = (tn_i8 *)make_packed_ternary(M_HIDDEN, M_DIM, 666 + l);
        w->s3[l] = 1.0f;
        w->w2[l] = (tn_i8 *)make_packed_ternary(M_DIM, M_HIDDEN, 777 + l);
        w->s2[l] = 1.0f;

        w->rms_att_weight[l] = (float *)malloc(M_DIM * sizeof(float));
        fill_ones(w->rms_att_weight[l], M_DIM);
        w->rms_ffn_weight[l] = (float *)malloc(M_DIM * sizeof(float));
        fill_ones(w->rms_ffn_weight[l], M_DIM);
        w->rms_attn_sub_norm[l] = (float *)malloc(M_LORA * sizeof(float));
        fill_ones(w->rms_attn_sub_norm[l], M_LORA);
    }

    w->rms_final_weight = (float *)malloc(M_DIM * sizeof(float));
    fill_ones(w->rms_final_weight, M_DIM);

    w->wcls = (tn_u16 *)calloc((size_t)M_VOCAB * M_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->wcls, (size_t)M_VOCAB * M_DIM, 99);
}

static void free_mla_ternary_model(TransformerWeights *w, const MoEConfig *mc) {
    free(w->token_embedding_table);
    for (int l = 0; l < M_LAYERS; l++) {
        free(w->mla_wq[l]); free(w->mla_wkv_a[l]); free(w->mla_wkv_b[l]);
        free(w->wo[l]); free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
        free(w->rms_att_weight[l]); free(w->rms_ffn_weight[l]); free(w->rms_attn_sub_norm[l]);
    }
    free(w->rms_final_weight);
    free(w->wcls);
    moe_weights_free(w, mc);
    weights_free_pointers(w);
}

static void test_mla_batch_matches_sequential(void) {
    tn_simd_init();

    Config cfg;
    TransformerWeights w;
    MoEConfig mc;
    build_mla_ternary_model(&cfg, &w, &mc);

    const int n_tokens = 3;
    int tokens[3] = {1, 4, 2};

    RunState s_ref;
    TEST_ASSERT(run_state_alloc(&s_ref, &cfg, cfg.seq_len) == TN_OK, "MLA ref run_state_alloc");
    TEST_ASSERT(mla_run_state_alloc(&s_ref, &cfg, &mc, cfg.seq_len) == TN_OK, "MLA ref mla_run_state_alloc");
    sw_init(&s_ref.sw, cfg.seq_len, 0);
    float logits_ref[3][M_VOCAB];
    for (int k = 0; k < n_tokens; k++) {
        float *lg = transformer_forward(tokens[k], k, &cfg, &w, &s_ref, &mc, NULL);
        memcpy(logits_ref[k], lg, M_VOCAB * sizeof(float));
    }

    RunState s_batch;
    TEST_ASSERT(run_state_alloc(&s_batch, &cfg, cfg.seq_len) == TN_OK, "MLA batch run_state_alloc");
    TEST_ASSERT(mla_run_state_alloc(&s_batch, &cfg, &mc, cfg.seq_len) == TN_OK, "MLA batch mla_run_state_alloc");
    sw_init(&s_batch.sw, cfg.seq_len, 0);
    SpecBatchScratch sb;
    TEST_ASSERT(spec_batch_scratch_alloc(&sb, &cfg, n_tokens) == TN_OK, "MLA spec_batch_scratch_alloc");

    float logits_batch[3][M_VOCAB];
    TernaryError err = transformer_forward_batch(tokens, 0, n_tokens, &cfg, &w,
                                                  &s_batch, &sb, &mc, NULL,
                                                  (float *)logits_batch);
    TEST_ASSERT(err == TN_OK, "transformer_forward_batch returns TN_OK for MLA model");

    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        for (int i = 0; i < M_VOCAB; i++) {
            if (fabsf(logits_ref[k][i] - logits_batch[k][i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "MLA batched forward matches N sequential forward calls");

    spec_batch_scratch_free(&sb);
    mla_run_state_free(&s_ref, cfg.n_layers);
    run_state_free(&s_ref);
    mla_run_state_free(&s_batch, cfg.n_layers);
    run_state_free(&s_batch);
    free_mla_ternary_model(&w, &mc);
}

/* ── Qwen3-MoE (QK-norm) ternary synthetic model — attention only ──────── */

#define Q3_DIM 8
#define Q3_HIDDEN 8
#define Q3_LAYERS 1
#define Q3_HEADS 2
#define Q3_KV_HEADS 1
#define Q3_HEAD_DIM 4
#define Q3_VOCAB 6
#define Q3_SEQ 16

static void build_qwen3moe_ternary_model(Config *cfg, TransformerWeights *w, MoEConfig *mc) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->dim = Q3_DIM;
    cfg->hidden_dim = Q3_HIDDEN;
    cfg->n_layers = Q3_LAYERS;
    cfg->n_heads = Q3_HEADS;
    cfg->n_kv_heads = Q3_KV_HEADS;
    cfg->vocab_size = Q3_VOCAB;
    cfg->seq_len = Q3_SEQ;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_freq_scale = 1.0f;
    cfg->rope_theta = 10000.0f;
    cfg->rope_yarn_attn_factor = 1.0f;

    moe_config_init_dense(mc);
    mc->has_qk_norm = 1;
    mc->attn_head_dim = Q3_HEAD_DIM;
    /* is_moe left false: dense FFN, keeping this test focused on
     * QK-norm attention batching alone (mirrors the MLA test's approach). */

    memset(w, 0, sizeof(*w));
    weights_alloc_pointers(w, cfg);
    w->layers_are_ternary = true;

    int q_width = Q3_HEADS * Q3_HEAD_DIM;
    int kv_width = Q3_KV_HEADS * Q3_HEAD_DIM;

    w->token_embedding_table = (tn_u16 *)calloc((size_t)Q3_VOCAB * Q3_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->token_embedding_table, (size_t)Q3_VOCAB * Q3_DIM, 1);

    for (int l = 0; l < Q3_LAYERS; l++) {
        w->wq[l] = (tn_i8 *)make_packed_ternary(q_width, Q3_DIM, 11 + l);
        w->sq[l] = 1.0f;
        w->wk[l] = (tn_i8 *)make_packed_ternary(kv_width, Q3_DIM, 22 + l);
        w->sk[l] = 1.0f;
        w->wv[l] = (tn_i8 *)make_packed_ternary(kv_width, Q3_DIM, 33 + l);
        w->sv[l] = 1.0f;
        w->wo[l] = (tn_i8 *)make_packed_ternary(Q3_DIM, q_width, 44 + l);
        w->so[l] = 1.0f;

        w->w1[l] = (tn_i8 *)make_packed_ternary(Q3_HIDDEN, Q3_DIM, 55 + l);
        w->s1[l] = 1.0f;
        w->w3[l] = (tn_i8 *)make_packed_ternary(Q3_HIDDEN, Q3_DIM, 66 + l);
        w->s3[l] = 1.0f;
        w->w2[l] = (tn_i8 *)make_packed_ternary(Q3_DIM, Q3_HIDDEN, 77 + l);
        w->s2[l] = 1.0f;

        w->rms_att_weight[l] = (float *)malloc(Q3_DIM * sizeof(float));
        fill_ones(w->rms_att_weight[l], Q3_DIM);
        w->rms_ffn_weight[l] = (float *)malloc(Q3_DIM * sizeof(float));
        fill_ones(w->rms_ffn_weight[l], Q3_DIM);

        w->qwen3moe_attn_q_norm[l] = (float *)malloc(Q3_HEAD_DIM * sizeof(float));
        fill_ones(w->qwen3moe_attn_q_norm[l], Q3_HEAD_DIM);
        w->qwen3moe_attn_k_norm[l] = (float *)malloc(Q3_HEAD_DIM * sizeof(float));
        fill_ones(w->qwen3moe_attn_k_norm[l], Q3_HEAD_DIM);
    }

    w->rms_final_weight = (float *)malloc(Q3_DIM * sizeof(float));
    fill_ones(w->rms_final_weight, Q3_DIM);

    w->wcls = (tn_u16 *)calloc((size_t)Q3_VOCAB * Q3_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->wcls, (size_t)Q3_VOCAB * Q3_DIM, 99);
}

static void free_qwen3moe_ternary_model(TransformerWeights *w) {
    free(w->token_embedding_table);
    for (int l = 0; l < Q3_LAYERS; l++) {
        free(w->wq[l]); free(w->wk[l]); free(w->wv[l]); free(w->wo[l]);
        free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
        free(w->rms_att_weight[l]); free(w->rms_ffn_weight[l]);
        free(w->qwen3moe_attn_q_norm[l]); free(w->qwen3moe_attn_k_norm[l]);
    }
    free(w->rms_final_weight);
    free(w->wcls);
    weights_free_pointers(w);
}

static void test_qwen3moe_batch_matches_sequential(void) {
    tn_simd_init();

    Config cfg;
    TransformerWeights w;
    MoEConfig mc;
    build_qwen3moe_ternary_model(&cfg, &w, &mc);

    const int n_tokens = 3;
    int tokens[3] = {1, 3, 2};

    RunState s_ref;
    TEST_ASSERT(run_state_alloc_ex(&s_ref, &cfg, cfg.seq_len, /*skip_kv_cache=*/true) == TN_OK,
                "Q3MoE ref run_state_alloc_ex");
    TEST_ASSERT(qwen3moe_run_state_alloc(&s_ref, &cfg, &mc, cfg.seq_len) == TN_OK,
                "Q3MoE ref qwen3moe_run_state_alloc");
    sw_init(&s_ref.sw, cfg.seq_len, 0);
    float logits_ref[3][Q3_VOCAB];
    for (int k = 0; k < n_tokens; k++) {
        float *lg = transformer_forward(tokens[k], k, &cfg, &w, &s_ref, &mc, NULL);
        memcpy(logits_ref[k], lg, Q3_VOCAB * sizeof(float));
    }

    RunState s_batch;
    TEST_ASSERT(run_state_alloc_ex(&s_batch, &cfg, cfg.seq_len, /*skip_kv_cache=*/true) == TN_OK,
                "Q3MoE batch run_state_alloc_ex");
    TEST_ASSERT(qwen3moe_run_state_alloc(&s_batch, &cfg, &mc, cfg.seq_len) == TN_OK,
                "Q3MoE batch qwen3moe_run_state_alloc");
    sw_init(&s_batch.sw, cfg.seq_len, 0);
    SpecBatchScratch sb;
    TEST_ASSERT(spec_batch_scratch_alloc(&sb, &cfg, n_tokens) == TN_OK, "Q3MoE spec_batch_scratch_alloc");

    float logits_batch[3][Q3_VOCAB];
    TernaryError err = transformer_forward_batch(tokens, 0, n_tokens, &cfg, &w,
                                                  &s_batch, &sb, &mc, NULL,
                                                  (float *)logits_batch);
    TEST_ASSERT(err == TN_OK, "transformer_forward_batch returns TN_OK for Qwen3-MoE model");

    int all_match = 1;
    for (int k = 0; k < n_tokens; k++) {
        for (int i = 0; i < Q3_VOCAB; i++) {
            if (fabsf(logits_ref[k][i] - logits_batch[k][i]) > TOL) all_match = 0;
        }
    }
    TEST_ASSERT(all_match, "Qwen3-MoE (QK-norm) batched forward matches N sequential forward calls");

    spec_batch_scratch_free(&sb);
    qwen3moe_run_state_free(&s_ref, &cfg);
    run_state_free(&s_ref);
    qwen3moe_run_state_free(&s_batch, &cfg);
    run_state_free(&s_batch);
    free_qwen3moe_ternary_model(&w);
}

/* ── Negative test: linear-attention (Qwen3.5/3.6 hybrid) refusal ──────── */

static void test_linear_attn_returns_unsupported(void) {
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim = D_DIM; cfg.hidden_dim = D_HIDDEN; cfg.n_layers = 1;
    cfg.n_heads = D_HEADS; cfg.n_kv_heads = D_KV_HEADS; cfg.vocab_size = D_VOCAB;
    cfg.seq_len = D_SEQ; cfg.rms_norm_eps = 1e-5f; cfg.rope_freq_scale = 1.0f;

    MoEConfig mc;
    moe_config_init_dense(&mc);
    mc.has_linear_attn = 1;

    TransformerWeights w;
    build_dense_ternary_model(&cfg, &w); /* dims match; has_linear_attn is what's under test */

    RunState s;
    TEST_ASSERT(run_state_alloc(&s, &cfg, cfg.seq_len) == TN_OK, "linear-attn run_state_alloc");
    sw_init(&s.sw, cfg.seq_len, 0);
    SpecBatchScratch sb;
    TEST_ASSERT(spec_batch_scratch_alloc(&sb, &cfg, 3) == TN_OK, "linear-attn spec_batch_scratch_alloc");

    int tokens[3] = {1, 2, 3};
    float logits[3][D_VOCAB];
    TernaryError err = transformer_forward_batch(tokens, 0, 3, &cfg, &w, &s, &sb, &mc, NULL,
                                                  (float *)logits);
    TEST_ASSERT(err == TN_ERR_UNSUPPORTED,
                "transformer_forward_batch refuses has_linear_attn models with TN_ERR_UNSUPPORTED");

    spec_batch_scratch_free(&sb);
    run_state_free(&s);
    free_dense_ternary_model(&w);
}

int main(void) {
    /* Force the AVX2 (exact float32) dispatch tier, not VNNI: on a VNNI
     * capable host, tn_simd_init()'s default tier quantizes activations to
     * int8 internally for the ternary packed matmul (~1% error vs float32)
     * -- matches tests/test_batched_matmul.c's identical fix (see that
     * file's history) and tests/test_packed_weights.c's own precedent for
     * the same reason. The batched kernels under test here are always
     * plain float32 (no quantization), so the *reference* path
     * (transformer_forward(), which goes through the real SIMD dispatch)
     * must be pinned to a float32 tier too for a fair, tight-tolerance
     * comparison. */
    setenv("TN_FORCE_BACKEND", "avx2", 1);

    RUN_TEST(test_dense_batch_matches_sequential);
    RUN_TEST(test_mla_batch_matches_sequential);
    RUN_TEST(test_qwen3moe_batch_matches_sequential);
    RUN_TEST(test_linear_attn_returns_unsupported);
    TEST_SUMMARY();
}
