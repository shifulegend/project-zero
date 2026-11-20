#ifndef TESTS_SYNTH_MODELS_H
#define TESTS_SYNTH_MODELS_H

/* Synthetic ternary models and fill helpers shared by tests/test_forward_batch.c and
 * tests/test_qwen3moe_concurrent.c. Include after test_harness.h and the core headers. */

/* ── Shared fill helpers ──────────────────────────────────────────────── */

static void fill_ternary_i8(tn_i8 *buf, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed   = seed * 1103515245 + 12345;
        int r  = (seed >> 16) % 3;
        buf[i] = (tn_i8)(r - 1);
    }
}

/* Packs a [rows][cols] ternary matrix (row-major) into 2-bit packed form. */
static tn_u8 *make_packed_ternary(int rows, int cols, unsigned seed) {
    size_t row_bytes = packed_bytes(cols);
    tn_u8 *packed    = (tn_u8 *)calloc((size_t)rows * row_bytes, 1);
    tn_i8 *tmp       = (tn_i8 *)malloc((size_t)rows * cols);
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
        int v   = ((int)(i * 7 + (unsigned)seed * 13) % 3) - 1;
        float f = (float)v * 0.5f;
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        buf[i] = (tn_u16)(bits >> 16);
    }
}

static void fill_ones(float *buf, int count) {
    for (int i = 0; i < count; i++) buf[i] = 1.0f;
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
    cfg->dim                   = Q3_DIM;
    cfg->hidden_dim            = Q3_HIDDEN;
    cfg->n_layers              = Q3_LAYERS;
    cfg->n_heads               = Q3_HEADS;
    cfg->n_kv_heads            = Q3_KV_HEADS;
    cfg->vocab_size            = Q3_VOCAB;
    cfg->seq_len               = Q3_SEQ;
    cfg->rms_norm_eps          = 1e-5f;
    cfg->rope_freq_scale       = 1.0f;
    cfg->rope_theta            = 10000.0f;
    cfg->rope_yarn_attn_factor = 1.0f;

    moe_config_init_dense(mc);
    mc->has_qk_norm   = 1;
    mc->attn_head_dim = Q3_HEAD_DIM;
    /* is_moe left false: dense FFN, keeping this test focused on
     * QK-norm attention batching alone (mirrors the MLA test's approach). */

    memset(w, 0, sizeof(*w));
    weights_alloc_pointers(w, cfg);
    w->layers_are_ternary = true;

    int q_width  = Q3_HEADS * Q3_HEAD_DIM;
    int kv_width = Q3_KV_HEADS * Q3_HEAD_DIM;

    w->token_embedding_table = (tn_u16 *)calloc((size_t)Q3_VOCAB * Q3_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->token_embedding_table, (size_t)Q3_VOCAB * Q3_DIM, 1);

    for (int l = 0; l < Q3_LAYERS; l++) {
        w->attn.wq[l] = (tn_i8 *)make_packed_ternary(q_width, Q3_DIM, 11 + l);
        w->attn.sq[l] = 1.0f;
        w->attn.wk[l] = (tn_i8 *)make_packed_ternary(kv_width, Q3_DIM, 22 + l);
        w->attn.sk[l] = 1.0f;
        w->attn.wv[l] = (tn_i8 *)make_packed_ternary(kv_width, Q3_DIM, 33 + l);
        w->attn.sv[l] = 1.0f;
        w->attn.wo[l] = (tn_i8 *)make_packed_ternary(Q3_DIM, q_width, 44 + l);
        w->attn.so[l] = 1.0f;

        w->ffn.w1[l] = (tn_i8 *)make_packed_ternary(Q3_HIDDEN, Q3_DIM, 55 + l);
        w->ffn.s1[l] = 1.0f;
        w->ffn.w3[l] = (tn_i8 *)make_packed_ternary(Q3_HIDDEN, Q3_DIM, 66 + l);
        w->ffn.s3[l] = 1.0f;
        w->ffn.w2[l] = (tn_i8 *)make_packed_ternary(Q3_DIM, Q3_HIDDEN, 77 + l);
        w->ffn.s2[l] = 1.0f;

        w->attn.norm[l] = (float *)malloc(Q3_DIM * sizeof(float));
        fill_ones(w->attn.norm[l], Q3_DIM);
        w->ffn.norm[l] = (float *)malloc(Q3_DIM * sizeof(float));
        fill_ones(w->ffn.norm[l], Q3_DIM);

        w->qwen3moe.attn_q_norm[l] = (float *)malloc(Q3_HEAD_DIM * sizeof(float));
        fill_ones(w->qwen3moe.attn_q_norm[l], Q3_HEAD_DIM);
        w->qwen3moe.attn_k_norm[l] = (float *)malloc(Q3_HEAD_DIM * sizeof(float));
        fill_ones(w->qwen3moe.attn_k_norm[l], Q3_HEAD_DIM);
    }

    w->rms_final_weight = (float *)malloc(Q3_DIM * sizeof(float));
    fill_ones(w->rms_final_weight, Q3_DIM);

    w->cls.w = (tn_u16 *)calloc((size_t)Q3_VOCAB * Q3_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->cls.w, (size_t)Q3_VOCAB * Q3_DIM, 99);
}

static void free_qwen3moe_ternary_model(TransformerWeights *w) {
    free(w->token_embedding_table);
    for (int l = 0; l < Q3_LAYERS; l++) {
        free(w->attn.wq[l]);
        free(w->attn.wk[l]);
        free(w->attn.wv[l]);
        free(w->attn.wo[l]);
        free(w->ffn.w1[l]);
        free(w->ffn.w2[l]);
        free(w->ffn.w3[l]);
        free(w->attn.norm[l]);
        free(w->ffn.norm[l]);
        free(w->qwen3moe.attn_q_norm[l]);
        free(w->qwen3moe.attn_k_norm[l]);
    }
    free(w->rms_final_weight);
    free(w->cls.w);
    weights_free_pointers(w);
}

#endif
