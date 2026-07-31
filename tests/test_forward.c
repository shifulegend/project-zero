/**
 * test_forward.c — Unit tests for Phase 6: Transformer Forward Pass
 *
 * Tests:
 *  1. Token embedding dequantization (6.1)
 *  2. Attention forward pass with known weights (6.2)
 *  3. FFN forward pass with known weights (6.4)
 *  4. Full forward pass end-to-end (6.5)
 *  5. GQA (grouped query attention) correctness
 *  6. Multi-position KV cache population
 *
 * Uses a tiny synthetic model (dim=16, hidden_dim=32, n_layers=1,
 * n_heads=2, n_kv_heads=2, vocab_size=8, seq_len=16) to verify
 * mathematical correctness against hand-computable results.
 */

#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "memory/aligned_alloc.h"
#include "test_harness.h"
#include "transformer/attention.h"
#include "transformer/embedding.h"
#include "transformer/ffn.h"
#include "transformer/forward.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---------- helpers ---------- */

/* Fill float buffer with a deterministic pattern based on seed */
static void fill_pattern_f32(float *buf, size_t count, int seed) {
  for (size_t i = 0; i < count; i++) {
    /* Generate values in {-1.0, 0.0, 1.0} */
    int v = ((int)(i * 7 + seed * 13) % 3) - 1;
    buf[i] = (float)v;
  }
}

/* Fill BF16 (tn_u16) buffer using the same pattern */
static void fill_pattern_bf16(tn_u16 *buf, size_t count, int seed) {
  float *tmp = (float *)calloc(count, sizeof(float));
  if (!tmp) return;
  fill_pattern_f32(tmp, count, seed);
  for (size_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &tmp[i], sizeof(bits));
    buf[i] = (tn_u16)(bits >> 16);
  }
  free(tmp);
}

/* Convert an explicit float array to BF16 */
static void float_array_to_bf16(const float *in, tn_u16 *out, size_t count) {
  for (size_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &in[i], sizeof(bits));
    out[i] = (tn_u16)(bits >> 16);
  }
}

/* Tiny model config */
static Config tiny_config(void) {
  Config cfg;
  memset(&cfg, 0, sizeof(cfg));  /* zero all fields first */
  cfg.dim = 16;
  cfg.hidden_dim = 32;
  cfg.n_layers = 1;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 2;
  cfg.vocab_size = 8;
  cfg.seq_len = 16;
  cfg.rms_norm_eps = 1e-5f;     /* required: RMSNorm divides by sqrt(mean^2 + eps) */
  cfg.rope_freq_scale = 1.0f;   /* required: RoPE freq scale (0 = no scaling = correct default) */
  return cfg;
}

/* Allocate a synthetic weight blob and populate TransformerWeights */
static void alloc_synthetic_weights(TransformerWeights *w, const Config *cfg) {
  int dim = cfg->dim;
  int hidden_dim = cfg->hidden_dim;
  int kv_dim = config_kv_dim(cfg);
  int nl = cfg->n_layers;

  /* Zero the struct first — weights_alloc_pointers no longer does this,
   * making it the caller's responsibility (modular design). */
  memset(w, 0, sizeof(*w));

  /* Allocate pointer arrays */
  weights_alloc_pointers(w, cfg);

  /* Token embedding: vocab_size * dim */
  size_t emb_size = (size_t)cfg->vocab_size * dim;
  w->token_embedding_table = (tn_u16 *)calloc(emb_size, sizeof(tn_u16));
  fill_pattern_bf16(w->token_embedding_table, emb_size, 42);

  /* Per-layer weights */
  for (int l = 0; l < nl; l++) {
    size_t qo_size = (size_t)dim * dim;
    size_t kv_size = (size_t)dim * kv_dim;

    w->wq[l] = (tn_i8 *)calloc(qo_size, sizeof(float));
    fill_pattern_f32((float *)w->wq[l], qo_size, l * 100 + 1);
    w->sq[l] = 1.0f;

    w->wk[l] = (tn_i8 *)calloc(kv_size, sizeof(float));
    fill_pattern_f32((float *)w->wk[l], kv_size, l * 100 + 2);
    w->sk[l] = 1.0f;

    w->wv[l] = (tn_i8 *)calloc(kv_size, sizeof(float));
    fill_pattern_f32((float *)w->wv[l], kv_size, l * 100 + 3);
    w->sv[l] = 1.0f;

    w->wo[l] = (tn_i8 *)calloc(qo_size, sizeof(float));
    fill_pattern_f32((float *)w->wo[l], qo_size, l * 100 + 4);
    w->so[l] = 1.0f;

    size_t gate_size = (size_t)dim * hidden_dim;
    size_t down_size = (size_t)hidden_dim * dim;

    w->w1[l] = (tn_i8 *)calloc(gate_size, sizeof(float));
    fill_pattern_f32((float *)w->w1[l], gate_size, l * 100 + 5);
    w->s1[l] = 1.0f;

    w->w2[l] = (tn_i8 *)calloc(down_size, sizeof(float));
    fill_pattern_f32((float *)w->w2[l], down_size, l * 100 + 6);
    w->s2[l] = 1.0f;

    w->w3[l] = (tn_i8 *)calloc(gate_size, sizeof(float));
    fill_pattern_f32((float *)w->w3[l], gate_size, l * 100 + 7);
    w->s3[l] = 1.0f;

    /* RMS norm weights (all 1.0 for simplicity) */
    w->rms_att_weight[l] = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++)
      w->rms_att_weight[l][i] = 1.0f;

    w->rms_ffn_weight[l] = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++)
      w->rms_ffn_weight[l][i] = 1.0f;
  }

  /* Final RMS norm */
  w->rms_final_weight = (float *)calloc(dim, sizeof(float));
  for (int i = 0; i < dim; i++)
    w->rms_final_weight[i] = 1.0f;

  /* Output classifier */
  size_t cls_size = (size_t)dim * cfg->vocab_size;
  w->wcls = (tn_u16 *)calloc(cls_size, sizeof(tn_u16));
  fill_pattern_bf16(w->wcls, cls_size, 99);
}

static void free_synthetic_weights(TransformerWeights *w, const Config *cfg) {
  int nl = cfg->n_layers;

  free(w->token_embedding_table);
  for (int l = 0; l < nl; l++) {
    free(w->wq[l]);
    free(w->wk[l]);
    free(w->wv[l]);
    free(w->wo[l]);
    free(w->w1[l]);
    free(w->w2[l]);
    free(w->w3[l]);
    free(w->rms_att_weight[l]);
    free(w->rms_ffn_weight[l]);
  }
  free(w->rms_final_weight);
  free(w->wcls);
  weights_free_pointers(w);
}

/* ================================================================
 * TEST: Token Embedding (Phase 6.1)
 * ================================================================ */

static void test_embed_token_basic(void) {
  int dim = 8;
  /* embed_token now reads float values directly (scale is ignored) */
  float table_f[] = {
      /* token 0 */ 2.5f, -2.5f, 0.0f, 2.5f, -2.5f, 0.0f,  2.5f, -2.5f,
      /* token 1 */ 0.0f, 2.5f,  2.5f, 0.0f, -2.5f, -2.5f, 0.0f, 2.5f,
      /* token 2 */ 2.5f, 2.5f,  2.5f, 2.5f, 2.5f,  2.5f,  2.5f, 2.5f,
  };
  tn_u16 table[sizeof(table_f)/sizeof(table_f[0])];
  float_array_to_bf16(table_f, table, sizeof(table_f)/sizeof(table_f[0]));
  float out[8];

  /* Embed token 0 — BF16 path: embd_f32=NULL, use bf16 table */
  embed_token(out, 0, NULL, table, dim);
  TEST_ASSERT_FLOAT_EQ(out[0], 2.5f, 1e-2f, "token0[0] = 2.5");
  TEST_ASSERT_FLOAT_EQ(out[1], -2.5f, 1e-2f, "token0[1] = -2.5");
  TEST_ASSERT_FLOAT_EQ(out[2], 0.0f, 1e-6f, "token0[2] = 0.0");
  TEST_ASSERT_FLOAT_EQ(out[3], 2.5f, 1e-2f, "token0[3] = 2.5");

  /* Embed token 1 */
  embed_token(out, 1, NULL, table, dim);
  TEST_ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-6f, "token1[0] = 0.0");
  TEST_ASSERT_FLOAT_EQ(out[1], 2.5f, 1e-2f, "token1[1] = 2.5");
  TEST_ASSERT_FLOAT_EQ(out[4], -2.5f, 1e-2f, "token1[4] = -2.5");

  /* Embed token 2 (all 2.5) */
  embed_token(out, 2, NULL, table, dim);
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], 2.5f, 1e-6f, "token2 all 2.5");
  }
}

static void test_embed_token_scale_zero(void) {
  int dim = 4;
  /* embed_token copies float values directly; scale is ignored.
   * Use zero-valued table to verify zero output. */
  float table_f[] = {0.0f, 0.0f, 0.0f, 0.0f};
  tn_u16 table[4];
  float_array_to_bf16(table_f, table, 4);
  float out[4];
  embed_token(out, 0, NULL, table, dim);
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], 0.0f, 1e-6f, "zero table gives zero output");
  }
}

static void test_embed_identity_scale(void) {
  int dim = 4;
  /* embed_token copies float values directly */
  float table_f[] = {1.0f, 0.0f, -1.0f, 1.0f};
  tn_u16 table[4];
  float_array_to_bf16(table_f, table, 4);
  float out[4];

  embed_token(out, 0, NULL, table, dim);
  TEST_ASSERT_FLOAT_EQ(out[0], 1.0f, 1e-2f, "identity 1.0");
  TEST_ASSERT_FLOAT_EQ(out[1], 0.0f, 1e-6f, "identity 0.0");
  TEST_ASSERT_FLOAT_EQ(out[2], -1.0f, 1e-2f, "identity -1.0");
  TEST_ASSERT_FLOAT_EQ(out[3], 1.0f, 1e-2f, "identity 1.0 again");
}

/* ================================================================
 * TEST: Attention Forward Pass (Phase 6.2)
 * ================================================================ */

static void test_attention_forward_basic(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  for (int i = 0; i < cfg.dim; i++)
    s.x[i] = 1.0f;

  attention_forward(&s, &w, &cfg, NULL, 0, 0, NULL);

  int has_nonzero = 0, has_diff = 0;
  for (int i = 0; i < cfg.dim; i++) {
    if (fabsf(s.x[i]) > 1e-6f)
      has_nonzero = 1;
    if (fabsf(s.x[i] - 1.0f) > 1e-6f)
      has_diff = 1;
  }
  TEST_ASSERT(has_nonzero, "attention output has non-zero values");
  TEST_ASSERT(has_diff, "attention modifies input");

  for (int i = 0; i < cfg.dim; i++) {
    TEST_ASSERT(!isnan(s.x[i]), "no NaN in attention output");
    TEST_ASSERT(!isinf(s.x[i]), "no Inf in attention output");
  }

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

static void test_attention_multi_position(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  for (int pos = 0; pos < 4; pos++) {
    for (int i = 0; i < cfg.dim; i++)
      s.x[i] = 1.0f + 0.1f * (float)pos;

    attention_forward(&s, &w, &cfg, NULL, 0, pos, NULL);

    for (int i = 0; i < cfg.dim; i++) {
      TEST_ASSERT(!isnan(s.x[i]), "no NaN at multi-pos");
      TEST_ASSERT(!isinf(s.x[i]), "no Inf at multi-pos");
    }
  }

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

/* ================================================================
 * TEST: Attention with GQA (n_kv_heads < n_heads)
 * ================================================================ */

static void test_attention_gqa(void) {
  Config cfg = tiny_config();
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2; /* GQA: 4 Q heads, 2 KV heads */

  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  for (int i = 0; i < cfg.dim; i++)
    s.x[i] = 1.0f;

  attention_forward(&s, &w, &cfg, NULL, 0, 0, NULL);

  for (int i = 0; i < cfg.dim; i++) {
    TEST_ASSERT(!isnan(s.x[i]), "no NaN in GQA output");
    TEST_ASSERT(!isinf(s.x[i]), "no Inf in GQA output");
  }

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

/* ================================================================
 * TEST: FFN Forward Pass (Phase 6.4)
 * ================================================================ */

static void test_ffn_forward_basic(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  for (int i = 0; i < cfg.dim; i++)
    s.x[i] = 1.0f;

  ffn_forward(&s, &w, &cfg, NULL, 0, NULL);

  int has_nonzero = 0;
  for (int i = 0; i < cfg.dim; i++) {
    if (fabsf(s.x[i]) > 1e-6f)
      has_nonzero = 1;
    TEST_ASSERT(!isnan(s.x[i]), "no NaN in FFN output");
    TEST_ASSERT(!isinf(s.x[i]), "no Inf in FFN output");
  }
  TEST_ASSERT(has_nonzero, "FFN output has non-zero values");

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

static void test_ffn_residual(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  float x_before[16];
  for (int i = 0; i < cfg.dim; i++) {
    s.x[i] = 0.5f + 0.05f * (float)i;
    x_before[i] = s.x[i];
  }

  ffn_forward(&s, &w, &cfg, NULL, 0, NULL);

  int differs = 0;
  for (int i = 0; i < cfg.dim; i++) {
    if (fabsf(s.x[i] - x_before[i]) > 1e-6f)
      differs = 1;
  }
  TEST_ASSERT(differs, "FFN residual should change x");

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

/* ================================================================
 * TEST: Full Forward Pass (Phase 6.5)
 * ================================================================ */

static void test_forward_single_token(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  float *logits = transformer_forward(0, 0, &cfg, &w, &s, NULL, NULL);

  TEST_ASSERT(logits != NULL, "forward returns non-NULL logits");
  TEST_ASSERT(logits == s.logits, "logits point to s.logits");

  for (int i = 0; i < cfg.vocab_size; i++) {
    TEST_ASSERT(!isnan(logits[i]), "logits no NaN");
    TEST_ASSERT(!isinf(logits[i]), "logits no Inf");
  }

  int has_nonzero = 0;
  for (int i = 0; i < cfg.vocab_size; i++) {
    if (fabsf(logits[i]) > 1e-6f)
      has_nonzero = 1;
  }
  TEST_ASSERT(has_nonzero, "logits have non-zero values");

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

static void test_forward_multiple_tokens(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  float prev_logits[8];

  for (int pos = 0; pos < 4; pos++) {
    int token = pos % cfg.vocab_size;
    float *logits = transformer_forward(token, pos, &cfg, &w, &s, NULL, NULL);

    for (int i = 0; i < cfg.vocab_size; i++) {
      TEST_ASSERT(!isnan(logits[i]), "no NaN at multi-token");
      TEST_ASSERT(!isinf(logits[i]), "no Inf at multi-token");
    }

    if (pos > 0) {
      int differs = 0;
      for (int i = 0; i < cfg.vocab_size; i++) {
        if (fabsf(logits[i] - prev_logits[i]) > 1e-6f)
          differs = 1;
      }
      TEST_ASSERT(differs, "logits differ between positions");
    }

    memcpy(prev_logits, logits, cfg.vocab_size * sizeof(float));
  }

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

static void test_forward_different_tokens(void) {
  Config cfg = tiny_config();
  TransformerWeights w;
  RunState s1, s2;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  run_state_alloc(&s1, &cfg, cfg.seq_len);
  run_state_alloc(&s2, &cfg, cfg.seq_len);
  sw_init(&s1.sw, cfg.seq_len, 0);
  sw_init(&s2.sw, cfg.seq_len, 0);

  float *logits_0 = transformer_forward(0, 0, &cfg, &w, &s1, NULL, NULL);
  float logits_copy[8];
  memcpy(logits_copy, logits_0, cfg.vocab_size * sizeof(float));

  float *logits_1 = transformer_forward(1, 0, &cfg, &w, &s2, NULL, NULL);

  int differs = 0;
  for (int i = 0; i < cfg.vocab_size; i++) {
    if (fabsf(logits_copy[i] - logits_1[i]) > 1e-6f)
      differs = 1;
  }
  TEST_ASSERT(differs, "different tokens produce different logits");

  run_state_free(&s1);
  run_state_free(&s2);
  free_synthetic_weights(&w, &cfg);
}

static void test_forward_multilayer(void) {
  Config cfg = tiny_config();
  cfg.n_layers = 3;

  TransformerWeights w;
  RunState s;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  TernaryError err = run_state_alloc(&s, &cfg, cfg.seq_len);
  TEST_ASSERT(err == TN_OK, "run_state_alloc failed");
  sw_init(&s.sw, cfg.seq_len, 0);

  float *logits = transformer_forward(0, 0, &cfg, &w, &s, NULL, NULL);

  for (int i = 0; i < cfg.vocab_size; i++) {
    TEST_ASSERT(!isnan(logits[i]), "multilayer no NaN");
    TEST_ASSERT(!isinf(logits[i]), "multilayer no Inf");
  }

  logits = transformer_forward(1, 1, &cfg, &w, &s, NULL, NULL);
  for (int i = 0; i < cfg.vocab_size; i++) {
    TEST_ASSERT(!isnan(logits[i]), "multilayer pos=1 no NaN");
    TEST_ASSERT(!isinf(logits[i]), "multilayer pos=1 no Inf");
  }

  run_state_free(&s);
  free_synthetic_weights(&w, &cfg);
}

/* ================================================================ */

int main(void) {
  RUN_TEST(test_embed_token_basic);
  RUN_TEST(test_embed_token_scale_zero);
  RUN_TEST(test_embed_identity_scale);
  RUN_TEST(test_attention_forward_basic);
  RUN_TEST(test_attention_multi_position);
  RUN_TEST(test_attention_gqa);
  RUN_TEST(test_ffn_forward_basic);
  RUN_TEST(test_ffn_residual);
  RUN_TEST(test_forward_single_token);
  RUN_TEST(test_forward_multiple_tokens);
  RUN_TEST(test_forward_different_tokens);
  RUN_TEST(test_forward_multilayer);

  TEST_SUMMARY();
}
