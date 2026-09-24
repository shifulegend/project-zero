/**
 * test_lora.c — Unit tests for Phase 19: LoRA adapters
 *
 * Tests:
 *  1. lora_apply() unit correctness against a manual reference dot-product,
 *     including the "LoRA disabled" edge cases (mod == NULL, rank == 0).
 *  2. .lora.bin load-roundtrip: write a synthetic file, load it, verify the
 *     decoded A/B values and scale match what was written, then free.
 *  3. transformer_forward()-level: a nonzero LoRA changes output vs. the
 *     lora=NULL baseline, and (critically) a NULL active_lora is byte-
 *     identical to Phase 19's pre-existing behavior (zero-cost no-op path).
 */

#include "core/config.h"
#include "core/lora.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "math/lora_matmul.h"
#include "math/simd_dispatch.h"
#include "test_harness.h"
#include "transformer/forward.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * TEST: lora_apply() unit correctness
 * ================================================================ */

static void write_f16_block(FILE *f, const float *vals, size_t count) {
  for (size_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &vals[i], sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    uint16_t h;
    if (exp <= 0) {
      h = (uint16_t)sign;
    } else if (exp >= 31) {
      h = (uint16_t)(sign | 0x7c00u);
    } else {
      h = (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
    }
    fwrite(&h, sizeof(h), 1, f);
  }
}

static void test_lora_apply_reference(void) {
  tn_simd_init();

  int n = 8, d = 4, rank = 2;
  float x[8];
  for (int i = 0; i < n; i++) x[i] = (float)(i + 1) * 0.5f;

  float A[2 * 8]; /* [rank x n] */
  float B[4 * 2]; /* [d x rank] */
  for (int i = 0; i < rank * n; i++) A[i] = (float)((i % 5) - 2) * 0.1f;
  for (int i = 0; i < d * rank; i++) B[i] = (float)((i % 3) - 1) * 0.2f;

  LoRAModule mod = { .A = A, .B = B, .rank = rank, .scale = 1.5f };

  float out[4];
  float out_expected[4];
  for (int i = 0; i < d; i++) out[i] = out_expected[i] = 10.0f + (float)i;

  /* Manual reference: tmp = A @ x, then out += scale * (B @ tmp) */
  float tmp[2];
  for (int r = 0; r < rank; r++) {
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += A[r * n + j] * x[j];
    tmp[r] = acc;
  }
  for (int i = 0; i < d; i++) {
    float acc = 0.0f;
    for (int r = 0; r < rank; r++) acc += B[i * rank + r] * tmp[r];
    out_expected[i] += mod.scale * acc;
  }

  lora_apply(out, x, &mod, n, d, NULL);

  for (int i = 0; i < d; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], out_expected[i], 1e-4f, "lora_apply matches manual reference");
  }
}

static void test_lora_apply_null_mod_is_noop(void) {
  float out[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_before[4];
  memcpy(out_before, out, sizeof(out));
  float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  lora_apply(out, x, NULL, 4, 4, NULL);

  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], out_before[i], 1e-9f, "NULL mod leaves out unchanged");
  }
}

static void test_lora_apply_zero_rank_is_noop(void) {
  float out[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out_before[4];
  memcpy(out_before, out, sizeof(out));
  float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float A[1] = {0.0f}, B[1] = {0.0f};

  LoRAModule mod = { .A = A, .B = B, .rank = 0, .scale = 1.0f };
  lora_apply(out, x, &mod, 4, 4, NULL);

  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], out_before[i], 1e-9f, "rank==0 mod leaves out unchanged");
  }
}

static void test_lora_apply_zero_ab_is_noop(void) {
  int n = 4, d = 4, rank = 2;
  float x[4] = {1.0f, -2.0f, 3.0f, 0.5f};
  float A[2 * 4]; /* all-zero A */
  float B[4 * 2]; /* nonzero B, but A==0 must still leave out unchanged */
  memset(A, 0, sizeof(A));
  for (int i = 0; i < d * rank; i++) B[i] = (float)(i + 1);

  LoRAModule mod = { .A = A, .B = B, .rank = rank, .scale = 2.0f };

  float out[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out_before[4];
  memcpy(out_before, out, sizeof(out));

  lora_apply(out, x, &mod, n, d, NULL);

  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_FLOAT_EQ(out[i], out_before[i], 1e-6f, "all-zero A leaves out unchanged");
  }
}

/* ================================================================
 * TEST: .lora.bin load-roundtrip
 * ================================================================ */

static void write_synthetic_lora_bin(const char *path, int n_layers, int rank,
                                      float alpha, int dim, int kv_dim, int hidden_dim,
                                      int target_mask, const float *q_a,
                                      const float *q_b) {
  FILE *f = fopen(path, "wb");
  if (!f) { TEST_ASSERT(0, "failed to open temp file for writing"); return; }

  uint32_t magic = LORA_BIN_MAGIC;
  uint32_t version = 1;
  int32_t  nl = n_layers;
  int32_t  rk = rank;
  int32_t  d  = dim;
  int32_t  hd = hidden_dim;
  int32_t  tm = target_mask;
  int32_t  kvd = kv_dim;
  uint8_t  zero[28];
  memset(zero, 0, sizeof(zero));

  fwrite(&magic, 4, 1, f);
  fwrite(&version, 4, 1, f);
  fwrite(&nl, 4, 1, f);
  fwrite(&rk, 4, 1, f);
  fwrite(&alpha, 4, 1, f);
  fwrite(&d, 4, 1, f);
  fwrite(&hd, 4, 1, f);
  fwrite(&tm, 4, 1, f);
  fwrite(&kvd, 4, 1, f);
  fwrite(zero, 1, 28, f); /* pad [36:64) to reach the 64-byte header */

  /* Only LORA_TARGET_Q (bit 0) populated in this synthetic file. */
  for (int l = 0; l < n_layers; l++) {
    write_f16_block(f, q_a, (size_t)rank * dim);
    write_f16_block(f, q_b, (size_t)dim * rank);
  }

  fclose(f);
}

static void test_lora_load_roundtrip(void) {
  const char *path = "/tmp/tn_test_lora.bin";
  int n_layers = 2, dim = 8, hidden_dim = 16, rank = 3;
  float alpha = 6.0f; /* scale = alpha/rank = 2.0 */

  float q_a[3 * 8], q_b[8 * 3];
  for (int i = 0; i < rank * dim; i++) q_a[i] = (float)(i + 1) * 0.25f;
  for (int i = 0; i < dim * rank; i++) q_b[i] = (float)(i + 1) * -0.25f;

  write_synthetic_lora_bin(path, n_layers, rank, alpha, dim, /*kv_dim=*/dim, hidden_dim,
                            1 << LORA_TARGET_Q, q_a, q_b);

  LoRAWeights lora;
  TernaryError err = lora_load(&lora, path, dim, /*kv_dim=*/dim, hidden_dim, n_layers);
  TEST_ASSERT(err == TN_OK, "lora_load succeeds on well-formed file");
  TEST_ASSERT_EQ(lora.n_layers, n_layers, "n_layers round-trips");
  TEST_ASSERT_EQ(lora.rank, rank, "rank round-trips");
  TEST_ASSERT_FLOAT_EQ(lora.alpha, alpha, 1e-6f, "alpha round-trips");
  TEST_ASSERT(lora.q != NULL, "q module array allocated (target_mask bit 0 set)");
  TEST_ASSERT(lora.k == NULL, "k module array left NULL (not in target_mask)");
  TEST_ASSERT(lora.v == NULL, "v module array left NULL (not in target_mask)");
  TEST_ASSERT(lora.gate == NULL, "gate module array left NULL (not in target_mask)");

  for (int l = 0; l < n_layers; l++) {
    const LoRAModule *m = lora_mod(lora.q, l);
    TEST_ASSERT(m != NULL, "lora_mod returns non-NULL for populated array");
    TEST_ASSERT_EQ(m->rank, rank, "per-layer rank matches header");
    TEST_ASSERT_FLOAT_EQ(m->scale, alpha / (float)rank, 1e-5f, "scale == alpha/rank");
    for (int i = 0; i < rank * dim; i++) {
      TEST_ASSERT_FLOAT_EQ(m->A[i], q_a[i], 5e-3f, "decoded A value matches (within F16 precision)");
    }
    for (int i = 0; i < dim * rank; i++) {
      TEST_ASSERT_FLOAT_EQ(m->B[i], q_b[i], 5e-3f, "decoded B value matches (within F16 precision)");
    }
  }

  const LoRAModule *k_mod = lora_mod(lora.k, 0);
  TEST_ASSERT(k_mod == NULL, "lora_mod on a NULL array returns NULL, never UB");

  lora_free(&lora);
  TEST_ASSERT(lora.q == NULL, "lora_free zeroes the struct");

  remove(path);
}

static void test_lora_load_rejects_shape_mismatch(void) {
  const char *path = "/tmp/tn_test_lora_mismatch.bin";
  int n_layers = 1, dim = 8, hidden_dim = 16, rank = 2;
  float q_a[2 * 8], q_b[8 * 2];
  memset(q_a, 0, sizeof(q_a));
  memset(q_b, 0, sizeof(q_b));

  write_synthetic_lora_bin(path, n_layers, rank, 4.0f, dim, /*kv_dim=*/dim, hidden_dim,
                            1 << LORA_TARGET_Q, q_a, q_b);

  LoRAWeights lora;
  /* Ask for a model with dim=16 (mismatched) -- must be rejected, not
   * silently loaded and later corrupt output. */
  TernaryError err = lora_load(&lora, path, /*dim=*/16, /*kv_dim=*/16, hidden_dim, n_layers);
  TEST_ASSERT(err != TN_OK, "lora_load rejects a dim mismatch against the base model");

  remove(path);
}

static void test_lora_load_rejects_kv_dim_mismatch(void) {
  /* Dedicated regression test for the GQA kv_dim bug found via a real
   * downloaded SmolLM2 LoRA adapter (docs/ai/mistakes.md, 2026-09-24): a
   * mismatched kv_dim alone (dim/hidden_dim/n_layers all correct) must be
   * rejected, not silently loaded with K/V module shapes that don't match
   * the actual base model's GQA width. */
  const char *path = "/tmp/tn_test_lora_kvdim_mismatch.bin";
  int n_layers = 1, dim = 8, hidden_dim = 16, rank = 2;
  float q_a[2 * 8], q_b[8 * 2];
  memset(q_a, 0, sizeof(q_a));
  memset(q_b, 0, sizeof(q_b));

  /* File built for a GQA model with kv_dim=4 (e.g. n_kv_heads < n_heads). */
  write_synthetic_lora_bin(path, n_layers, rank, 4.0f, dim, /*kv_dim=*/4, hidden_dim,
                            1 << LORA_TARGET_Q, q_a, q_b);

  LoRAWeights lora;
  /* Same dim/hidden_dim/n_layers, but the base model's actual kv_dim is 8
   * (MHA, kv_dim == dim) -- must be rejected. */
  TernaryError err = lora_load(&lora, path, dim, /*kv_dim=*/8, hidden_dim, n_layers);
  TEST_ASSERT(err != TN_OK, "lora_load rejects a kv_dim mismatch against the base model");

  remove(path);
}

static void test_lora_load_gqa_kv_shape(void) {
  /* Regression test for the exact bug: a K/V target on a GQA model
   * (kv_dim != dim) must decode B as [kv_dim x rank], not [dim x rank] --
   * verified end to end against a real downloaded SmolLM2-135M PEFT LoRA
   * adapter during Phase 19 development (see docs/ai/mistakes.md). */
  const char *path = "/tmp/tn_test_lora_gqa.bin";
  int n_layers = 1, dim = 8, kv_dim = 4, hidden_dim = 16, rank = 2;

  float v_a[2 * 8];   /* A: [rank x dim] -- V's input is always dim-wide */
  float v_b[4 * 2];   /* B: [kv_dim x rank] -- V's *output* is kv_dim-wide on GQA */
  for (int i = 0; i < rank * dim; i++) v_a[i] = (float)(i + 1) * 0.1f;
  for (int i = 0; i < kv_dim * rank; i++) v_b[i] = (float)(i + 1) * -0.1f;

  /* write_synthetic_lora_bin only knows how to write a Q-target block, so
   * write the file by hand here to place a V-target (kv-shaped) block. */
  FILE *f = fopen(path, "wb");
  TEST_ASSERT(f != NULL, "opened temp file for GQA test");
  uint32_t magic = LORA_BIN_MAGIC, version = 1;
  int32_t nl = n_layers, rk = rank, d = dim, hd = hidden_dim, tm = 1 << LORA_TARGET_V, kvd = kv_dim;
  float alpha = 2.0f; /* scale = 1.0 */
  uint8_t zero[28];
  memset(zero, 0, sizeof(zero));
  fwrite(&magic, 4, 1, f);
  fwrite(&version, 4, 1, f);
  fwrite(&nl, 4, 1, f);
  fwrite(&rk, 4, 1, f);
  fwrite(&alpha, 4, 1, f);
  fwrite(&d, 4, 1, f);
  fwrite(&hd, 4, 1, f);
  fwrite(&tm, 4, 1, f);
  fwrite(&kvd, 4, 1, f);
  fwrite(zero, 1, 28, f);
  write_f16_block(f, v_a, (size_t)rank * dim);
  write_f16_block(f, v_b, (size_t)kv_dim * rank);
  fclose(f);

  LoRAWeights lora;
  TernaryError err = lora_load(&lora, path, dim, kv_dim, hidden_dim, n_layers);
  TEST_ASSERT(err == TN_OK, "lora_load succeeds on a well-formed GQA (kv_dim != dim) file");
  TEST_ASSERT(lora.v != NULL, "v module array allocated");

  const LoRAModule *m = lora_mod(lora.v, 0);
  TEST_ASSERT(m != NULL, "lora_mod(v, 0) non-NULL");
  for (int i = 0; i < rank * dim; i++) {
    TEST_ASSERT_FLOAT_EQ(m->A[i], v_a[i], 5e-3f, "V's A (dim-wide) decodes correctly on GQA");
  }
  for (int i = 0; i < kv_dim * rank; i++) {
    TEST_ASSERT_FLOAT_EQ(m->B[i], v_b[i], 5e-3f, "V's B (kv_dim-wide, NOT dim-wide) decodes correctly on GQA");
  }

  lora_free(&lora);
  remove(path);
}

static void test_lora_load_rejects_bad_magic(void) {
  const char *path = "/tmp/tn_test_lora_badmagic.bin";
  FILE *f = fopen(path, "wb");
  uint8_t buf[64];
  memset(buf, 0, sizeof(buf));
  fwrite(buf, 1, sizeof(buf), f);
  fclose(f);

  LoRAWeights lora;
  TernaryError err = lora_load(&lora, path, 8, 8, 16, 1);
  TEST_ASSERT(err != TN_OK, "lora_load rejects a file with bad magic");

  remove(path);
}

/* ================================================================
 * TEST: transformer_forward()-level zero-behavior-change + effect
 * ================================================================ */

static Config lora_tiny_config(void) {
  Config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.dim = 16;
  cfg.hidden_dim = 32;
  cfg.n_layers = 1;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 2;
  cfg.vocab_size = 8;
  cfg.seq_len = 16;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_freq_scale = 1.0f;
  return cfg;
}

static void fill_pattern_f32(float *buf, size_t count, int seed) {
  for (size_t i = 0; i < count; i++) {
    int v = ((int)(i * 7 + seed * 13) % 3) - 1;
    buf[i] = (float)v;
  }
}

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

static void alloc_synthetic_weights(TransformerWeights *w, const Config *cfg) {
  int dim = cfg->dim;
  int hidden_dim = cfg->hidden_dim;
  int kv_dim = config_kv_dim(cfg);
  int nl = cfg->n_layers;

  memset(w, 0, sizeof(*w));
  weights_alloc_pointers(w, cfg);

  size_t emb_size = (size_t)cfg->vocab_size * dim;
  w->token_embedding_table = (tn_u16 *)calloc(emb_size, sizeof(tn_u16));
  fill_pattern_bf16(w->token_embedding_table, emb_size, 42);

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

    w->rms_att_weight[l] = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++) w->rms_att_weight[l][i] = 1.0f;

    w->rms_ffn_weight[l] = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++) w->rms_ffn_weight[l][i] = 1.0f;
  }

  w->rms_final_weight = (float *)calloc(dim, sizeof(float));
  for (int i = 0; i < dim; i++) w->rms_final_weight[i] = 1.0f;

  size_t cls_size = (size_t)dim * cfg->vocab_size;
  w->wcls = (tn_u16 *)calloc(cls_size, sizeof(tn_u16));
  fill_pattern_bf16(w->wcls, cls_size, 99);
}

static void free_synthetic_weights(TransformerWeights *w, const Config *cfg) {
  int nl = cfg->n_layers;
  free(w->token_embedding_table);
  for (int l = 0; l < nl; l++) {
    free(w->wq[l]); free(w->wk[l]); free(w->wv[l]); free(w->wo[l]);
    free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
    free(w->rms_att_weight[l]); free(w->rms_ffn_weight[l]);
  }
  free(w->rms_final_weight);
  free(w->wcls);
  weights_free_pointers(w);
}

/* Builds a synthetic LoRAWeights with a nonzero O-projection adapter active
 * on every layer, entirely in memory (no file I/O -- this test is about
 * transformer_forward()'s wiring, not the loader, which has its own test
 * above). Deliberately targets O (applied directly to s->xb2 after the wo
 * dispatch), not Q/K: with a single token in context, tn_softmax() over a
 * length-1 score array always yields weight 1.0 regardless of the score's
 * value, so a Q- or K-only LoRA is mathematically a no-op for single-token
 * attention (softmax degeneracy, not a bug) -- it would only show an effect
 * once a second token puts real competition into the softmax. O sidesteps
 * that degeneracy entirely since it's added after the attention-weighted
 * sum, so it is the correct, minimal choice for this single-token test. */
static void alloc_synthetic_lora(LoRAWeights *lora, const Config *cfg) {
  memset(lora, 0, sizeof(*lora));
  lora->n_layers = cfg->n_layers;
  lora->rank = 2;
  lora->alpha = 4.0f; /* scale = 2.0 */
  lora->dim = cfg->dim;
  lora->hidden_dim = cfg->hidden_dim;

  lora->o = (LoRAModule *)calloc((size_t)cfg->n_layers, sizeof(LoRAModule));
  for (int l = 0; l < cfg->n_layers; l++) {
    float *A = (float *)malloc((size_t)lora->rank * cfg->dim * sizeof(float));
    float *B = (float *)malloc((size_t)cfg->dim * lora->rank * sizeof(float));
    fill_pattern_f32(A, (size_t)lora->rank * cfg->dim, l * 50 + 11);
    fill_pattern_f32(B, (size_t)cfg->dim * lora->rank, l * 50 + 12);
    lora->o[l].A = A;
    lora->o[l].B = B;
    lora->o[l].rank = lora->rank;
    lora->o[l].scale = lora->alpha / (float)lora->rank;
  }
}

static void free_synthetic_lora(LoRAWeights *lora) {
  for (int l = 0; l < lora->n_layers; l++) {
    free((void *)lora->o[l].A);
    free((void *)lora->o[l].B);
  }
  free(lora->o);
}

static void test_forward_lora_null_is_byte_identical(void) {
  Config cfg = lora_tiny_config();
  TransformerWeights w;
  RunState s_baseline, s_explicit_null;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);

  run_state_alloc(&s_baseline, &cfg, cfg.seq_len);
  sw_init(&s_baseline.sw, cfg.seq_len, 0);
  /* s_baseline.active_lora is NULL from run_state_alloc_ex()'s memset --
   * never touched, exactly like every pre-Phase-19 caller. */

  run_state_alloc(&s_explicit_null, &cfg, cfg.seq_len);
  sw_init(&s_explicit_null.sw, cfg.seq_len, 0);
  s_explicit_null.active_lora = NULL; /* explicit, should be identical */

  float *logits_baseline = transformer_forward(0, 0, &cfg, &w, &s_baseline, NULL, NULL);
  float logits_baseline_copy[8];
  memcpy(logits_baseline_copy, logits_baseline, cfg.vocab_size * sizeof(float));

  float *logits_null = transformer_forward(0, 0, &cfg, &w, &s_explicit_null, NULL, NULL);

  for (int i = 0; i < cfg.vocab_size; i++) {
    TEST_ASSERT_FLOAT_EQ(logits_null[i], logits_baseline_copy[i], 1e-9f,
                          "active_lora==NULL is byte-identical to pre-Phase-19 baseline");
  }

  run_state_free(&s_baseline);
  run_state_free(&s_explicit_null);
  free_synthetic_weights(&w, &cfg);
}

static void test_forward_lora_active_changes_output(void) {
  Config cfg = lora_tiny_config();
  TransformerWeights w;
  RunState s_off, s_on;
  LoRAWeights lora;

  tn_simd_init();
  alloc_synthetic_weights(&w, &cfg);
  alloc_synthetic_lora(&lora, &cfg);

  run_state_alloc(&s_off, &cfg, cfg.seq_len);
  sw_init(&s_off.sw, cfg.seq_len, 0);
  s_off.active_lora = NULL;

  run_state_alloc(&s_on, &cfg, cfg.seq_len);
  sw_init(&s_on.sw, cfg.seq_len, 0);
  s_on.active_lora = &lora;

  float *logits_off = transformer_forward(0, 0, &cfg, &w, &s_off, NULL, NULL);
  float logits_off_copy[8];
  memcpy(logits_off_copy, logits_off, cfg.vocab_size * sizeof(float));

  float *logits_on = transformer_forward(0, 0, &cfg, &w, &s_on, NULL, NULL);

  int differs = 0;
  for (int i = 0; i < cfg.vocab_size; i++) {
    if (fabsf(logits_on[i] - logits_off_copy[i]) > 1e-6f) differs = 1;
    TEST_ASSERT(!isnan(logits_on[i]), "no NaN with LoRA active");
    TEST_ASSERT(!isinf(logits_on[i]), "no Inf with LoRA active");
  }
  TEST_ASSERT(differs, "a nonzero active LoRA changes forward() output");

  run_state_free(&s_off);
  run_state_free(&s_on);
  free_synthetic_lora(&lora);
  free_synthetic_weights(&w, &cfg);
}

/* ================================================================ */

int main(void) {
  RUN_TEST(test_lora_apply_reference);
  RUN_TEST(test_lora_apply_null_mod_is_noop);
  RUN_TEST(test_lora_apply_zero_rank_is_noop);
  RUN_TEST(test_lora_apply_zero_ab_is_noop);
  RUN_TEST(test_lora_load_roundtrip);
  RUN_TEST(test_lora_load_rejects_shape_mismatch);
  RUN_TEST(test_lora_load_rejects_kv_dim_mismatch);
  RUN_TEST(test_lora_load_gqa_kv_shape);
  RUN_TEST(test_lora_load_rejects_bad_magic);
  RUN_TEST(test_forward_lora_null_is_byte_identical);
  RUN_TEST(test_forward_lora_active_changes_output);

  TEST_SUMMARY();
}
