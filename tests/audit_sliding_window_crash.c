#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "test_harness.h"
#include "transformer/forward.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Audit Test: AUD-MEM-01 (KV Cache Overflow)
 */
static void aud_kv_cache_overflow(void) {
  Config cfg = {.dim = 16,
                .hidden_dim = 32,
                .n_layers = 1,
                .n_heads = 2,
                .n_kv_heads = 2,
                .vocab_size = 32,
                .seq_len = 128};

  TransformerWeights w;
  RunState s;

  tn_simd_init();

  /* Mock weights — allocate as float32 (layers_are_ternary defaults to false) */
  weights_alloc_pointers(&w, &cfg);
  w.token_embedding_table = calloc(cfg.vocab_size * cfg.dim, sizeof(float));
  w.wq[0] = calloc(cfg.dim * cfg.dim, sizeof(float));
  w.wk[0] = calloc(cfg.dim * config_kv_dim(&cfg), sizeof(float));
  w.wv[0] = calloc(cfg.dim * config_kv_dim(&cfg), sizeof(float));
  w.wo[0] = calloc(cfg.dim * cfg.dim, sizeof(float));

  w.w1[0] = calloc((size_t)cfg.hidden_dim * cfg.dim, sizeof(float));
  w.w2[0] = calloc((size_t)cfg.dim * cfg.hidden_dim, sizeof(float));
  w.w3[0] = calloc((size_t)cfg.hidden_dim * cfg.dim, sizeof(float));

  w.rms_att_weight[0] = calloc(cfg.dim, 4);
  w.rms_ffn_weight[0] = calloc(cfg.dim, 4);
  w.rms_final_weight = calloc(cfg.dim, 4);
  w.wcls = calloc((size_t)cfg.vocab_size * cfg.dim, sizeof(float));

  /* Allocate RunState with ONLY 10 tokens capacity */
  int limited_seq = 10;
  int head_dim = config_head_dim(&cfg);
  TernaryError err = run_state_alloc(&s, &cfg, limited_seq);
  TEST_ASSERT(err == TN_OK, "Allocated small RunState");

  /* Initialize Head 1 with a known value to check for corruption later */
  size_t h1_pos0 =
      KV_CACHE_IDX(0, 1, 0, 0, cfg.n_kv_heads, limited_seq, head_dim);
  s.key_cache[h1_pos0] = 123.45f;

  printf("[Auditor] Running forward pass at pos=5 (safe)...\n");
  transformer_forward(1, 5, &cfg, &w, &s, NULL, NULL);

  printf(
      "[Auditor] Running forward pass at pos=15 (Silent Corruption Test)...\n");
  /*
   * Since limited_seq=10, pos=15 is out of bounds for Head 0.
   */
  transformer_forward(1, 15, &cfg, &w, &s, NULL, NULL);

  size_t h0_pos15 =
      KV_CACHE_IDX(0, 0, 15, 0, cfg.n_kv_heads, limited_seq, head_dim);
  printf("[Auditor] H0 Pos 15 calculated index: %zu (Max H0 index: %d)\n",
         h0_pos15, limited_seq * head_dim - 1);

  if (h0_pos15 >= (size_t)limited_seq * head_dim) {
    printf("[Auditor] VULNERABILITY CONFIRMED: pos=15 maps outside Head 0 "
           "bounds.\n");
  }

  /* Cleanup */
  run_state_free(&s);
  free(w.token_embedding_table);
  free(w.wq[0]);
  free(w.wk[0]);
  free(w.wv[0]);
  free(w.wo[0]);
  free(w.w1[0]);
  free(w.w2[0]);
  free(w.w3[0]);
  free(w.rms_att_weight[0]);
  free(w.rms_ffn_weight[0]);
  free(w.rms_final_weight);
  free(w.wcls);
  weights_free_pointers(&w);
}

int main(void) {
  RUN_TEST(aud_kv_cache_overflow);
  TEST_SUMMARY();
}
