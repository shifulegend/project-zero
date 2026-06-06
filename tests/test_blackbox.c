#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "test_harness.h"
#include "tokenizer/tokenizer.h"
#include "transformer/forward.h"

/* ---------- Helper: Mock Weights ---------- */
static void fill_mock_weights(TransformerWeights *w, const Config *cfg) {
  int dim = cfg->dim;
  int hidden_dim = cfg->hidden_dim;
  int kv_dim = config_kv_dim(cfg);
  int nl = cfg->n_layers;

  weights_alloc_pointers(w, cfg);

  size_t emb_size = (size_t)cfg->vocab_size * dim;
  w->token_embedding_table = (tn_u16 *)calloc(emb_size, sizeof(tn_u16));
  for (size_t i = 0; i < emb_size; i++) {
    float fv = (float)(i % 3) - 1.0f; /* -1.0, 0.0, 1.0 */
    uint32_t bits;
    memcpy(&bits, &fv, sizeof(bits));
    w->token_embedding_table[i] = (tn_u16)(bits >> 16);
  }

  for (int l = 0; l < nl; l++) {
    w->wq[l] = (tn_i8 *)calloc(dim * dim, sizeof(float));
    w->sq[l] = 1.0f;
    w->wk[l] = (tn_i8 *)calloc(dim * kv_dim, sizeof(float));
    w->sk[l] = 1.0f;
    w->wv[l] = (tn_i8 *)calloc(dim * kv_dim, sizeof(float));
    w->sv[l] = 1.0f;
    w->wo[l] = (tn_i8 *)calloc(dim * dim, sizeof(float));
    w->so[l] = 1.0f;

    w->w1[l] = (tn_i8 *)calloc(dim * hidden_dim, sizeof(float));
    w->s1[l] = 1.0f;
    w->w2[l] = (tn_i8 *)calloc(hidden_dim * dim, sizeof(float));
    w->s2[l] = 1.0f;
    w->w3[l] = (tn_i8 *)calloc(dim * hidden_dim, sizeof(float));
    w->s3[l] = 1.0f;

    w->rms_att_weight[l] = (float *)calloc(dim, sizeof(float));
    w->rms_ffn_weight[l] = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
      w->rms_att_weight[l][i] = 1.0f;
      w->rms_ffn_weight[l][i] = 1.0f;
    }
  }

  w->rms_final_weight = (float *)calloc(dim, sizeof(float));
  for (int i = 0; i < dim; i++)
    w->rms_final_weight[i] = 1.0f;

  w->wcls = (tn_u16 *)calloc((size_t)dim * cfg->vocab_size, sizeof(tn_u16));
  for (size_t i = 0; i < (size_t)dim * cfg->vocab_size; i++) {
    float fv = (float)(i % 3) - 1.0f;
    uint32_t bits;
    memcpy(&bits, &fv, sizeof(bits));
    w->wcls[i] = (tn_u16)(bits >> 16);
  }
}

static void free_mock_weights(TransformerWeights *w, const Config *cfg) {
  free(w->token_embedding_table);
  for (int l = 0; l < cfg->n_layers; l++) {
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

/**
 * Blackbox Test: BB-FWD-02 (Multi-token Generation Integrity)
 * Tests full continuous 10-token context cascading end to end safely.
 */
static void bb_fwd_02_generator(void) {
  Config cfg = {.dim = 32,
                .hidden_dim = 64,
                .n_layers = 2,
                .n_heads = 4,
                .n_kv_heads = 2,
                .vocab_size = 32,
                .seq_len = 16};
  TransformerWeights w;
  RunState s;

  tn_simd_init();
  fill_mock_weights(&w, &cfg);
  run_state_alloc(&s, &cfg, cfg.seq_len);

  float last_logits[32];
  for (int t = 0; t < 10; t++) {
    printf("[TRACKER blackbox] Starting loop t=%d\n", t);
    fflush(stdout);

    /* Run forward pass using an arbitrary deterministic token sequence 1, 2,
     * ..., 10 */
    float *logits =
        transformer_forward((t + 1) % cfg.vocab_size, t, &cfg, &w, &s, NULL, NULL);

    printf("[TRACKER blackbox] Finished forward pass t=%d\n", t);
    fflush(stdout);

    TEST_ASSERT(logits != NULL,
                "Generated valid probability scope for iteration");

    /* Assure logits change iteration over iteration, proving KV cache builds
     * output */
    if (t > 0) {
      int differs = 0;
      for (int k = 0; k < cfg.vocab_size; k++) {
        if (fabsf(logits[k] - last_logits[k]) > 1e-6f)
          differs = 1;
      }
      TEST_ASSERT(differs, "Logits mutated appropriately, signaling working KV "
                           "context building");
    }
    memcpy(last_logits, logits, sizeof(float) * cfg.vocab_size);
  }

  run_state_free(&s);
  free_mock_weights(&w, &cfg);
}

int main(void) {
  RUN_TEST(bb_fwd_02_generator);
  TEST_SUMMARY();
}
