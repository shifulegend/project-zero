/*
 * test_speculative.c — Phase 18 (speculative decoding) Stage 5:
 * accept_reject_round()'s decision logic, and end-to-end equivalence
 * between speculative_generate_with_callback() and plain
 * generate_with_callback() in greedy mode.
 */
#include "test_harness.h"
#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/unpack.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "speculative/accept_reject.h"
#include "speculative/draft_model.h"
#include "speculative/spec_decode.h"
#include "tokenizer/tokenizer.h"
#include "transformer/generate.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SPEC_VOCAB 6

/* ── accept_reject_round() unit tests ────────────────────────────────────
 * Directly construct verify_check_logits / bonus_logits so each branch
 * (full acceptance, rejection at a given position) is exercised
 * deterministically, rather than hoping random model weights happen to
 * produce these cases at the full-pipeline level. */

static void set_one_hot(float *logits, int vocab, int winner) {
    for (int i = 0; i < vocab; i++) logits[i] = (i == winner) ? 10.0f : 0.0f;
}

static void test_accept_reject_all_accepted(void) {
    const int spec_length = 4;
    int draft_tokens[4] = {2, 0, 3, 1};
    float verify_check_logits[4][SPEC_VOCAB];
    float bonus_logits[SPEC_VOCAB];

    /* Row i's argmax matches draft_tokens[i] exactly -- every token accepted. */
    for (int i = 0; i < spec_length; i++) {
        set_one_hot(verify_check_logits[i], SPEC_VOCAB, draft_tokens[i]);
    }
    set_one_hot(bonus_logits, SPEC_VOCAB, 5);

    unsigned long long rng = 1;
    int n_accept = -1, emit_token = -1;
    accept_reject_round(draft_tokens, NULL, &verify_check_logits[0][0], bonus_logits,
                         spec_length, SPEC_VOCAB, /*temperature=*/0.0f, &rng,
                         &n_accept, &emit_token);

    TEST_ASSERT_EQ(n_accept, spec_length, "all draft tokens accepted when they match the verifier");
    TEST_ASSERT_EQ(emit_token, 5, "bonus token is argmax(bonus_logits)");
}

static void test_accept_reject_rejects_at_first_mismatch(void) {
    const int spec_length = 5;
    int draft_tokens[5] = {0, 1, 2, 3, 4};
    float verify_check_logits[5][SPEC_VOCAB];
    float bonus_logits[SPEC_VOCAB];

    /* Positions 0,1 match the draft; position 2 disagrees (verifier wants 5
     * instead of the drafted 2); positions after a rejection must never be
     * consulted for acceptance (checked implicitly: n_accept stops at 2). */
    set_one_hot(verify_check_logits[0], SPEC_VOCAB, 0);
    set_one_hot(verify_check_logits[1], SPEC_VOCAB, 1);
    set_one_hot(verify_check_logits[2], SPEC_VOCAB, 5);
    set_one_hot(verify_check_logits[3], SPEC_VOCAB, 3);
    set_one_hot(verify_check_logits[4], SPEC_VOCAB, 4);
    set_one_hot(bonus_logits, SPEC_VOCAB, 0);

    unsigned long long rng = 1;
    int n_accept = -1, emit_token = -1;
    accept_reject_round(draft_tokens, NULL, &verify_check_logits[0][0], bonus_logits,
                         spec_length, SPEC_VOCAB, /*temperature=*/0.0f, &rng,
                         &n_accept, &emit_token);

    TEST_ASSERT_EQ(n_accept, 2, "acceptance stops at the first mismatch");
    TEST_ASSERT_EQ(emit_token, 5, "emitted token is the verifier's own pick at the mismatch position");
}

static void test_accept_reject_rejects_immediately(void) {
    const int spec_length = 3;
    int draft_tokens[3] = {1, 1, 1};
    float verify_check_logits[3][SPEC_VOCAB];
    float bonus_logits[SPEC_VOCAB];

    set_one_hot(verify_check_logits[0], SPEC_VOCAB, 4); /* disagrees immediately */
    set_one_hot(verify_check_logits[1], SPEC_VOCAB, 1);
    set_one_hot(verify_check_logits[2], SPEC_VOCAB, 1);
    set_one_hot(bonus_logits, SPEC_VOCAB, 0);

    unsigned long long rng = 1;
    int n_accept = -1, emit_token = -1;
    accept_reject_round(draft_tokens, NULL, &verify_check_logits[0][0], bonus_logits,
                         spec_length, SPEC_VOCAB, 0.0f, &rng, &n_accept, &emit_token);

    TEST_ASSERT_EQ(n_accept, 0, "zero tokens accepted when position 0 disagrees");
    TEST_ASSERT_EQ(emit_token, 4, "emitted token is the verifier's argmax at position 0");
}

/* ── Full-pipeline greedy equivalence ─────────────────────────────────────
 * Because transformer_forward_batch() is teacher-forcing-equivalent to
 * sequential transformer_forward() calls, greedy speculative_generate()
 * output must be byte-identical to plain generate()'s, by construction.
 * Uses a tiny synthetic byte-level tokenizer (ASCII printable characters
 * map to themselves under this project's GPT-2-style byte-BPE, so a vocab
 * containing exactly those single-character strings needs no merges) and
 * two independently-seeded tiny ternary models so the draft's own
 * predictions genuinely differ from the verifier's (exercising real
 * accept/reject decisions, not just always-agree). */

static void fill_ternary_i8(tn_i8 *buf, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        int r = (int)((seed >> 16) % 3);
        buf[i] = (tn_i8)(r - 1);
    }
}

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

static void fill_pattern_bf16(tn_u16 *buf, size_t count, unsigned seed) {
    for (size_t i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        int v = (int)((seed >> 16) % 3) - 1;
        float f = (float)v * 0.5f;
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        buf[i] = (tn_u16)(bits >> 16);
    }
}

static void fill_ones(float *buf, int count) {
    for (int i = 0; i < count; i++) buf[i] = 1.0f;
}

#define SM_DIM 16
#define SM_HIDDEN 32
#define SM_LAYERS 1
#define SM_HEADS 2
#define SM_KV_HEADS 2
#define SM_SEQ 32

static void build_spec_model(Config *cfg, TransformerWeights *w, unsigned seed_base) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->dim = SM_DIM;
    cfg->hidden_dim = SM_HIDDEN;
    cfg->n_layers = SM_LAYERS;
    cfg->n_heads = SM_HEADS;
    cfg->n_kv_heads = SM_KV_HEADS;
    cfg->vocab_size = SPEC_VOCAB;
    cfg->seq_len = SM_SEQ;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_freq_scale = 1.0f;
    cfg->rope_theta = 10000.0f;
    cfg->rope_yarn_attn_factor = 1.0f;
    cfg->bos_token_id = -1;  /* legacy path uses tok->bos_token_id instead */
    cfg->eos_token_id = -1;  /* never fires -- keeps the test's token count exact */

    memset(w, 0, sizeof(*w));
    weights_alloc_pointers(w, cfg);
    w->layers_are_ternary = true;

    int kv_dim = config_kv_dim(cfg);

    w->token_embedding_table = (tn_u16 *)calloc((size_t)SPEC_VOCAB * SM_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->token_embedding_table, (size_t)SPEC_VOCAB * SM_DIM, seed_base + 1);

    for (int l = 0; l < SM_LAYERS; l++) {
        w->wq[l] = (tn_i8 *)make_packed_ternary(SM_DIM, SM_DIM, seed_base + 10 + l);
        w->sq[l] = 1.0f;
        w->wk[l] = (tn_i8 *)make_packed_ternary(kv_dim, SM_DIM, seed_base + 20 + l);
        w->sk[l] = 1.0f;
        w->wv[l] = (tn_i8 *)make_packed_ternary(kv_dim, SM_DIM, seed_base + 30 + l);
        w->sv[l] = 1.0f;
        w->wo[l] = (tn_i8 *)make_packed_ternary(SM_DIM, SM_DIM, seed_base + 40 + l);
        w->so[l] = 1.0f;

        w->w1[l] = (tn_i8 *)make_packed_ternary(SM_HIDDEN, SM_DIM, seed_base + 50 + l);
        w->s1[l] = 1.0f;
        w->w3[l] = (tn_i8 *)make_packed_ternary(SM_HIDDEN, SM_DIM, seed_base + 60 + l);
        w->s3[l] = 1.0f;
        w->w2[l] = (tn_i8 *)make_packed_ternary(SM_DIM, SM_HIDDEN, seed_base + 70 + l);
        w->s2[l] = 1.0f;

        w->rms_att_weight[l] = (float *)malloc(SM_DIM * sizeof(float));
        fill_ones(w->rms_att_weight[l], SM_DIM);
        w->rms_ffn_weight[l] = (float *)malloc(SM_DIM * sizeof(float));
        fill_ones(w->rms_ffn_weight[l], SM_DIM);
    }

    w->rms_final_weight = (float *)malloc(SM_DIM * sizeof(float));
    fill_ones(w->rms_final_weight, SM_DIM);

    w->wcls = (tn_u16 *)calloc((size_t)SPEC_VOCAB * SM_DIM, sizeof(tn_u16));
    fill_pattern_bf16(w->wcls, (size_t)SPEC_VOCAB * SM_DIM, seed_base + 99);
}

static void free_spec_model(TransformerWeights *w) {
    free(w->token_embedding_table);
    for (int l = 0; l < SM_LAYERS; l++) {
        free(w->wq[l]); free(w->wk[l]); free(w->wv[l]); free(w->wo[l]);
        free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
        free(w->rms_att_weight[l]); free(w->rms_ffn_weight[l]);
    }
    free(w->rms_final_weight);
    free(w->wcls);
    weights_free_pointers(w);
}

/* ASCII printable bytes 33-126 map to themselves under this project's
 * GPT-2-style byte-BPE (see tokenizer_encode.c's g_byte_unicode table) --
 * vocab entries are exactly the single characters used in the test prompt,
 * so byte-level pre-tokenization already produces the final token IDs with
 * no merges possible (no multi-character vocab entry exists to merge into). */
static void build_test_tokenizer(Tokenizer *t) {
    memset(t, 0, sizeof(*t));
    t->vocab_size = SPEC_VOCAB;
    t->vocab = (char **)malloc((size_t)SPEC_VOCAB * sizeof(char *));
    t->vocab_scores = (float *)calloc((size_t)SPEC_VOCAB, sizeof(float));
    const char *strs[SPEC_VOCAB] = { "\x01BOS\x01", "A", "B", "C", "D", "\x02EOS\x02" };
    for (int i = 0; i < SPEC_VOCAB; i++) {
        t->vocab[i] = strdup(strs[i]);
    }
    t->max_token_len = 8;

    t->sorted_vocab_indices = (int *)malloc((size_t)SPEC_VOCAB * sizeof(int));
    for (int i = 0; i < SPEC_VOCAB; i++) t->sorted_vocab_indices[i] = i;
    /* Insertion sort by vocab string -- SPEC_VOCAB is tiny, no need for
     * anything fancier (tokenizer_encode's binary search requires this
     * index to be sorted by strcmp(vocab[idx], ...)). */
    for (int i = 1; i < SPEC_VOCAB; i++) {
        int key = t->sorted_vocab_indices[i];
        int j = i - 1;
        while (j >= 0 && strcmp(t->vocab[t->sorted_vocab_indices[j]], t->vocab[key]) > 0) {
            t->sorted_vocab_indices[j + 1] = t->sorted_vocab_indices[j];
            j--;
        }
        t->sorted_vocab_indices[j + 1] = key;
    }
    t->sorted = 1;

    t->chat_template = NULL; /* legacy prompt path: manual BOS + raw encode */
    t->bos_token_id = 0;
    t->eos_token_id = -1;
    t->n_eos = 0;
}

typedef struct {
    char buf[256];
    int len;
} CaptureCtx;

static int capture_callback(const char *piece, void *userdata) {
    CaptureCtx *c = (CaptureCtx *)userdata;
    int plen = (int)strlen(piece);
    if (c->len + plen < (int)sizeof(c->buf)) {
        memcpy(c->buf + c->len, piece, (size_t)plen);
        c->len += plen;
        c->buf[c->len] = '\0';
    }
    return 0;
}

static void test_speculative_greedy_matches_plain_generation(void) {
    tn_simd_init();

    Config verifier_cfg, draft_cfg;
    TransformerWeights verifier_w, draft_w;
    build_spec_model(&verifier_cfg, &verifier_w, /*seed_base=*/1000);
    build_spec_model(&draft_cfg, &draft_w, /*seed_base=*/9000); /* different weights */

    Tokenizer tok;
    build_test_tokenizer(&tok);

    MoEConfig mc;
    moe_config_init_dense(&mc);

    const char *prompt = "AB";
    const int max_tokens = 6;
    const int spec_length = 3;

    /* ── Baseline: plain greedy generation ────────────────────────────── */
    RunState s_plain;
    TEST_ASSERT(run_state_alloc(&s_plain, &verifier_cfg, verifier_cfg.seq_len) == TN_OK,
                "plain run_state_alloc");
    CaptureCtx plain_out; plain_out.len = 0; plain_out.buf[0] = '\0';
    generate_with_callback(&verifier_cfg, &verifier_w, &s_plain, &mc, &tok, NULL, prompt,
                            max_tokens, /*temperature=*/0.0f, /*top_p=*/1.0f, /*json_mode=*/false,
                            capture_callback, &plain_out);
    run_state_free(&s_plain);

    /* ── Speculative: same verifier weights, a differently-seeded draft ── */
    RunState s_spec;
    TEST_ASSERT(run_state_alloc(&s_spec, &verifier_cfg, verifier_cfg.seq_len) == TN_OK,
                "spec run_state_alloc");
    RunState s_draft;
    TEST_ASSERT(run_state_alloc(&s_draft, &draft_cfg, draft_cfg.seq_len) == TN_OK,
                "draft run_state_alloc");

    DraftModel draft;
    memset(&draft, 0, sizeof(draft));
    draft.config = draft_cfg;
    draft.weights = draft_w;
    moe_config_init_dense(&draft.moe_config);
    draft.state = &s_draft;

    CaptureCtx spec_out; spec_out.len = 0; spec_out.buf[0] = '\0';
    speculative_generate_with_callback(&verifier_cfg, &verifier_w, &s_spec, &mc, &tok, NULL,
                                        prompt, max_tokens, /*temperature=*/0.0f, /*top_p=*/1.0f,
                                        /*json_mode=*/false, &draft, spec_length,
                                        capture_callback, &spec_out);
    run_state_free(&s_spec);
    run_state_free(&s_draft);

    TEST_ASSERT(plain_out.len > 0, "plain generation produced output");
    TEST_ASSERT(strcmp(plain_out.buf, spec_out.buf) == 0,
                "greedy speculative output is byte-identical to plain greedy generation");

    tokenizer_free(&tok);
    free_spec_model(&verifier_w);
    free_spec_model(&draft_w);
}

/* ── draft_model_load() vocab-mismatch validation ────────────────────────
 * Uses the project's real demo model (models/smollm2.gguf, downloaded by
 * `make demo`) as a real, valid GGUF file to load as a "draft" against a
 * deliberately wrong vocab_size -- avoids needing a synthetic-GGUF-writer
 * test helper just for this one validation check. Skips gracefully (per
 * this project's asset-dependent-test convention) when the file is absent,
 * e.g. in a fresh checkout that hasn't run `make demo` yet. */
static void test_draft_model_vocab_mismatch_rejected(void) {
    const char *path = "models/smollm2.gguf";
    if (access(path, F_OK) != 0) {
        printf("  (skipped: %s not present -- run `make demo` to fetch it)\n", path);
        return;
    }

    DraftModel dm;
    TernaryError err = draft_model_load(&dm, path, /*verifier_vocab_size=*/1, /*max_seq_len_hint=*/64, NULL);
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_ARGS,
                   "draft model load is rejected when vocab_size doesn't match the verifier");
}

int main(void) {
    RUN_TEST(test_accept_reject_all_accepted);
    RUN_TEST(test_accept_reject_rejects_at_first_mismatch);
    RUN_TEST(test_accept_reject_rejects_immediately);
    RUN_TEST(test_speculative_greedy_matches_plain_generation);
    RUN_TEST(test_draft_model_vocab_mismatch_rejected);
    TEST_SUMMARY();
}
