/*
 * test_fsm.c — Phase 20: grammar-aware token masking + constrained sampling.
 *
 * Builds a tiny synthetic Tokenizer directly (no file I/O -- tokenizer_decode
 * only touches t->vocab/t->bos_token_id, and mask/advance only additionally
 * touch t->eos_token_id/t->eos_list/t->n_eos, so a hand-built struct with
 * just those fields set is a faithful, minimal fixture).
 */
#include "sampling/fsm.h"
#include "sampling/constrained_sample.h"
#include "test_harness.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

enum {
    TOK_LBRACE = 0, TOK_RBRACE, TOK_QUOTE, TOK_A, TOK_COLON,
    TOK_ONE, TOK_COMMA, TOK_SPACE, TOK_EOS, TOK_VOCAB_SIZE
};

static Tokenizer *make_tiny_tokenizer(void) {
    static const char *pieces[TOK_VOCAB_SIZE] = {
        "{", "}", "\"", "a", ":", "1", ",", " ", "<eos>"
    };
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->vocab = (char **)calloc(TOK_VOCAB_SIZE, sizeof(char *));
    for (int i = 0; i < TOK_VOCAB_SIZE; i++) t->vocab[i] = (char *)pieces[i];
    t->vocab_size = TOK_VOCAB_SIZE;
    t->bos_token_id = -1;
    t->eos_token_id = TOK_EOS;
    t->eos_list[0] = TOK_EOS;
    t->n_eos = 1;
    return t;
}

static void free_tiny_tokenizer(Tokenizer *t) {
    free(t->vocab);
    free(t);
}

#define MASKED(logits, i) ((logits)[i] < -1e29f)

static void test_initial_mask_allows_value_starts_only(void) {
    Tokenizer *t = make_tiny_tokenizer();
    FSMState fsm;
    fsm_init(&fsm);

    float logits[TOK_VOCAB_SIZE];
    for (int i = 0; i < TOK_VOCAB_SIZE; i++) logits[i] = 0.0f;
    fsm_compute_token_mask(&fsm, t, -1, logits, TOK_VOCAB_SIZE);

    TEST_ASSERT(!MASKED(logits, TOK_LBRACE), "'{' allowed as a value start");
    TEST_ASSERT(!MASKED(logits, TOK_QUOTE),  "'\"' allowed as a value start");
    TEST_ASSERT(!MASKED(logits, TOK_ONE),    "'1' allowed as a value start");
    TEST_ASSERT(!MASKED(logits, TOK_SPACE),  "leading whitespace allowed");
    TEST_ASSERT(MASKED(logits, TOK_RBRACE),  "'}' NOT allowed before any '{'");
    TEST_ASSERT(MASKED(logits, TOK_A),       "'a' NOT a valid value start");
    TEST_ASSERT(MASKED(logits, TOK_COLON),   "':' NOT a valid value start");
    TEST_ASSERT(MASKED(logits, TOK_COMMA),   "',' NOT a valid value start");
    TEST_ASSERT(MASKED(logits, TOK_EOS),     "EOS NOT allowed before any content");

    free_tiny_tokenizer(t);
}

static void test_mask_after_open_brace(void) {
    Tokenizer *t = make_tiny_tokenizer();
    FSMState fsm;
    fsm_init(&fsm);
    fsm_advance(&fsm, t, -1, TOK_LBRACE);

    float logits[TOK_VOCAB_SIZE];
    for (int i = 0; i < TOK_VOCAB_SIZE; i++) logits[i] = 0.0f;
    fsm_compute_token_mask(&fsm, t, TOK_LBRACE, logits, TOK_VOCAB_SIZE);

    TEST_ASSERT(!MASKED(logits, TOK_QUOTE),  "'\"' allowed to start an object key");
    TEST_ASSERT(!MASKED(logits, TOK_RBRACE), "'}' allowed to close an empty object");
    TEST_ASSERT(MASKED(logits, TOK_ONE),     "digit NOT allowed right after '{'");
    TEST_ASSERT(MASKED(logits, TOK_COLON),   "':' NOT allowed right after '{'");
    TEST_ASSERT(MASKED(logits, TOK_EOS),     "EOS NOT allowed mid-object");

    free_tiny_tokenizer(t);
}

static void test_full_object_via_constrained_sample(void) {
    /* Drive the grammar through {"a":1} one forced token at a time using
     * sample_constrained itself (greedy, temperature=0), checking at each
     * step that the token we *want* to feed was actually left unmasked
     * (a masked token would have logit -1e30 and argmax would never pick
     * it if any other token had a higher, unmasked logit -- so this test
     * gives every "correct" token an artificially high logit and confirms
     * sample_constrained picks exactly it, never a grammar-illegal one). */
    Tokenizer *t = make_tiny_tokenizer();
    FSMState fsm;
    fsm_init(&fsm);
    unsigned long long rng = 12345;

    int expected[] = { TOK_LBRACE, TOK_QUOTE, TOK_A, TOK_QUOTE, TOK_COLON, TOK_ONE, TOK_RBRACE };
    int prev = -1;
    for (size_t step = 0; step < sizeof(expected) / sizeof(expected[0]); step++) {
        float logits[TOK_VOCAB_SIZE];
        for (int i = 0; i < TOK_VOCAB_SIZE; i++) logits[i] = 0.0f;
        logits[expected[step]] = 100.0f; /* bias strongly toward the "intended" token */
        int chosen = sample_constrained(logits, TOK_VOCAB_SIZE, &fsm, t, prev, 0.0f, 1.0f, &rng);
        TEST_ASSERT(chosen == expected[step], "constrained sample picked the intended grammar-legal token");
        prev = chosen;
    }
    TEST_ASSERT(json_grammar_is_complete(&fsm.json), "grammar reports DONE after {\"a\":1}");

    /* Now EOS should be unmasked. */
    float logits[TOK_VOCAB_SIZE];
    for (int i = 0; i < TOK_VOCAB_SIZE; i++) logits[i] = 0.0f;
    fsm_compute_token_mask(&fsm, t, prev, logits, TOK_VOCAB_SIZE);
    TEST_ASSERT(!MASKED(logits, TOK_EOS), "EOS allowed once the JSON document is complete");

    free_tiny_tokenizer(t);
}

static void test_constrained_sample_never_picks_masked_token(void) {
    /* Even when a grammar-illegal token has the highest raw logit,
     * sample_constrained (greedy) must never pick it. */
    Tokenizer *t = make_tiny_tokenizer();
    FSMState fsm;
    fsm_init(&fsm);
    unsigned long long rng = 42;

    float logits[TOK_VOCAB_SIZE];
    for (int i = 0; i < TOK_VOCAB_SIZE; i++) logits[i] = 0.0f;
    logits[TOK_RBRACE] = 1000.0f; /* illegal at the very start -- highest logit anyway */
    logits[TOK_LBRACE] = 1.0f;    /* legal, much lower logit */

    int chosen = sample_constrained(logits, TOK_VOCAB_SIZE, &fsm, t, -1, 0.0f, 1.0f, &rng);
    TEST_ASSERT(chosen != TOK_RBRACE, "grammar-illegal token never chosen even with the highest raw logit");
    TEST_ASSERT(chosen == TOK_LBRACE, "falls back to the highest-logit LEGAL token");

    free_tiny_tokenizer(t);
}

int main(void) {
    RUN_TEST(test_initial_mask_allows_value_starts_only);
    RUN_TEST(test_mask_after_open_brace);
    RUN_TEST(test_full_object_via_constrained_sample);
    RUN_TEST(test_constrained_sample_never_picks_masked_token);
    TEST_SUMMARY();
}
