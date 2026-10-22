/*
 * tests/test_chat_stops.c — chat-mode end of turn (2026-10-22): a generated
 * CONTROL token the active chat template never writes ends the turn
 * (Falcon3-1.58bit ends turns with ">>UNUSED_1<<" though its metadata says
 * "<|endoftext|>"); template-used control tokens and USER_DEFINED tokens stay
 * generatable; raw prompts keep only the declared EOG set.
 *
 * Synthetic tokenizer: tokenizer_set_chat_stops/tokenizer_is_eog only read
 * vocab_size, the special list, BOS/EOS ids and eos_list.
 */
#include "test_harness.h"
#include "tokenizer/tokenizer.h"
#include <stdlib.h>
#include <string.h>

enum { T_HELLO, T_EOT, T_UNUSED1, T_USER, T_PYTAG, T_THINK, T_BOS, T_N };

static Tokenizer make(void) {
    static char *vocab[T_N] = { "hello", "<|endoftext|>", ">>UNUSED_1<<", "<|user|>",
                                "<|python_tag|>", "<think>", "<s>" };
    static char *spec[] = { "<|endoftext|>", ">>UNUSED_1<<", "<|user|>", "<|python_tag|>", "<think>", "<s>" };
    static int ids[] = { T_EOT, T_UNUSED1, T_USER, T_PYTAG, T_THINK, T_BOS };
    static unsigned char ctrl[] = { 1, 1, 1, 1, 0 /* USER_DEFINED */, 1 };
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    t.vocab = vocab;
    t.vocab_size = T_N;
    t.special_tokens = spec;
    t.special_token_ids = ids;
    t.special_is_control = ctrl;
    t.n_special = 6;
    t.eos_token_id = T_EOT;
    t.bos_token_id = T_BOS;
    t.eos_list[0] = T_EOT;
    t.n_eos = 1;
    return t;
}

static void test_raw_mode_declared_only(void) {
    Tokenizer t = make();
    tokenizer_set_chat_stops(&t, NULL);
    TEST_ASSERT(tokenizer_is_eog(&t, T_EOT), "declared EOS stops");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_UNUSED1), "raw prompt: an undeclared control token does not stop");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_HELLO), "text never stops");
}

static void test_chat_mode_turn_ends(void) {
    Tokenizer t = make();
    const char *tmpl = "{% for m in messages %}<|user|>\n{{ m.content }}{% endfor %}"
                       "{% if tools %}<|python_tag|>{% endif %}";
    tokenizer_set_chat_stops(&t, tmpl);
    TEST_ASSERT(tokenizer_is_eog(&t, T_UNUSED1), "control token the template never writes ends the turn");
    TEST_ASSERT(tokenizer_is_eog(&t, T_BOS), "generating BOS (a new sequence) ends the turn");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_PYTAG), "control token the template writes stays generatable");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_USER), "template turn marker is not added (EOG set decides)");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_THINK), "USER_DEFINED token (<think>) never ends the turn");
    TEST_ASSERT(!tokenizer_is_eog(&t, T_HELLO), "text never stops");
    TEST_ASSERT(tokenizer_is_eog(&t, T_EOT), "declared EOS still stops");
    /* A later raw generation must not inherit the chat stops. */
    tokenizer_set_chat_stops(&t, NULL);
    TEST_ASSERT(!tokenizer_is_eog(&t, T_UNUSED1), "reset for raw mode");
    TEST_ASSERT(!tokenizer_is_eog(&t, -1) && !tokenizer_is_eog(&t, T_N + 5), "out-of-range ids are safe");
}

int main(void) {
    printf("=== chat-mode end of turn from CONTROL tokens ===\n");
    RUN_TEST(test_raw_mode_declared_only);
    RUN_TEST(test_chat_mode_turn_ends);
    TEST_SUMMARY();
}
