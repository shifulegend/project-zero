/*
 * tests/test_template_guard.c — known-broken chat templates (2026-11-16).
 *
 * The official BitNet b1.58 2B-4T GGUF ships a template that appends
 * {{ eos_token }} after its "BITNETAssistant: " prefix. The prompt then ends in
 * EOS, the model predicts BOS and generation stops at 0 tokens (a silent empty
 * answer). template_guard.c replaces that template, matched by metadata
 * (architecture + template text), and flags any rendered prompt that ends in
 * EOS. No model file is needed: the check is on rendered text.
 */
#include "test_harness.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/template_guard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tokenizer.chat_template of bitnet-b1.58-2B-4T-gguf (ggml-model-i2_s.gguf),
 * read from the file on 2026-11-16. */
static const char *BROKEN_BITNET =
    "{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'] + '\\n\\nBITNETAssistant: "
    "' + eos_token }}"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% "
    "endfor %}";

#define BOS "<|begin_of_text|>"
#define EOS "<|end_of_text|>"

static char *render(const char *tmpl, int add_gen) {
    const char *roles[]    = {"user"};
    const char *contents[] = {"Say hi."};
    return chat_template_apply(tmpl, roles, contents, 1, BOS, EOS, add_gen);
}

static void test_known_fix_matches_bitnet(void) {
    const char *fix = tn_template_known_fix("bitnet-b1.58", BROKEN_BITNET);
    TEST_ASSERT(fix != NULL, "the broken BitNet template is recognised");
    TEST_ASSERT(fix == tn_template_bitnet_b158(), "and replaced by the built-in BitNet template");
}

static void test_known_fix_leaves_others_alone(void) {
    TEST_ASSERT(tn_template_known_fix("llama", BROKEN_BITNET) == NULL,
                "another architecture is never touched");
    TEST_ASSERT(tn_template_known_fix("bitnet-b1.58", tn_template_bitnet_b158()) == NULL,
                "an already-correct BitNet template is not replaced");
    TEST_ASSERT(tn_template_known_fix("bitnet-b1.58", "{{ messages[0].content }}") == NULL,
                "a custom template without the marker is not replaced");
    TEST_ASSERT(tn_template_known_fix("bitnet-b1.58", "Human: x\n\nBITNETAssistant: ") == NULL,
                "the marker without an eos_token after it is not replaced");
    TEST_ASSERT(tn_template_known_fix(NULL, BROKEN_BITNET) == NULL, "NULL arch");
    TEST_ASSERT(tn_template_known_fix("bitnet-b1.58", NULL) == NULL, "NULL template");
}

static void test_ends_with_token(void) {
    TEST_ASSERT(tn_prompt_ends_with_token("a" EOS, EOS), "ends with the token");
    TEST_ASSERT(tn_prompt_ends_with_token("a" EOS " \n\t", EOS), "trailing whitespace is ignored");
    TEST_ASSERT(!tn_prompt_ends_with_token("a" EOS "b", EOS), "token not at the end");
    TEST_ASSERT(!tn_prompt_ends_with_token("Assistant: ", EOS), "no token");
    TEST_ASSERT(!tn_prompt_ends_with_token("a", ""), "empty token never matches");
    TEST_ASSERT(!tn_prompt_ends_with_token(NULL, EOS), "NULL prompt");
    TEST_ASSERT(!tn_prompt_ends_with_token("a", NULL), "NULL token");
}

static void test_render_broken_vs_builtin(void) {
    char *bad = render(BROKEN_BITNET, 1);
    TEST_ASSERT(bad != NULL, "broken template renders");
    TEST_ASSERT(bad && strcmp(bad, BOS "Human: Say hi.\n\nBITNETAssistant: " EOS) == 0,
                "the broken template's prompt is exactly what the model saw");
    TEST_ASSERT(tn_prompt_ends_with_token(bad, EOS), "and it ends in EOS (the bug)");
    free(bad);

    char *good = render(tn_template_bitnet_b158(), 1);
    TEST_ASSERT(good != NULL, "built-in template renders");
    TEST_ASSERT(good && strcmp(good, BOS "User: Say hi.<|eot_id|>Assistant: ") == 0,
                "the built-in prompt is the model card's format");
    TEST_ASSERT(!tn_prompt_ends_with_token(good, EOS), "and does not end in EOS");
    free(good);
}

/* The embedded copy must equal docs/templates/bitnet-b1.58.jinja (the file
 * users may pass with --chat-template-file). Skipped when run outside the repo. */
static void test_embedded_equals_docs_file(void) {
    FILE *f = fopen("docs/templates/bitnet-b1.58.jinja", "rb");
    if (!f) {
        printf("docs/templates/bitnet-b1.58.jinja not found from here -- skipping\n");
        return;
    }
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    TEST_ASSERT(n > 0 && n < sizeof(buf) - 1, "docs template read");
    TEST_ASSERT(strcmp(buf, tn_template_bitnet_b158()) == 0,
                "embedded template == docs/templates/bitnet-b1.58.jinja");
}

int main(void) {
    RUN_TEST(test_known_fix_matches_bitnet);
    RUN_TEST(test_known_fix_leaves_others_alone);
    RUN_TEST(test_ends_with_token);
    RUN_TEST(test_render_broken_vs_builtin);
    RUN_TEST(test_embedded_equals_docs_file);
    TEST_SUMMARY();
    return tn_tests_failed ? 1 : 0;
}
