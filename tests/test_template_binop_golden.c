/*
 * tests/test_template_binop_golden.c — every binary operator of the chat-template language
 * (+ ~ - * / % == != < <= > >= in not in) applied to every pair of operands of every value kind
 * (ints incl. 0 and negatives, strings, none, booleans, an undefined name, objects, arrays,
 * range results). Each template renders `{{ (A) OP (B) }}`; the hash covers the rendered text (or
 * NULL) of all of them, recorded from the evaluator before binop_apply was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/chat_template.h"

#define EXPECTED_HASH 0x9d0efdf26e1a0e14ull

static const char *const OPERANDS[] = {
    "0",     "1",      "7",        "-3",       "'ab'",
    "''",    "'b'",    "'1'",      "none",     "true",
    "false", "undefd", "range(3)", "range(0)", "namespace()",
    "'a b'", "[1, 2]", "[]",       "msg",      "'a b'.split(' ')",
};
static const char *const OPS[] = {
    "+", "~", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "in", "not in"};

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static void render_one(const char *a, const char *op, const char *b) {
    char tmpl[256];
    snprintf(tmpl, sizeof tmpl, "{%% set msg = messages[0].content %%}{{ (%s) %s (%s) }}", a, op,
             b);
    const char *roles[]    = {"user"};
    const char *contents[] = {"hello"};
    char *out              = chat_template_apply(tmpl, roles, contents, 1, "", "", 0);
    int present            = out != NULL;
    mix(&present, sizeof present);
    if (out) mix(out, strlen(out) + 1);
    free(out);
}

static void test_binop_golden(void) {
    size_t na = sizeof OPERANDS / sizeof OPERANDS[0], no = sizeof OPS / sizeof OPS[0];
    for (size_t o = 0; o < no; o++)
        for (size_t i = 0; i < na; i++)
            for (size_t j = 0; j < na; j++) render_one(OPERANDS[i], OPS[o], OPERANDS[j]);
    if (getenv("TEMPLATE_BINOP_GOLDEN_PRINT"))
        printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "operator results match the recorded hash");
}

int main(void) {
    printf("=== chat-template binary operator golden ===\n");
    RUN_TEST(test_binop_golden);
    TEST_SUMMARY();
}
