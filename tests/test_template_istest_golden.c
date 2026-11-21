/*
 * tests/test_template_istest_golden.c — the chat-template `is` tests (`x is defined / none / None /
 * string / integer / number / iterable / mapping / odd / even`, plus names that are not tests),
 * plain and negated (`is not`), over operands of every value kind: undefined and defined names,
 * attributes of both, indexes inside and past the end, literals, strings, ints (even, odd,
 * negative, zero), arrays, objects, ranges and none. Each template renders `{{ A is [not] T }}`;
 * the hash covers every rendered result, recorded from the evaluator before eval_isdef was split.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/chat_template.h"

#define EXPECTED_HASH 0x72a2d33d0c163275ull

static const char *const OPERANDS[] = {
    "undefd",      "undefd.a",
    "msg",         "msg.role",
    "ns",          "ns.a",
    "ns.missing",  "messages",
    "messages[0]", "messages[0].role",
    "messages[9]", "messages[9].x",
    "none",        "None",
    "true",        "false",
    "0",           "1",
    "2",           "7",
    "-3",          "-4",
    "'abc'",       "''",
    "range(3)",    "range(0)",
    "namespace()", "'a-b'.split('-')",
    "msg|length",  "undefd|length",
    "x",           "y",
    "y.a",         "loop",
    "1 + 2",       "1 == 1",
    "not x",       "(msg)",
};
static const char *const TESTS[] = {
    "defined", "none", "None", "string", "integer", "number", "iterable",
    "mapping", "odd",  "even", "sameas", "lower",   "foo",    "undefined",
};

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static int g_true, g_total;

static void render_one(const char *a, int negate, const char *test) {
    char tmpl[256];
    snprintf(tmpl, sizeof tmpl,
             "{%% set msg = messages[0].content %%}{%% set x = 1 %%}{%% set y = undefd %%}"
             "{%% set ns = namespace(a=2) %%}{{ %s is %s%s }}",
             a, negate ? "not " : "", test);
    const char *roles[]    = {"user"};
    const char *contents[] = {"hello"};
    char *out              = chat_template_apply(tmpl, roles, contents, 1, "", "", 0);
    int present            = out != NULL;
    mix(&present, sizeof present);
    if (out) mix(out, strlen(out) + 1);
    g_total++;
    g_true += out && strcmp(out, "True") == 0;
    free(out);
}

static void test_istest_golden(void) {
    size_t na = sizeof OPERANDS / sizeof OPERANDS[0], nt = sizeof TESTS / sizeof TESTS[0];
    for (size_t i = 0; i < na; i++)
        for (size_t t = 0; t < nt; t++)
            for (int neg = 0; neg < 2; neg++) render_one(OPERANDS[i], neg, TESTS[t]);
    if (getenv("TEMPLATE_ISTEST_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d of %d render True)\n", (unsigned long long)g_hash, g_true,
               g_total);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "`is` tests match the recorded hash");
}

int main(void) {
    printf("=== chat-template `is` test golden ===\n");
    RUN_TEST(test_istest_golden);
    TEST_SUMMARY();
}
