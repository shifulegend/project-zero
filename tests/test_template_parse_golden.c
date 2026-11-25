/*
 * tests/test_template_parse_golden.c — the chat-template expression parser over 30,000 generated
 * templates: literals (strings, ints, true/True/false/False/none/None), identifiers, parenthesised
 * expressions and any chain of postfixes ([index], [a:b:c] slices with every part optional, .attr,
 * .0, calls with positional and keyword arguments), some cut short or with a closing token dropped
 * so the error-recovery paths run too. Each template is rendered against fixed messages; the hash
 * covers whether rendering succeeded and every output byte, recorded from the parser before
 * parse_primary was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/chat_template.h"

#define EXPECTED_HASH 0x176844c53b838ba0ull
#define N_CASES 30000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x2468ace013579bdfull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const ATOMS[] = {
    "'abc'",
    "''",
    "0",
    "5",
    "-1",
    "true",
    "True",
    "false",
    "False",
    "none",
    "None",
    "msg",
    "msgs",
    "x",
    "ns",
    "roles",
    "range(4)",
    "namespace(a=1, b='z')",
    "'a-b-c'.split('-')",
    "messages",
    "loop",
};
static const char *const POSTFIX[] = {
    "[0]",      "[1]",      "[-1]",  "[2]",   "['a']",    "[x]",         "[::-1]",
    "[1:]",     "[:2]",     "[1:3]", "[::2]", "[:]",      "[-2:]",       "[:-1]",
    "[1:3:1]",  "[::]",     ".a",    ".b",    ".0",       ".1",          ".length",
    ".role",    ".content", ".",     "()",    "(1)",      "(1, 2)",      "(a=1)",
    "(x, k=2)", "('-')",    "(,)",   "(=)",   ".upper()", ".split('-')", ".startswith('a')",
    "[",        "[1",       "(",     "(1",    "[1:",      "[:",
};

static void build(char *out, size_t cap) {
    char expr[200] = "";
    size_t n       = 0;
    int paren      = rnd(5) == 0;
    const char *a  = ATOMS[rnd(sizeof ATOMS / sizeof ATOMS[0])];
    if (paren) n += (size_t)snprintf(expr + n, sizeof expr - n, "(");
    n += (size_t)snprintf(expr + n, sizeof expr - n, "%s", a);
    for (unsigned k = rnd(4); k > 0 && n < 120; k--)
        n += (size_t)snprintf(expr + n, sizeof expr - n, "%s",
                              POSTFIX[rnd(sizeof POSTFIX / sizeof POSTFIX[0])]);
    if (paren && rnd(4) != 0) n += (size_t)snprintf(expr + n, sizeof expr - n, ")");
    if (rnd(6) == 0) expr[rnd((unsigned)n + 1)] = '\0'; /* cut short */
    snprintf(out, cap,
             "{%% set msg = messages[0].content %%}{%% set msgs = messages %%}"
             "{%% set x = 1 %%}{%% set ns = namespace(a=2) %%}{%% set roles = ['u'] %%}"
             "{{ %s }}|{{ %s | length }}",
             expr, expr);
}

static void test_parse_golden(void) {
    const char *roles[]    = {"user", "assistant"};
    const char *contents[] = {"hello", "world"};
    char tmpl[800];
    int n_ok = 0, n_text = 0;
    for (int t = 0; t < N_CASES; t++) {
        build(tmpl, sizeof tmpl);
        char *out   = chat_template_apply(tmpl, roles, contents, 2, "<s>", "</s>", 1);
        int present = out != NULL;
        mix(&present, sizeof present);
        n_ok += present;
        n_text += present && strlen(out) > 1;
        if (out) mix(out, strlen(out) + 1);
        free(out);
    }
    if (getenv("TEMPLATE_PARSE_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d rendered, %d with text)\n", (unsigned long long)g_hash, n_ok,
               n_text);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "parsing matches the recorded hash");
}

int main(void) {
    printf("=== chat-template parser golden ===\n");
    RUN_TEST(test_parse_golden);
    TEST_SUMMARY();
}
