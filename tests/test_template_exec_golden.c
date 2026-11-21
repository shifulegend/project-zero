/*
 * tests/test_template_exec_golden.c — the chat-template statement executor over 40,000 generated
 * templates built by nesting text, {{ }} output, for loops (single and tuple targets, over the
 * messages, a range and a split string, with every loop.* field), if / elif / else chains, set
 * (plain, namespace attribute, nested attribute on a missing base) and macros. Each is rendered
 * against fixed messages; the hash covers whether rendering succeeded and every output byte,
 * recorded from the executor before jinja_exec was split into one function per statement.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/chat_template.h"

#define EXPECTED_HASH 0xc8b3ddf1c87ff5deull
#define N_CASES 40000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x13579bdf2468ace0ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const LEAVES[] = {
    "t",
    "{{ loop.index }}",
    "{{ loop.index0 }}{{ loop.first }}{{ loop.last }}",
    "{{ loop.length }}{{ loop.revindex }}{{ loop.revindex0 }}",
    "{{ loop.previtem }}|{{ loop.nextitem }}",
    "{{ x }}",
    "{{ k }}{{ v }}",
    "{{ ns.a }}",
    "{{ ns.a.b }}",
    "{{ m.role }}",
    "{{ m.content }}",
    "{{ greet('w') }}",
    "{{ none }}",
    "\n",
    "{% set a = 7 %}{{ a }}",
    "{% set ns.a = loop.index %}",
    "{% set ns.a.b = 5 %}",
    "{% set missing.a = 1 %}{{ missing }}",
    "{% set ns.n = namespace(z=3) %}{{ ns.n.z }}",
    "{% set ns.n = namespace(z=3) %}{% set ns.n.z = 9 %}{{ ns.n.z }}",
    "{% set ns.n = namespace(z=3, y=namespace(w=1)) %}{% set ns.n.y.w = 8 %}{{ ns.n.y.w }}",
    "{{ loop.previtem.role }}{{ loop.nextitem.content }}",
    "{% macro dup(q) %}[{{ q }}]{% endmacro %}{{ dup(1) }}",
};
static const char *const LISTS[] = {
    "messages",
    "range(3)",
    "range(0)",
    "'a-b'.split('-')",
    "messages[::-1]",
    "none",
    "5",
    "x",
    "ns|items",
    "messages[0]|items",
    "m|items",
    "namespace(p=1, q=2)|items",
};
static const char *const TARGETS[] = {"m", "x", "k, v", "k,v", "m, v"};
static const char *const CONDS[]   = {
    "true",
    "false",
    "x",
    "ns.a",
    "loop.first",
    "not loop.last",
    "m.role == 'user'",
    "1 == 1 and 2 == 3",
    "none",
    "'a' in 'abc'",
    "undefd",
};

static void gen(char *out, size_t cap, int depth);

static void append(char *out, size_t cap, const char *s) {
    size_t n = strlen(out);
    if (n + strlen(s) + 1 < cap) strcpy(out + n, s);
}

static void gen_block(char *out, size_t cap, int depth) {
    for (unsigned k = 1 + rnd(3); k > 0; k--) gen(out, cap, depth);
}

static void gen(char *out, size_t cap, int depth) {
    unsigned kind = depth <= 0 ? 0 : rnd(6);
    char buf[128];
    if (kind <= 1) {
        append(out, cap, LEAVES[rnd(sizeof LEAVES / sizeof LEAVES[0])]);
    } else if (kind == 2 || kind == 3) {
        const char *target = TARGETS[rnd(5)];
        const char *list   = LISTS[rnd(12)];
        snprintf(buf, sizeof buf, "{%% for %s in %s %%}", target, list);
        append(out, cap, buf);
        gen_block(out, cap, depth - 1);
        append(out, cap, "{% endfor %}");
    } else {
        snprintf(buf, sizeof buf, "{%% if %s %%}", CONDS[rnd(11)]);
        append(out, cap, buf);
        gen_block(out, cap, depth - 1);
        if (rnd(2)) {
            snprintf(buf, sizeof buf, "{%% elif %s %%}", CONDS[rnd(11)]);
            append(out, cap, buf);
            gen_block(out, cap, depth - 1);
        }
        if (rnd(2)) {
            append(out, cap, "{% else %}");
            gen_block(out, cap, depth - 1);
        }
        append(out, cap, "{% endif %}");
    }
}

static void test_exec_golden(void) {
    const char *roles[]    = {"user", "assistant", "user"};
    const char *contents[] = {"hello", "world", "again"};
    char tmpl[4096];
    int n_ok = 0, n_text = 0;
    for (int t = 0; t < N_CASES; t++) {
        snprintf(tmpl, sizeof tmpl,
                 "{%% set x = 1 %%}{%% set ns = namespace(a=2) %%}"
                 "{%% macro greet(who) %%}hi {{ who }}{%% endmacro %%}"
                 "{%% for m in messages %%}{%% endfor %%}");
        gen_block(tmpl, sizeof tmpl, 3);
        char *out   = chat_template_apply(tmpl, roles, contents, 3, "<s>", "</s>", 1);
        int present = out != NULL;
        mix(&present, sizeof present);
        n_ok += present;
        if (out) {
            n_text += strlen(out) > 0;
            mix(out, strlen(out) + 1);
        }
        free(out);
    }
    if (getenv("TEMPLATE_EXEC_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d rendered, %d with text)\n", (unsigned long long)g_hash, n_ok,
               n_text);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "execution matches the recorded hash");
}

int main(void) {
    printf("=== chat-template executor golden ===\n");
    RUN_TEST(test_exec_golden);
    TEST_SUMMARY();
}
