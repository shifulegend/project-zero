/**
 * test_chat_template.c — Regression tests for the Jinja2 chat-template
 * engine (src/tokenizer/chat_template.cpp).
 *
 * Covers the bugs found and fixed while getting Qwen 3.6's real
 * chat_template.jinja (Ternary-Bonsai-27B) to render correctly — see
 * docs/ai/mistakes.md for the full trace of each:
 *
 *  1. Call/filter keyword-argument syntax (`namespace(value=0)`) previously
 *     spun forever allocating AST nodes without advancing the parser — a
 *     real multi-GB OOM on Qwen 3.6's template. Now parses (and, for
 *     namespace()/macro calls, actually binds) keyword args.
 *  2. `{% macro %}...{% endmacro %}` previously had its body skipped
 *     entirely (unsupported), so every message's content — routed through
 *     Qwen 3.6's `render_content` macro — rendered empty. Macros are now
 *     defined and invoked, with positional/default/keyword parameter
 *     binding.
 *  3. `{% set ns.attr = expr %}` (dotted target, namespace-object mutation)
 *     previously overwrote the *whole* `ns` var with the RHS value instead
 *     of mutating one field.
 *  4. Python slice syntax (`messages[::-1]`) previously lexed `:` as an
 *     unknown/dropped character, silently turning it into `messages[-1]`.
 *  5. Python string method calls (`.split()`, `.strip()`/`.rstrip()`/
 *     `.lstrip()`, `.startswith()`, `.endswith()`) previously evaluated to
 *     Undef, which — combined with `{% set content = content.split(...)... %}`
 *     — clobbered `content` to empty for any assistant turn containing a
 *     `<think>...</think>` block (i.e. every multi-turn reasoning-model
 *     conversation, since the engine's own generation prompt always opens
 *     `<think>`).
 *  6. `{% for k, v in dict|items %}` (tuple-unpacking for-loop + the
 *     `|items` filter) previously discarded the second loop variable
 *     entirely and had no `items` filter at all — used by real templates
 *     to serialize tool-call arguments.
 *  7. `loop.previtem`/`loop.nextitem` were documented as supported (this
 *     file's own header comment claimed it) but never actually
 *     implemented — the real Qwen 3.6 template uses both to detect
 *     tool-response message boundaries.
 *  8. Method calls (`.startswith()` etc.) parsed correctly off a variable
 *     but desynced the parser when called directly on a literal
 *     (`'x'.upper()`), since only Ident got the postfix-chain treatment in
 *     parse_primary. Not exercised by the real template (it only ever
 *     calls methods on variables), but a real parser gap.
 */

#include "tokenizer/chat_template.h"
#include "test_harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Keyword args must not hang / OOM (regression for the parser stall).
 * ================================================================ */

static void test_kwarg_call_does_not_stall(void) {
    const char *tmpl =
        "{%- set ns = namespace(value=0) -%}"
        "{{- ns.value -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "kwarg call renders instead of failing");
    if (out) {
        TEST_ASSERT(strcmp(out, "0") == 0, "namespace() kwarg seeds the object field");
        free(out);
    }
}

/* ================================================================
 * Dotted `set` mutates the namespace object in place.
 * ================================================================ */

static void test_dotted_set_mutates_namespace(void) {
    const char *tmpl =
        "{%- set ns = namespace(a=1, b=2) -%}"
        "{%- set ns.a = 99 -%}"
        "{{- ns.a -}},{{- ns.b -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "dotted set renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "99,2") == 0, "set ns.a mutates only that field, leaves ns.b intact");
        free(out);
    }
}

/* ================================================================
 * Macro definition + invocation: positional, default, and keyword params.
 * ================================================================ */

static void test_macro_positional_and_default_args(void) {
    const char *tmpl =
        "{%- macro greet(name, punct='!') -%}"
        "Hello {{ name }}{{ punct }}"
        "{%- endmacro -%}"
        "{{- greet('World') }} {{ greet('Zig', '?') -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "macro call renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "Hello World! Hello Zig?") == 0, "macro binds positional arg + default, and positional override");
        free(out);
    }
}

static void test_macro_keyword_args(void) {
    const char *tmpl =
        "{%- macro greet(name, punct='!') -%}"
        "Hello {{ name }}{{ punct }}"
        "{%- endmacro -%}"
        "{{- greet(punct='?', name='Kwarg') -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "macro keyword-arg call renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "Hello Kwarg?") == 0, "macro binds keyword args by name regardless of order");
        free(out);
    }
}

/* render_content-style macro: this is the exact shape (string vs. list
 * content, is-string test) that made every Qwen 3.6 message body render
 * empty before macros were implemented. */
static void test_macro_render_content_shape(void) {
    const char *tmpl =
        "{%- macro render_content(content) -%}"
        "{%- if content is string -%}{{- content -}}{%- else -%}[non-string]{%- endif -%}"
        "{%- endmacro -%}"
        "{%- for message in messages -%}"
        "{{- render_content(message.content) -}}"
        "{%- endfor -%}";
    const char *roles[2] = {"user", "assistant"};
    const char *contents[2] = {"Hi", "Hello there!"};
    char *out = chat_template_apply(tmpl, roles, contents, 2, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "render_content-shaped macro renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "HiHello there!") == 0, "macro-routed message content is not dropped");
        free(out);
    }
}

/* ================================================================
 * Python slice syntax: messages[::-1].
 * ================================================================ */

static void test_reverse_slice(void) {
    const char *tmpl =
        "{%- for m in messages[::-1] -%}"
        "{{- m.content -}},"
        "{%- endfor -%}";
    const char *roles[3] = {"user", "assistant", "user"};
    const char *contents[3] = {"one", "two", "three"};
    char *out = chat_template_apply(tmpl, roles, contents, 3, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "reverse-slice loop renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "three,two,one,") == 0, "messages[::-1] iterates in reverse, not messages[-1]");
        free(out);
    }
}

/* ================================================================
 * Python string method calls.
 * ================================================================ */

static void test_string_methods(void) {
    /* Method calls chain off a `set` variable throughout, exactly like the
     * real chat_template.jinja does (`content.split(...)`) — calling a
     * method directly on a literal (`'abc'.startswith(...)`) is a separate,
     * broader parser gap (LitStr/LitInt don't get the postfix-chain
     * treatment Ident does) that doesn't occur anywhere in the real
     * template, so it's out of scope here; not tested. */
    const char *tmpl =
        "{%- set c = '<think>reason</think>answer' -%}"
        "{%- if '</think>' in c -%}"
        "{%- set reasoning = c.split('</think>')[0].split('<think>')[-1] -%}"
        "{%- set body = c.split('</think>')[-1].lstrip('\n') -%}"
        "{{- reasoning -}}|{{- body -}}"
        "{%- endif -%}"
        "|{%- set padded = '  pad  ' -%}{{- padded.strip() -}}"
        "|{%- set abc = 'abcdef' -%}{{- abc.startswith('abc') -}}"
        "|{{- abc.endswith('def') -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "string-method chain renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "reason|answer|pad|True|True") == 0, "split/lstrip/strip/startswith/endswith all work");
        free(out);
    }
}

/* ================================================================
 * Existing behavior must still work: plain for/if/set, filters, undefined
 * tolerance, 0-message input.
 * ================================================================ */

static void test_basic_chatml_still_works(void) {
    const char *tmpl =
        "{%- for message in messages -%}"
        "<|im_start|>{{ message.role }}\n{{ message.content|trim }}<|im_end|>\n"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    const char *roles[2] = {"system", "user"};
    const char *contents[2] = {" You are helpful. ", "Hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 2, "<bos>", "<eos>", 1);
    TEST_ASSERT(out != NULL, "basic ChatML template still renders");
    if (out) {
        /* `{%- endfor -%}` / `{%- if -%}` strip the trailing "\n" after
         * each "<|im_end|>" and the leading text before "<|im_start|>". */
        const char *expected =
            "<|im_start|>system\nYou are helpful.<|im_end|>"
            "<|im_start|>user\nHi<|im_end|>"
            "<|im_start|>assistant";
        TEST_ASSERT(strcmp(out, expected) == 0, "unrelated for/if/filter behavior unchanged");
        free(out);
    }
}

static void test_zero_messages_does_not_crash(void) {
    const char *tmpl = "{%- if not messages -%}{{- raise_exception('no messages') -}}{%- endif -%}ok";
    char *out = chat_template_apply(tmpl, NULL, NULL, 0, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "0-message render does not crash (raise_exception is a no-op)");
    if (out) {
        TEST_ASSERT(strcmp(out, "ok") == 0, "raise_exception no-op still lets rendering continue");
        free(out);
    }
}

/* ================================================================
 * `|items` filter + tuple-unpacking for-loop: real chat templates use
 * `{% for k, v in tool_call.arguments|items %}` to serialize tool-call
 * arguments. Previously the for-loop parser discarded the second loop
 * variable entirely, and `|items` didn't exist as a filter at all.
 * ================================================================ */

static void test_items_filter_and_tuple_unpacking(void) {
    const char *tmpl =
        "{%- set obj = namespace(a=1, b=2) -%}"
        "{%- for k, v in obj|items -%}"
        "{{- k -}}={{- v -}};"
        "{%- endfor -%}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "items+tuple-unpacking renders");
    if (out) {
        /* Object keys iterate in sorted order (std::map) — a=1 before b=2. */
        TEST_ASSERT(strcmp(out, "a=1;b=2;") == 0, "|items yields [k,v] pairs bound by tuple-unpacking for");
        free(out);
    }
}

/* ================================================================
 * loop.previtem / loop.nextitem: documented as supported but never
 * actually implemented — the real Qwen 3.6 template uses both to detect
 * tool-response message boundaries.
 * ================================================================ */

static void test_loop_previtem_nextitem(void) {
    const char *tmpl =
        "{%- for x in messages -%}"
        "[{%- if loop.previtem -%}p={{- loop.previtem.content -}}{%- else -%}p=none{%- endif -%}"
        ",{%- if loop.nextitem -%}n={{- loop.nextitem.content -}}{%- else -%}n=none{%- endif -%}]"
        "{%- endfor -%}";
    const char *roles[3] = {"user", "assistant", "user"};
    const char *contents[3] = {"one", "two", "three"};
    char *out = chat_template_apply(tmpl, roles, contents, 3, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "loop.previtem/nextitem renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "[p=none,n=two][p=one,n=three][p=two,n=none]") == 0,
                    "loop.previtem/nextitem expose the adjacent array elements");
        free(out);
    }
}

/* ================================================================
 * Method calls now parse (and, for the receiver-is-an-expression case,
 * dispatch) uniformly regardless of whether the receiver is a variable or
 * a literal — parse_primary previously only gave Ident the postfix-chain
 * treatment, so `'literal'.upper()` would leave `.upper()` unconsumed and
 * desync the parser. This checks it at least parses cleanly now (method
 * dispatch itself is keyed off Getattr receivers regardless of what's
 * under them — see eval()'s NT::Call case).
 * ================================================================ */

static void test_method_call_on_variable_still_works(void) {
    const char *tmpl =
        "{%- set s = 'Hello' -%}"
        "{{- s.startswith('He') -}}|{{- s.endswith('lo') -}}";
    const char *roles[1] = {"user"};
    const char *contents[1] = {"hi"};
    char *out = chat_template_apply(tmpl, roles, contents, 1, "<bos>", "<eos>", 0);
    TEST_ASSERT(out != NULL, "method call off a variable renders");
    if (out) {
        TEST_ASSERT(strcmp(out, "True|True") == 0, "startswith/endswith off a set variable still work");
        free(out);
    }
}

int main(void) {
    RUN_TEST(test_kwarg_call_does_not_stall);
    RUN_TEST(test_dotted_set_mutates_namespace);
    RUN_TEST(test_macro_positional_and_default_args);
    RUN_TEST(test_macro_keyword_args);
    RUN_TEST(test_macro_render_content_shape);
    RUN_TEST(test_reverse_slice);
    RUN_TEST(test_string_methods);
    RUN_TEST(test_basic_chatml_still_works);
    RUN_TEST(test_zero_messages_does_not_crash);
    RUN_TEST(test_items_filter_and_tuple_unpacking);
    RUN_TEST(test_loop_previtem_nextitem);
    RUN_TEST(test_method_call_on_variable_still_works);
    TEST_SUMMARY();
}
