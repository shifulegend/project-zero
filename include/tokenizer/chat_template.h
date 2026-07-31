#ifndef TN_CHAT_TEMPLATE_H
#define TN_CHAT_TEMPLATE_H

/*
 * chat_template.h — Jinja2 renderer for LLM chat templates.
 *
 * Implemented in C++17 (src/tokenizer/chat_template.cpp) and exposed via
 * extern "C" so the rest of the C99 engine calls it as a plain C function.
 *
 * The template string is read from the GGUF file at tokenizer.chat_template
 * and rendered dynamically — nothing is hardcoded per model.
 *
 * Supported Jinja2 subset (covers DeepSeek, ChatML, Llama-3, Mistral,
 * Qwen 3.6's hybrid-attention chat_template.jinja):
 *   {{ expr }}                 — output expression
 *   {% for var in iterable %}  — loop; sets loop.{first,last,index,index0,
 *                                 previtem,nextitem,length,revindex,revindex0}
 *   {% if / elif / else %}     — conditional chain
 *   {% set var = expr %}       — variable assignment
 *   {% set ns.attr = expr %}   — namespace-object field mutation (in place)
 *   {% macro name(a,b=1) %}    — macro definition + {{ name(...) }} call,
 *                                 positional/default/keyword param binding
 *   expr + expr  / expr ~ expr — string concatenation
 *   message['key'] / message.key — object subscript / attribute
 *   arr[start:stop:step]       — Python slice (array only)
 *   expr is [not] defined/string/none/iterable/mapping/integer/odd/even
 *   str.split/strip/rstrip/lstrip/startswith/endswith(...) — string methods
 *     (receiver must be a variable/expression, e.g. `content.split(...)`;
 *     a method called directly on a literal, `'x'.upper()`, parses but
 *     isn't specially dispatched — see eval()'s NT::Call case)
 *   namespace(k=v, ...)        — object literal, seeded from keyword args
 *   range/dict/list(...)       — builtins
 *   {% for k, v in dict|items %} — dict iteration, tuple-unpacking for-loop
 *   expr | filter              — filters: trim, upper, lower, length, default,
 *                                 replace, join, int, string, tojson/safe/e,
 *                                 indent, items
 *   raise_exception(...)       — no-op (ignored)
 *   {%- / -%}  {{- / -}}       — whitespace control
 *
 * NOT supported: {% block/call/filter/raw %} — definitions are parsed and
 * skipped (see skip_matching_block()), rendering as nothing rather than
 * erroring. Deliberately left unimplemented: no real chat template
 * encountered so far uses these (they're Jinja's template-inheritance/
 * macro-body-wrapping features, without an analogous need in a
 * render-one-flat-prompt context) — revisit if a real template needs one.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Render a Jinja2 chat template.
 *
 *   tmpl                  — Jinja2 template string from GGUF (NUL-terminated)
 *   roles[]               — role strings: "user", "assistant", "system", …
 *   contents[]            — message content strings (parallel to roles)
 *   n_messages            — number of messages
 *   bos_token_str         — BOS special-token string (e.g. "<｜begin▁of▁sentence｜>")
 *   eos_token_str         — EOS special-token string
 *   add_generation_prompt — 1 = append assistant generation prefix, 0 = don't
 *
 * Returns malloc'd NUL-terminated rendered string, or NULL on error.
 * Caller must free() the result.
 */
char *chat_template_apply(const char *tmpl,
                          const char * const *roles,
                          const char * const *contents,
                          int n_messages,
                          const char *bos_token_str,
                          const char *eos_token_str,
                          int add_generation_prompt);

#ifdef __cplusplus
}
#endif

#endif /* TN_CHAT_TEMPLATE_H */
