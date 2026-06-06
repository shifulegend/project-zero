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
 * Supported Jinja2 subset (covers DeepSeek, ChatML, Llama-3, Mistral):
 *   {{ expr }}                 — output expression
 *   {% for var in iterable %}  — loop; sets loop.{first,last,index,index0}
 *   {% if / elif / else %}     — conditional chain
 *   {% set var = expr %}       — variable assignment
 *   expr + expr  / expr ~ expr — string concatenation
 *   message['key']             — object subscript
 *   expr is [not] defined      — defined test
 *   expr | filter              — filters: trim, upper, lower, default, replace, join
 *   raise_exception(...)       — no-op (ignored)
 *   {%- / -%}  {{- / -}}       — whitespace control
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
