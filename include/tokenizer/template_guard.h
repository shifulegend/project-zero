#ifndef TN_TEMPLATE_GUARD_H
#define TN_TEMPLATE_GUARD_H

/*
 * template_guard.h — known-bad chat templates and the "prompt ends in EOS" check.
 *
 * Some GGUF files ship a chat template that makes the model generate nothing.
 * The official BitNet b1.58 2B-4T file appends {{ eos_token }} after its
 * "BITNETAssistant: " prefix, so the prompt ends in the end-of-sequence token,
 * the model predicts a new document (BOS) and generation stops at 0 tokens
 * (docs/ai/mistakes.md, 2026-11-16). Detection is by metadata (architecture +
 * template text), never by file name.
 *
 * Precedence in the CLI/server: --chat-template-file > known fix > the GGUF's
 * template. A request's own "chat_template" still wins in gen_session.c.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* The corrected BitNet b1.58 template. Same text as
 * docs/templates/bitnet-b1.58.jinja (tests/test_template_guard.c checks it). */
const char *tn_template_bitnet_b158(void);

/* If `gguf_template` is the model's own template and a known-broken one for
 * `arch` (general.architecture), returns the corrected template (static
 * storage, never freed); otherwise NULL. NULL arguments return NULL. */
const char *tn_template_known_fix(const char *arch, const char *gguf_template);

/* 1 when `rendered` ends with `token` once trailing ASCII whitespace is
 * ignored; 0 for an empty/NULL token or a NULL prompt. */
int tn_prompt_ends_with_token(const char *rendered, const char *token);

/* Prints the "prompt ends in the end-of-sequence token" warning to stderr, at
 * most once per process (safe to call from concurrent generations). */
void tn_template_warn_eos_ending_once(void);

#ifdef __cplusplus
}
#endif

#endif /* TN_TEMPLATE_GUARD_H */
