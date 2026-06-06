#ifndef TN_PROMPT_INJECT_H
#define TN_PROMPT_INJECT_H

/**
 * Reasoning Prompt Injector.
 *
 * Appends a chain-of-thought instruction to the user's prompt,
 * asking the model to reason inside <think> tags before answering.
 *
 * Does NOT modify the original prompt string — copies into output buffer.
 */

/**
 * Inject reasoning instructions into a prompt.
 *
 * Copies user_prompt into output_prompt and appends:
 *   "\nThink step-by-step inside <think> and </think> tags before answering."
 *
 * @param output_prompt  Output buffer (must be at least max_len bytes)
 * @param user_prompt    Original user prompt (null-terminated)
 * @param max_len        Maximum length of output_prompt (including null
 * terminator)
 */
void inject_reasoning_prompt(char *output_prompt, const char *user_prompt,
                             int max_len);

#endif /* TN_PROMPT_INJECT_H */
