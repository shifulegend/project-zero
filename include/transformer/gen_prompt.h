#ifndef TN_GEN_PROMPT_H
#define TN_GEN_PROMPT_H

#include "transformer/gen_session.h"

/* Renders and tokenizes the prompt into g->prompt.tokens / g->prompt.n_tokens (chat template with
 * history, system prompt and assistant prefix, or raw prompt with BOS). Sets g->req.chat_stops.
 * Returns 0, or -1 when there is nothing to generate. */
int gen_session_build_prompt(GenSession *g, const char *prompt);

#endif /* TN_GEN_PROMPT_H */
