#ifndef AGENT_LOOP_H
#define AGENT_LOOP_H

#include "core/config.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"
#include "rag/rag_context.h"  /* Phase 15: RagContext (may be NULL to disable) */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run an agentic generation loop.
 *
 * @param rag  Optional RAG context (Phase 15).  Pass NULL to disable memory.
 *             When non-NULL and enabled, save_memory / search_memory tags are
 *             handled by the RAG subsystem.
 *
 * @return 0 on success.
 */
int run_agent_loop(const char *prompt,
                   const Config *cfg,
                   const TransformerWeights *w,
                   RunState *s,
                   Tokenizer *tok,
                   ThreadPool *tp,
                   int max_tokens,
                   float temperature,
                   float top_p,
                   RagContext *rag);

/* Prompt the user for approval of a command. Returns 1 if approved, 0 otherwise. */
int user_approval_prompt(const char *cmd);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_LOOP_H */
