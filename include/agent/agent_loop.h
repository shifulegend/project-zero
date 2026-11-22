#ifndef TN_AGENT_AGENT_LOOP_H
#define TN_AGENT_AGENT_LOOP_H

#include "core/config.h"
#include "core/weights.h"
#include "rag/rag_context.h" /* Phase 15: RagContext (may be NULL to disable) */
#include "runtime/run_state.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

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
int run_agent_loop(const char *prompt, const Config *cfg, const TransformerWeights *w, RunState *s,
                   Tokenizer *tok, ThreadPool *tp, int max_tokens, float temperature, float top_p,
                   RagContext *rag);

/**
 * The agent loop's token choice from one position's logits (vocab_size of them, rescaled in place):
 * greedy when temperature is below 1e-6, else temperature and, for 0 < top_p < 1, nucleus sampling,
 * otherwise a draw from the whole distribution. *rng_state is advanced by the draw.
 */
int agent_sample_next(float *logits, int vocab_size, float temperature, float top_p,
                      unsigned long long *rng_state);

/* Prompt the user for approval of a command. Returns 1 if approved, 0 otherwise. */
int user_approval_prompt(const char *cmd);

#ifdef __cplusplus
}
#endif

#endif /* TN_AGENT_AGENT_LOOP_H */
