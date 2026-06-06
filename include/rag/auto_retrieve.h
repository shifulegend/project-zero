#ifndef TN_RAG_AUTO_RETRIEVE_H
#define TN_RAG_AUTO_RETRIEVE_H

/**
 * Phase 15.5 — Auto-Retrieve on Every Prompt (Transparent RAG)
 *
 * Called automatically before every generation cycle.  Embeds the user
 * prompt, searches the vector DB, and injects any relevant memories into
 * the KV cache so the model "sees" them as part of its context.
 *
 * Budget-aware: if fewer than AUTO_RETRIEVE_MIN_FREE_TOKENS remain in the
 * KV cache, the injection is skipped to preserve space for the actual reply.
 *
 * Injected text is formatted as:
 *   \n<memory>\nRelevant context:\n- {result1}\n- {result2}\n</memory>\n
 */

#include "rag/vector_db.h"
#include "rag/embedder.h"
#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimum remaining KV cache tokens before retrieval injection is skipped */
#define AUTO_RETRIEVE_MIN_FREE_TOKENS 512

/* Minimum similarity score to include a result (stricter than search) */
#define AUTO_RETRIEVE_MIN_SCORE 0.60f

/* Top-N results to inject */
#define AUTO_RETRIEVE_TOP_N 3

/**
 * Retrieve relevant memories and inject them into the main RunState KV cache.
 *
 * @param user_prompt  The user's text prompt
 * @param db           Open VectorDB
 * @param cfg          Model config
 * @param w            Transformer weights (read-only)
 * @param s            Main RunState (KV cache modified by injection)
 * @param emb          Embedder (separate RunState — not s)
 * @param tok          Tokenizer
 * @param tp           Thread pool
 * @return             Number of tokens injected (≥ 0), or -1 on error.
 *                     0 means no relevant memories found or budget exceeded.
 */
int auto_retrieve_and_inject(const char *user_prompt,
                             VectorDB *db,
                             const Config *cfg,
                             const TransformerWeights *w,
                             RunState *s,
                             Embedder *emb,
                             Tokenizer *tok,
                             ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_AUTO_RETRIEVE_H */
