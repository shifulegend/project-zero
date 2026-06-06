#ifndef TN_RAG_AUTO_SAVE_H
#define TN_RAG_AUTO_SAVE_H

/**
 * Phase 15.6 — Auto-Save Agent Hook
 *
 * Called when the tool interceptor detects a <save_memory>…</save_memory>
 * tag in the model's output.
 *
 * Behaviour:
 *   1. Generate an embedding for `text`.
 *   2. Deduplication: check if any existing entry has cosine similarity
 *      ≥ AUTO_SAVE_DEDUP_THRESHOLD.  If so, skip (return 1).
 *   3. Store via vector_db_store().
 *   4. Return 0 on fresh save, 1 if duplicate skipped, -1 on error.
 */

#include "rag/vector_db.h"
#include "rag/embedder.h"
#include "core/config.h"
#include "core/weights.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Cosine similarity above which an entry is considered a duplicate */
#define AUTO_SAVE_DEDUP_THRESHOLD 0.95f

/**
 * Save a memory snippet to the vector DB (with deduplication).
 *
 * @param text  Null-terminated text to save (the <save_memory> content)
 * @param db    Open VectorDB
 * @param cfg   Model config
 * @param w     Transformer weights (read-only)
 * @param emb   Embedder
 * @param tok   Tokenizer
 * @param tp    Thread pool
 * @return  0 = saved successfully
 *          1 = duplicate detected, skipped
 *         -1 = error (embedding or DB write failed)
 */
int auto_save_memory(const char *text,
                     VectorDB *db,
                     const Config *cfg,
                     const TransformerWeights *w,
                     Embedder *emb,
                     Tokenizer *tok,
                     ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_AUTO_SAVE_H */
