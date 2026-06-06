#ifndef TN_RAG_MEMORY_SEARCH_H
#define TN_RAG_MEMORY_SEARCH_H

/**
 * Phase 15.4 — Memory Search & Retrieval
 *
 * High-level search interface: given a query string, embed it, find the
 * top-k most similar entries in the vector DB, and return the text + scores.
 *
 * Entries with similarity < MEMORY_SEARCH_MIN_SCORE are filtered out.
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

/* Minimum cosine similarity for a result to be considered relevant */
#define MEMORY_SEARCH_MIN_SCORE 0.5f

/* Maximum results returned by memory_search() */
#define MEMORY_SEARCH_MAX_K 10

/**
 * A single search result.
 */
typedef struct {
    const char *text;   /* pointer into VectorDB text pool (do not free) */
    float       score;  /* cosine similarity score in [0, 1] */
} MemoryResult;

/**
 * Search the vector database for the top-k entries most similar to `query_text`.
 *
 * Steps:
 *   1. Generate an embedding for `query_text` using `emb`.
 *   2. Call rag_find_top_k() against the DB's flat embedding matrix.
 *   3. Filter out results with score < MEMORY_SEARCH_MIN_SCORE.
 *   4. Return filtered results in `results` (descending score order).
 *
 * @param db          Open VectorDB to search
 * @param query_text  Text query (null-terminated)
 * @param cfg         Model config
 * @param w           Transformer weights (read-only)
 * @param emb         Initialised Embedder (not thread-safe — single caller)
 * @param tok         Tokenizer
 * @param tp          Thread pool (may be NULL)
 * @param results     Output array (caller provides, size ≥ max_results)
 * @param max_results Maximum number of results to return (≤ MEMORY_SEARCH_MAX_K)
 * @return            Number of results filled in `results` (≥ 0), or -1 on error.
 */
int memory_search(VectorDB *db,
                  const char *query_text,
                  const Config *cfg,
                  const TransformerWeights *w,
                  Embedder *emb,
                  Tokenizer *tok,
                  ThreadPool *tp,
                  MemoryResult *results,
                  int max_results);

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_MEMORY_SEARCH_H */
