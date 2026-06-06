#ifndef TN_RAG_CONTEXT_H
#define TN_RAG_CONTEXT_H

/**
 * RagContext — convenience bundle of all Phase 15 RAG state.
 *
 * Pass a single `RagContext *rag` (possibly NULL when RAG is disabled)
 * to agent_loop, repl, and other high-level functions instead of
 * threading five separate pointers everywhere.
 */

#include "rag/vector_db.h"
#include "rag/embedder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VectorDB db;      /* owns the file-backed vector store */
    Embedder emb;     /* owns the dedicated embedding RunState */
    int      enabled; /* 1 = fully initialised and ready; 0 = disabled */
} RagContext;

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_CONTEXT_H */
