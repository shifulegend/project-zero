#ifndef TN_RAG_EMBEDDER_H
#define TN_RAG_EMBEDDER_H

/**
 * Phase 15.1 — Embedding Generator
 *
 * Converts a text string into a fixed-length float vector ("embedding")
 * by running the LLM forward pass over all tokens and mean-pooling
 * the final-layer hidden states (s->x after the last transformer layer
 * and final RMSNorm).
 *
 * The output is L2-normalised so that cosine_similarity(a, b) == dot(a, b).
 *
 * A dedicated small RunState (max_seq_len = EMBEDDER_MAX_SEQ) is used so
 * that embedding calls never pollute the main conversation's KV cache.
 */

#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "core/error.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum token sequence length for the embedding RunState.
 * 512 tokens is sufficient for typical memory snippets. */
#define EMBEDDER_MAX_SEQ 512

/**
 * Embedder context.
 * Owns a dedicated RunState used exclusively for embedding generation.
 * Shared (non-owned) references to weights, config, tokenizer, thread pool.
 */
typedef struct {
    RunState  *state;     /* dedicated embedding RunState (owned) */
    int        embed_dim; /* = cfg->dim */
    int        initialised;
} Embedder;

/**
 * Initialise an Embedder.
 * Allocates a small RunState (EMBEDDER_MAX_SEQ tokens).
 *
 * @param emb   Embedder to initialise (caller provides storage)
 * @param cfg   Model configuration (not stored, only used for alloc sizes)
 * @return TN_OK on success, TN_ERR_OOM on allocation failure.
 */
TernaryError embedder_init(Embedder *emb, const Config *cfg);

/**
 * Free resources owned by the Embedder.
 */
void embedder_free(Embedder *emb);

/**
 * Generate a normalised embedding vector for `text`.
 *
 * Algorithm:
 *   1. Tokenise `text` (up to EMBEDDER_MAX_SEQ tokens).
 *   2. Reset the dedicated RunState (clear KV cache, current_pos = 0).
 *   3. Run transformer_forward() for each token, accumulating s->x.
 *   4. Divide accumulated sum by token count (mean pool).
 *   5. L2-normalise the result → unit vector stored in `out`.
 *
 * @param emb   Initialised Embedder
 * @param out   Output buffer, must hold embed_dim floats
 * @param text  Input text (null-terminated)
 * @param cfg   Model config
 * @param w     Transformer weights (read-only)
 * @param tok   Tokenizer
 * @param tp    Thread pool (may be NULL for single-threaded)
 * @return      Number of tokens processed (> 0), or -1 on error.
 */
int embedder_generate(Embedder *emb, float *out, const char *text,
                      const Config *cfg, const TransformerWeights *w,
                      Tokenizer *tok, ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_EMBEDDER_H */
