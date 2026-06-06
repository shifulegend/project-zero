#include "rag/memory_search.h"
#include "rag/similarity.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int memory_search(VectorDB *db,
                  const char *query_text,
                  const Config *cfg,
                  const TransformerWeights *w,
                  Embedder *emb,
                  Tokenizer *tok,
                  ThreadPool *tp,
                  MemoryResult *results,
                  int max_results) {
    if (!db || !query_text || !cfg || !w || !emb || !tok || !results)
        return -1;
    if (db->num_entries == 0) return 0;

    /* Clamp k */
    int k = max_results;
    if (k > MEMORY_SEARCH_MAX_K) k = MEMORY_SEARCH_MAX_K;
    if (k > db->num_entries)     k = db->num_entries;

    /* Generate query embedding */
    float *query_vec = (float *)tn_aligned_calloc((size_t)emb->embed_dim,
                                                   sizeof(float), 64);
    if (!query_vec) return -1;

    int n_toks = embedder_generate(emb, query_vec, query_text, cfg, w, tok, tp);
    if (n_toks <= 0) {
        tn_aligned_free(query_vec);
        return -1;
    }

    /* Top-k search */
    int   *top_indices = (int   *)malloc((size_t)k * sizeof(int));
    float *top_scores  = (float *)malloc((size_t)k * sizeof(float));
    if (!top_indices || !top_scores) {
        free(top_indices);
        free(top_scores);
        tn_aligned_free(query_vec);
        return -1;
    }

    int found = rag_find_top_k(query_vec,
                                vector_db_embeddings(db),
                                db->num_entries,
                                db->embed_dim,
                                k,
                                top_indices,
                                top_scores);

    /* Filter by minimum score and fill results */
    int n_results = 0;
    for (int i = 0; i < found && n_results < max_results; i++) {
        if (top_scores[i] < MEMORY_SEARCH_MIN_SCORE) continue;
        results[n_results].text  = vector_db_text(db, top_indices[i]);
        results[n_results].score = top_scores[i];
        n_results++;
    }

    free(top_indices);
    free(top_scores);
    tn_aligned_free(query_vec);
    return n_results;
}
