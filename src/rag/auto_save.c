#include "rag/auto_save.h"
#include "rag/similarity.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int auto_save_memory(const char *text,
                     VectorDB *db,
                     const Config *cfg,
                     const TransformerWeights *w,
                     Embedder *emb,
                     Tokenizer *tok,
                     ThreadPool *tp) {
    if (!text || !db || !cfg || !w || !emb || !tok) return -1;
    if (db->num_entries >= VRDB_MAX_ENTRIES) return -1;

    /* Generate embedding for new text */
    float *new_vec = (float *)tn_aligned_calloc((size_t)emb->embed_dim,
                                                 sizeof(float), 64);
    if (!new_vec) return -1;

    int n_toks = embedder_generate(emb, new_vec, text, cfg, w, tok, tp);
    if (n_toks <= 0) {
        tn_aligned_free(new_vec);
        return -1;
    }

    /* Deduplication: check if any existing entry is too similar */
    const float *existing = vector_db_embeddings(db);
    for (int i = 0; i < db->num_entries; i++) {
        float sim = rag_cosine_similarity(new_vec,
                                          existing + (size_t)i * db->embed_dim,
                                          db->embed_dim);
        if (sim >= AUTO_SAVE_DEDUP_THRESHOLD) {
            tn_aligned_free(new_vec);
            return 1; /* duplicate — skipped */
        }
    }

    /* Not a duplicate — store it */
    TernaryError err = vector_db_store(db, new_vec, text);
    tn_aligned_free(new_vec);

    if (err != TN_OK) return -1;
    return 0; /* saved successfully */
}
