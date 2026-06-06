#include "rag/similarity.h"
#include "math/simd_dispatch.h"
#include <stdlib.h>
#include <string.h>

float rag_cosine_similarity(const float *a, const float *b, int dim) {
    /* Both vectors are pre-normalised → dot product == cosine similarity */
    return tn_vec_dot(a, b, dim);
}

/* Simple insertion-sort into a fixed-size top-k array (descending by score).
 * O(n·k) — fine for k ≤ 10 and n ≤ 100K. */
int rag_find_top_k(const float *query,
                   const float *embeddings,
                   int n_entries,
                   int dim,
                   int k,
                   int *out_indices,
                   float *out_scores) {
    if (!query || !embeddings || n_entries <= 0 || dim <= 0 || k <= 0 ||
        !out_indices || !out_scores)
        return 0;

    /* Clamp k to available entries */
    if (k > n_entries) k = n_entries;

    /* Initialise with -2 (below any valid cosine similarity) */
    for (int i = 0; i < k; i++) {
        out_scores[i]  = -2.0f;
        out_indices[i] = -1;
    }

    int filled = 0; /* how many slots actually have a real value */

    for (int i = 0; i < n_entries; i++) {
        float score = rag_cosine_similarity(query, embeddings + (size_t)i * dim, dim);

        /* Find insertion position in descending-sorted top-k list */
        int insert_pos = -1;
        for (int j = 0; j < k; j++) {
            if (score > out_scores[j]) {
                insert_pos = j;
                break;
            }
        }
        if (insert_pos < 0) continue; /* not in top-k */

        /* Shift down to make room */
        for (int j = k - 1; j > insert_pos; j--) {
            out_scores[j]  = out_scores[j - 1];
            out_indices[j] = out_indices[j - 1];
        }
        out_scores[insert_pos]  = score;
        out_indices[insert_pos] = i;
        if (filled < k) filled++;
    }

    return filled;
}
