#ifndef TN_RAG_SIMILARITY_H
#define TN_RAG_SIMILARITY_H

/**
 * Phase 15.2 — Cosine Similarity Search
 *
 * Cosine similarity between unit-length vectors reduces to a dot product.
 * find_top_k() performs a brute-force O(n·dim) scan — faster than any
 * index structure for databases < ~100 K entries.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Cosine similarity between two L2-normalised vectors.
 * Since both vectors are unit-length, this equals dot(a, b).
 * Uses tn_vec_dot() from the SIMD dispatch table.
 *
 * @param a   First unit-length vector
 * @param b   Second unit-length vector
 * @param dim Vector dimension
 * @return    Similarity score in [-1, 1].
 */
float rag_cosine_similarity(const float *a, const float *b, int dim);

/**
 * Find the k most similar entries in the database.
 *
 * Scans all `n_entries` embeddings stored row-major in `embeddings`
 * (each row is `dim` floats, assumed unit-length).
 * Returns indices and scores of the top-k entries in descending order.
 *
 * @param query        Query embedding (unit-length, dim floats)
 * @param embeddings   Flat row-major matrix: embeddings[i * dim + j]
 * @param n_entries    Number of rows in `embeddings`
 * @param dim          Embedding dimension
 * @param k            Number of results to return
 * @param out_indices  Output: indices of top-k entries (size k)
 * @param out_scores   Output: cosine scores of top-k entries (size k)
 * @return             Actual number of results filled (≤ k).
 */
int rag_find_top_k(const float *query,
                   const float *embeddings,
                   int n_entries,
                   int dim,
                   int k,
                   int *out_indices,
                   float *out_scores);

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_SIMILARITY_H */
