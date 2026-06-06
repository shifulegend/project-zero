#ifndef TN_RAG_VECTOR_DB_H
#define TN_RAG_VECTOR_DB_H

/**
 * Phase 15.3 — Vector Database (File-Backed Storage)
 *
 * A simple, append-only binary file that stores (embedding, text) pairs.
 * No external dependencies — same philosophy as the model weight file.
 *
 * File format (little-endian):
 * ┌─────────────────────────────────────────┐
 * │ Header (16 bytes)                        │
 * │   [0-3]  magic:       uint32 = 0x42445256 "VRDB" │
 * │   [4-7]  version:     uint32 = 1         │
 * │   [8-11] num_entries: uint32             │
 * │   [12-15] embed_dim:  uint32             │
 * ├─────────────────────────────────────────┤
 * │ Record 0                                 │
 * │   [embed_dim * 4 bytes] float32 embedding│
 * │   [4 bytes]             uint32 text_len  │
 * │   [text_len bytes]      UTF-8 text + \0  │
 * ├─────────────────────────────────────────┤
 * │ Record 1, 2, …                          │
 * └─────────────────────────────────────────┘
 *
 * On open, all records are loaded into heap-allocated arrays for
 * fast in-memory search. Writes append a new record to the file
 * and update the header in-place.
 */

#include "core/error.h"
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VRDB_MAGIC   0x42445256u  /* "VRDB" as little-endian uint32 */
#define VRDB_VERSION 1u
#define VRDB_HEADER_SIZE 16

/* Maximum text length stored per entry (including null terminator) */
#define VRDB_MAX_TEXT 4096

/* Maximum entries kept in memory — large enough for practical use */
#define VRDB_MAX_ENTRIES 65536

/**
 * In-memory representation of the vector database.
 * All pointers are owned by this struct (freed on vector_db_close).
 */
typedef struct {
    int     num_entries;  /* current number of stored entries */
    int     embed_dim;    /* dimension of each embedding */
    int     capacity;     /* allocated capacity in embeddings/texts arrays */

    float  *embeddings;   /* flat row-major: embeddings[i * embed_dim + j] */
    char  **texts;        /* text[i] points into text_pool */
    char   *text_pool;    /* concatenated null-terminated strings */
    size_t  text_pool_used;
    size_t  text_pool_capacity;

    FILE   *fp;           /* open file handle (r+b or wb) */
    char    path[512];    /* file path (for error messages) */
} VectorDB;

/**
 * Open (or create) a vector database at `path`.
 *
 * - If the file does not exist, a new empty DB is created with `embed_dim`.
 * - If the file exists, its header is validated and all entries are loaded
 *   into memory.
 * - Returns TN_ERR_DIMENSION_MISMATCH if the file's embed_dim ≠ `embed_dim`.
 *
 * @param db        VectorDB to initialise (caller provides storage)
 * @param path      File path (created if absent)
 * @param embed_dim Expected embedding dimension
 * @return TN_OK on success.
 */
TernaryError vector_db_open(VectorDB *db, const char *path, int embed_dim);

/**
 * Append an (embedding, text) pair to the database.
 *
 * Updates both the in-memory arrays and the on-disk file.
 * The embedding is assumed to be unit-length (pre-normalised by the embedder).
 *
 * @param db        Open VectorDB
 * @param embedding Unit-length float vector (embed_dim floats)
 * @param text      Null-terminated text string (max VRDB_MAX_TEXT bytes)
 * @return TN_OK on success, TN_ERR_DB_WRITE on I/O failure, TN_ERR_OOM on alloc.
 */
TernaryError vector_db_store(VectorDB *db, const float *embedding, const char *text);

/**
 * Flush pending writes and close the database.
 * Frees all heap memory owned by `db`.
 */
void vector_db_close(VectorDB *db);

/**
 * Return a pointer to the flat embedding matrix for similarity search.
 * Valid until the next vector_db_store() call.
 */
static inline const float *vector_db_embeddings(const VectorDB *db) {
    return db->embeddings;
}

/**
 * Return the text for entry `i` (0-indexed).
 */
static inline const char *vector_db_text(const VectorDB *db, int i) {
    if (i < 0 || i >= db->num_entries) return NULL;
    return db->texts[i];
}

#ifdef __cplusplus
}
#endif

#endif /* TN_RAG_VECTOR_DB_H */
