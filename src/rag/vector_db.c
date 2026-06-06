#include "rag/vector_db.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <errno.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static TernaryError write_header(FILE *fp, int num_entries, int embed_dim) {
    uint32_t hdr[4] = {
        VRDB_MAGIC,
        VRDB_VERSION,
        (uint32_t)num_entries,
        (uint32_t)embed_dim
    };
    rewind(fp);
    if (fwrite(hdr, sizeof(uint32_t), 4, fp) != 4) return TN_ERR_DB_WRITE;
    fflush(fp);
    return TN_OK;
}

/* Ensure text_pool has room for `extra` bytes */
static TernaryError pool_reserve(VectorDB *db, size_t extra) {
    size_t needed = db->text_pool_used + extra;
    if (needed <= db->text_pool_capacity) return TN_OK;

    size_t new_cap = db->text_pool_capacity == 0 ? 65536 : db->text_pool_capacity * 2;
    while (new_cap < needed) new_cap *= 2;

    char *new_pool = (char *)realloc(db->text_pool, new_cap);
    if (!new_pool) return TN_ERR_OOM;

    /* Fix up text pointers — they point into the old pool */
    ptrdiff_t delta = new_pool - db->text_pool;
    if (delta != 0) {
        for (int i = 0; i < db->num_entries; i++) {
            db->texts[i] += delta;
        }
    }
    db->text_pool          = new_pool;
    db->text_pool_capacity = new_cap;
    return TN_OK;
}

/* Ensure embeddings array has room for one more entry */
static TernaryError embeddings_reserve(VectorDB *db) {
    if (db->num_entries < db->capacity) return TN_OK;

    int new_cap = db->capacity == 0 ? 64 : db->capacity * 2;
    if (new_cap > VRDB_MAX_ENTRIES) return TN_ERR_OOM;

    float *new_emb = (float *)realloc(db->embeddings,
                                      (size_t)new_cap * db->embed_dim * sizeof(float));
    if (!new_emb) return TN_ERR_OOM;
    db->embeddings = new_emb;

    char **new_texts = (char **)realloc(db->texts, (size_t)new_cap * sizeof(char *));
    if (!new_texts) return TN_ERR_OOM;
    db->texts    = new_texts;
    db->capacity = new_cap;
    return TN_OK;
}

/* ── public API ──────────────────────────────────────────────────────────── */

TernaryError vector_db_open(VectorDB *db, const char *path, int embed_dim) {
    if (!db || !path || embed_dim <= 0) return TN_ERR_INVALID_ARGS;
    memset(db, 0, sizeof(*db));
    db->embed_dim = embed_dim;
    strncpy(db->path, path, sizeof(db->path) - 1);

    /* Try to open existing file */
    FILE *fp = fopen(path, "r+b");
    if (!fp) {
        /* File does not exist — create it */
        fp = fopen(path, "w+b");
        if (!fp) return TN_ERR_DB_OPEN;

        TernaryError err = write_header(fp, 0, embed_dim);
        if (err != TN_OK) { fclose(fp); return err; }

        db->fp = fp;
        return TN_OK;
    }

    /* Existing file: read and validate header */
    uint32_t hdr[4];
    if (fread(hdr, sizeof(uint32_t), 4, fp) != 4) {
        fclose(fp); return TN_ERR_DB_OPEN;
    }
    if (hdr[0] != VRDB_MAGIC)   { fclose(fp); return TN_ERR_INVALID_MAGIC; }
    if (hdr[1] != VRDB_VERSION) { fclose(fp); return TN_ERR_VERSION_MISMATCH; }
    if ((int)hdr[3] != embed_dim) {
        fprintf(stderr, "[RAG] vector_db: embed_dim mismatch (file=%u, expected=%d)\n",
                hdr[3], embed_dim);
        fclose(fp); return TN_ERR_DIMENSION_MISMATCH;
    }

    int num_entries = (int)hdr[2];
    db->fp = fp;

    /* Load all records into memory */
    for (int i = 0; i < num_entries; i++) {
        TernaryError err;

        /* Reserve space */
        err = embeddings_reserve(db);
        if (err != TN_OK) return err;

        float *emb_row = db->embeddings + (size_t)i * embed_dim;
        if (fread(emb_row, sizeof(float), (size_t)embed_dim, fp) != (size_t)embed_dim) {
            fprintf(stderr, "[RAG] vector_db: truncated embedding at entry %d\n", i);
            return TN_ERR_DB_OPEN;
        }

        uint32_t text_len;
        if (fread(&text_len, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "[RAG] vector_db: truncated text_len at entry %d\n", i);
            return TN_ERR_DB_OPEN;
        }
        if (text_len == 0 || text_len > VRDB_MAX_TEXT) {
            fprintf(stderr, "[RAG] vector_db: invalid text_len %u at entry %d\n", text_len, i);
            return TN_ERR_DB_OPEN;
        }

        err = pool_reserve(db, (size_t)text_len);
        if (err != TN_OK) return err;

        char *text_dst = db->text_pool + db->text_pool_used;
        if (fread(text_dst, 1, (size_t)text_len, fp) != (size_t)text_len) {
            fprintf(stderr, "[RAG] vector_db: truncated text at entry %d\n", i);
            return TN_ERR_DB_OPEN;
        }
        text_dst[text_len - 1] = '\0'; /* ensure null-termination */

        db->texts[i]       = text_dst;
        db->text_pool_used += text_len;
        db->num_entries++;
    }

    return TN_OK;
}

TernaryError vector_db_store(VectorDB *db, const float *embedding, const char *text) {
    if (!db || !embedding || !text) return TN_ERR_INVALID_ARGS;
    if (!db->fp) return TN_ERR_DB_OPEN;
    if (db->num_entries >= VRDB_MAX_ENTRIES) return TN_ERR_OOM;

    size_t text_len = strlen(text) + 1; /* include null terminator */
    if (text_len > VRDB_MAX_TEXT) text_len = VRDB_MAX_TEXT;

    /* Grow in-memory arrays */
    TernaryError err = embeddings_reserve(db);
    if (err != TN_OK) return err;
    err = pool_reserve(db, text_len);
    if (err != TN_OK) return err;

    /* Copy into in-memory arrays */
    int idx = db->num_entries;
    float *emb_row = db->embeddings + (size_t)idx * db->embed_dim;
    memcpy(emb_row, embedding, (size_t)db->embed_dim * sizeof(float));

    char *text_dst = db->text_pool + db->text_pool_used;
    memcpy(text_dst, text, text_len - 1);
    text_dst[text_len - 1] = '\0';
    db->texts[idx]       = text_dst;
    db->text_pool_used  += text_len;
    db->num_entries++;

    /* Append record to file (seek to end) */
    if (fseek(db->fp, 0, SEEK_END) != 0) return TN_ERR_DB_WRITE;
    if (fwrite(embedding, sizeof(float), (size_t)db->embed_dim, db->fp) != (size_t)db->embed_dim)
        return TN_ERR_DB_WRITE;

    uint32_t tl32 = (uint32_t)text_len;
    if (fwrite(&tl32, sizeof(uint32_t), 1, db->fp) != 1) return TN_ERR_DB_WRITE;
    if (fwrite(text_dst, 1, text_len, db->fp) != text_len) return TN_ERR_DB_WRITE;

    /* Update num_entries in header */
    return write_header(db->fp, db->num_entries, db->embed_dim);
}

void vector_db_close(VectorDB *db) {
    if (!db) return;
    if (db->fp) {
        fflush(db->fp);
        fclose(db->fp);
        db->fp = NULL;
    }
    free(db->embeddings);
    free(db->texts);
    free(db->text_pool);
    memset(db, 0, sizeof(*db));
}
