#include "rag/vector_db.h"
#include "core/platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <errno.h>

#if TN_POSIX
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* CRC-32 (IEEE 802.3, reflected). Bitwise: records are small, and a static
 * lookup table would need a thread-safe lazy init. */
static uint32_t crc32_update(uint32_t crc, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    crc                    = ~crc;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static TernaryError sync_file(FILE *fp) {
    if (fflush(fp) != 0) return TN_ERR_DB_WRITE;
#if TN_POSIX
    if (fsync(fileno(fp)) != 0) return TN_ERR_DB_WRITE;
#endif
    return TN_OK;
}

/* Write the v2 header (with its CRC) at offset 0 and sync it. */
static TernaryError write_header(FILE *fp, int num_entries, int embed_dim) {
    uint32_t hdr[5] = {VRDB_MAGIC, VRDB_VERSION, (uint32_t)num_entries, (uint32_t)embed_dim, 0};
    hdr[4]          = crc32_update(0, hdr, 4 * sizeof(uint32_t));
    if (fseek(fp, 0, SEEK_SET) != 0) return TN_ERR_DB_WRITE;
    if (fwrite(hdr, sizeof(uint32_t), 5, fp) != 5) return TN_ERR_DB_WRITE;
    return sync_file(fp);
}

/* Ensure text_pool has room for `extra` bytes */
static TernaryError pool_reserve(VectorDB *db, size_t extra) {
    size_t needed = db->text_pool_used + extra;
    if (needed <= db->text_pool_capacity) return TN_OK;

    size_t new_cap = db->text_pool_capacity == 0 ? 65536 : db->text_pool_capacity * 2;
    while (new_cap < needed) new_cap *= 2;

    /* text pointers are rebuilt from offsets: they point into the old pool,
     * which realloc may free. */
    size_t *offs = NULL;
    if (db->num_entries > 0) {
        offs = (size_t *)malloc((size_t)db->num_entries * sizeof(size_t));
        if (!offs) return TN_ERR_OOM;
        for (int i = 0; i < db->num_entries; i++) offs[i] = (size_t)(db->texts[i] - db->text_pool);
    }
    char *new_pool = (char *)realloc(db->text_pool, new_cap);
    if (!new_pool) {
        free(offs);
        return TN_ERR_OOM;
    }
    for (int i = 0; i < db->num_entries; i++) db->texts[i] = new_pool + offs[i];
    free(offs);
    db->text_pool          = new_pool;
    db->text_pool_capacity = new_cap;
    return TN_OK;
}

/* Ensure embeddings array has room for one more entry */
static TernaryError embeddings_reserve(VectorDB *db) {
    if (db->num_entries < db->capacity) return TN_OK;

    int new_cap = db->capacity == 0 ? 64 : db->capacity * 2;
    if (new_cap > VRDB_MAX_ENTRIES) new_cap = VRDB_MAX_ENTRIES;
    if (new_cap <= db->num_entries) return TN_ERR_OOM;

    float *new_emb =
        (float *)realloc(db->embeddings, (size_t)new_cap * db->embed_dim * sizeof(float));
    if (!new_emb) return TN_ERR_OOM;
    db->embeddings = new_emb; /* larger than `capacity` says: still consistent */

    char **new_texts = (char **)realloc(db->texts, (size_t)new_cap * sizeof(char *));
    if (!new_texts) return TN_ERR_OOM;
    db->texts    = new_texts;
    db->capacity = new_cap; /* only now do both arrays hold new_cap entries */
    return TN_OK;
}

/* Lock the open file for exclusive use; a second process must not interleave writes. */
static int lock_file(FILE *fp) {
#if TN_POSIX
    if (flock(fileno(fp), LOCK_EX | LOCK_NB) != 0) return -1;
#else
    (void)fp;
#endif
    return 0;
}

/* Build one v2 record in a heap buffer: embedding | text_len | text+NUL | crc32. */
static unsigned char *build_record(const VectorDB *db, const float *emb, const char *text,
                                   uint32_t text_len, size_t *out_len) {
    size_t emb_bytes = (size_t)db->embed_dim * sizeof(float);
    size_t len       = emb_bytes + sizeof(uint32_t) + text_len + sizeof(uint32_t);
    unsigned char *r = (unsigned char *)malloc(len);
    if (!r) return NULL;
    memcpy(r, emb, emb_bytes);
    memcpy(r + emb_bytes, &text_len, sizeof(uint32_t));
    memcpy(r + emb_bytes + sizeof(uint32_t), text, text_len - 1);
    r[emb_bytes + sizeof(uint32_t) + text_len - 1] = '\0';
    uint32_t crc                                   = crc32_update(0, r, len - sizeof(uint32_t));
    memcpy(r + len - sizeof(uint32_t), &crc, sizeof(uint32_t));
    *out_len = len;
    return r;
}

/* ── v1 -> v2 upgrade (called by the first store into a v1 file) ───────────── */

#if TN_POSIX
static int copy_file_synced(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    int fd = open(dst, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        fclose(in);
        return -1;
    }
    FILE *out = fdopen(fd, "wb");
    if (!out) {
        close(fd);
        fclose(in);
        return -1;
    }
    char buf[65536];
    size_t n;
    int ok = 1;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) {
            ok = 0;
            break;
        }
    if (ferror(in)) ok = 0;
    ok = (fflush(out) == 0) && (fsync(fd) == 0) && ok;
    ok = (fclose(out) == 0) && ok;
    fclose(in);
    if (!ok) unlink(dst);
    return ok ? 0 : -1;
}

static TernaryError upgrade_v1_to_v2(VectorDB *db) {
    char bak[sizeof(db->path) + 24], tmp[sizeof(db->path) + 8];
    int made = 0;
    for (int i = 0; i < 1000 && !made; i++) {
        if (i == 0)
            snprintf(bak, sizeof bak, "%s.v1.bak", db->path);
        else
            snprintf(bak, sizeof bak, "%s.v1.bak.%d", db->path, i);
        if (access(bak, F_OK) == 0) continue; /* never overwrite a backup */
        if (copy_file_synced(db->path, bak) == 0)
            made = 1;
        else
            return TN_ERR_DB_WRITE;
    }
    if (!made) return TN_ERR_DB_WRITE;
    fprintf(stderr, "[RAG] vector_db: upgrading %s to format v2 (backup: %s)\n", db->path, bak);

    snprintf(tmp, sizeof tmp, "%s.new", db->path);
    (void)unlink(tmp);
    int fd = open(tmp, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return TN_ERR_DB_WRITE;
    FILE *nf = fdopen(fd, "r+b");
    if (!nf) {
        close(fd);
        unlink(tmp);
        return TN_ERR_DB_WRITE;
    }
    if (lock_file(nf) != 0) {
        fclose(nf);
        unlink(tmp);
        return TN_ERR_DB_WRITE;
    }

    TernaryError err = write_header(nf, db->num_entries, db->embed_dim);
    long end         = VRDB_HEADER_SIZE;
    for (int i = 0; err == TN_OK && i < db->num_entries; i++) {
        size_t rl;
        const char *t      = db->texts[i];
        unsigned char *rec = build_record(db, db->embeddings + (size_t)i * db->embed_dim, t,
                                          (uint32_t)strlen(t) + 1, &rl);
        if (!rec) {
            err = TN_ERR_OOM;
            break;
        }
        if (fseek(nf, end, SEEK_SET) != 0 || fwrite(rec, 1, rl, nf) != rl) err = TN_ERR_DB_WRITE;
        end += (long)rl;
        free(rec);
    }
    if (err == TN_OK) err = sync_file(nf);
    if (err != TN_OK || rename(tmp, db->path) != 0) {
        fclose(nf);
        unlink(tmp);
        return err != TN_OK ? err : TN_ERR_DB_WRITE;
    }
    /* make the rename durable */
    char dir[sizeof(db->path)];
    snprintf(dir, sizeof dir, "%s", db->path);
    char *sl = strrchr(dir, '/');
    if (!sl)
        snprintf(dir, sizeof dir, ".");
    else if (sl == dir)
        dir[1] = '\0';
    else
        *sl = '\0';
    int dfd = open(dir, O_RDONLY);
    if (dfd >= 0) {
        (void)fsync(dfd);
        close(dfd);
    }

    fclose(db->fp); /* releases the old lock; the new file is already locked */
    db->fp        = nf;
    db->format    = VRDB_VERSION;
    db->valid_end = end;
    return TN_OK;
}
#endif

/* ── public API ──────────────────────────────────────────────────────────── */

/* Every error path funnels through here: frees buffers and the FILE*. */
static TernaryError open_fail(VectorDB *db, TernaryError err) {
    vector_db_close(db);
    return err;
}

TernaryError vector_db_open(VectorDB *db, const char *path, int embed_dim) {
    if (!db || !path || embed_dim <= 0) return TN_ERR_INVALID_ARGS;
    memset(db, 0, sizeof(*db));
    db->embed_dim = embed_dim;
    strncpy(db->path, path, sizeof(db->path) - 1);

    FILE *fp = NULL;
#if TN_POSIX
    /* O_CREAT without O_TRUNC: opening never truncates a file another process owns. */
    int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) return TN_ERR_DB_OPEN;
    fp = fdopen(fd, "r+b");
    if (!fp) {
        close(fd);
        return TN_ERR_DB_OPEN;
    }
#else
    fp = fopen(path, "r+b");
    if (!fp) fp = fopen(path, "w+b");
    if (!fp) return TN_ERR_DB_OPEN;
#endif
    db->fp = fp;
    if (lock_file(fp) != 0) {
        fprintf(stderr, "[RAG] vector_db: %s is in use by another process\n", path);
        return open_fail(db, TN_ERR_DB_OPEN);
    }

    if (fseek(fp, 0, SEEK_END) != 0) return open_fail(db, TN_ERR_DB_OPEN);
    long size = ftell(fp);
    if (size < 0) return open_fail(db, TN_ERR_DB_OPEN);
    if (size == 0) { /* new database */
        TernaryError err = write_header(fp, 0, embed_dim);
        if (err != TN_OK) return open_fail(db, err);
        db->format    = VRDB_VERSION;
        db->valid_end = VRDB_HEADER_SIZE;
        return TN_OK;
    }

    /* Existing file: read and validate header */
    if (fseek(fp, 0, SEEK_SET) != 0) return open_fail(db, TN_ERR_DB_OPEN);
    uint32_t hdr[5] = {0};
    if (fread(hdr, sizeof(uint32_t), 4, fp) != 4) return open_fail(db, TN_ERR_DB_OPEN);
    if (hdr[0] != VRDB_MAGIC) return open_fail(db, TN_ERR_INVALID_MAGIC);
    if (hdr[1] != VRDB_VERSION && hdr[1] != VRDB_VERSION_V1)
        return open_fail(db, TN_ERR_VERSION_MISMATCH);
    db->format = hdr[1];
    if (db->format == VRDB_VERSION) {
        if (fread(&hdr[4], sizeof(uint32_t), 1, fp) != 1) return open_fail(db, TN_ERR_DB_OPEN);
        if (crc32_update(0, hdr, 4 * sizeof(uint32_t)) != hdr[4]) {
            fprintf(stderr, "[RAG] vector_db: header checksum mismatch in %s (file is corrupt)\n",
                    path);
            return open_fail(db, TN_ERR_DB_OPEN);
        }
    }
    if ((int)hdr[3] != embed_dim) {
        fprintf(stderr, "[RAG] vector_db: embed_dim mismatch (file=%u, expected=%d)\n", hdr[3],
                embed_dim);
        return open_fail(db, TN_ERR_DIMENSION_MISMATCH);
    }
    if (hdr[2] > (uint32_t)VRDB_MAX_ENTRIES) {
        fprintf(stderr, "[RAG] vector_db: entry count %u exceeds the limit of %d\n", hdr[2],
                VRDB_MAX_ENTRIES);
        return open_fail(db, TN_ERR_DB_OPEN);
    }
    int num_entries = (int)hdr[2];

    /* Load all records into memory */
    for (int i = 0; i < num_entries; i++) {
        TernaryError err = embeddings_reserve(db);
        if (err != TN_OK) return open_fail(db, err);

        float *emb_row   = db->embeddings + (size_t)i * embed_dim;
        size_t emb_bytes = (size_t)embed_dim * sizeof(float);
        if (fread(emb_row, sizeof(float), (size_t)embed_dim, fp) != (size_t)embed_dim) {
            fprintf(stderr, "[RAG] vector_db: truncated embedding at entry %d\n", i);
            return open_fail(db, TN_ERR_DB_OPEN);
        }

        uint32_t text_len;
        if (fread(&text_len, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "[RAG] vector_db: truncated text_len at entry %d\n", i);
            return open_fail(db, TN_ERR_DB_OPEN);
        }
        if (text_len == 0 || text_len > VRDB_MAX_TEXT) {
            fprintf(stderr, "[RAG] vector_db: invalid text_len %u at entry %d\n", text_len, i);
            return open_fail(db, TN_ERR_DB_OPEN);
        }

        err = pool_reserve(db, (size_t)text_len);
        if (err != TN_OK) return open_fail(db, err);

        char *text_dst = db->text_pool + db->text_pool_used;
        if (fread(text_dst, 1, (size_t)text_len, fp) != (size_t)text_len) {
            fprintf(stderr, "[RAG] vector_db: truncated text at entry %d\n", i);
            return open_fail(db, TN_ERR_DB_OPEN);
        }
        if (db->format == VRDB_VERSION) {
            uint32_t want;
            if (fread(&want, sizeof(uint32_t), 1, fp) != 1) {
                fprintf(stderr, "[RAG] vector_db: truncated checksum at entry %d\n", i);
                return open_fail(db, TN_ERR_DB_OPEN);
            }
            uint32_t got = crc32_update(0, emb_row, emb_bytes);
            got          = crc32_update(got, &text_len, sizeof text_len);
            got          = crc32_update(got, text_dst, text_len);
            if (got != want) {
                fprintf(stderr,
                        "[RAG] vector_db: checksum mismatch at entry %d (file is corrupt)\n", i);
                return open_fail(db, TN_ERR_DB_OPEN);
            }
        }
        text_dst[text_len - 1] = '\0'; /* ensure null-termination */

        db->texts[i] = text_dst;
        db->text_pool_used += text_len;
        db->num_entries++;
    }
    db->valid_end = ftell(fp);
    return TN_OK;
}

TernaryError vector_db_store(VectorDB *db, const float *embedding, const char *text) {
    if (!db || !embedding || !text) return TN_ERR_INVALID_ARGS;
    if (!db->fp) return TN_ERR_DB_OPEN;
    if (db->num_entries >= VRDB_MAX_ENTRIES) return TN_ERR_OOM;

    size_t text_len = strlen(text) + 1; /* include null terminator */
    if (text_len > VRDB_MAX_TEXT) text_len = VRDB_MAX_TEXT;

    TernaryError err;
#if TN_POSIX
    if (db->format == VRDB_VERSION_V1) {
        err = upgrade_v1_to_v2(db);
        if (err != TN_OK) return err;
    }
#else
    if (db->format == VRDB_VERSION_V1) return TN_ERR_VERSION_MISMATCH;
#endif

    /* 1. Everything that can fail, before any state changes. */
    err = embeddings_reserve(db);
    if (err != TN_OK) return err;
    err = pool_reserve(db, text_len);
    if (err != TN_OK) return err;
    char *clipped = (char *)malloc(text_len);
    if (!clipped) return TN_ERR_OOM;
    memcpy(clipped, text, text_len - 1);
    clipped[text_len - 1] = '\0';
    size_t rec_len;
    unsigned char *rec = build_record(db, embedding, clipped, (uint32_t)text_len, &rec_len);
    if (!rec) {
        free(clipped);
        return TN_ERR_OOM;
    }

    /* 2. Append the record at the end of the counted data (overwriting any
     * orphan bytes a previous crash left), sync, then bump the header and sync. */
    err = TN_OK;
    if (fseek(db->fp, db->valid_end, SEEK_SET) != 0 || fwrite(rec, 1, rec_len, db->fp) != rec_len)
        err = TN_ERR_DB_WRITE;
    if (err == TN_OK) err = sync_file(db->fp);
    if (err == TN_OK) err = write_header(db->fp, db->num_entries + 1, db->embed_dim);
    free(rec);
    if (err != TN_OK) {
        free(clipped);
        return err;
    } /* memory and counted disk state unchanged */

    /* 3. Commit to memory. */
    int idx = db->num_entries;
    memcpy(db->embeddings + (size_t)idx * db->embed_dim, embedding,
           (size_t)db->embed_dim * sizeof(float));
    char *text_dst = db->text_pool + db->text_pool_used;
    memcpy(text_dst, clipped, text_len);
    free(clipped);
    db->texts[idx] = text_dst;
    db->text_pool_used += text_len;
    db->num_entries++;
    db->valid_end += (long)rec_len;
    return TN_OK;
}

void vector_db_close(VectorDB *db) {
    if (!db) return;
    if (db->fp) {
        fflush(db->fp);
        fclose(db->fp); /* also drops the flock */
        db->fp = NULL;
    }
    free(db->embeddings);
    free(db->texts);
    free(db->text_pool);
    memset(db, 0, sizeof(*db));
}
