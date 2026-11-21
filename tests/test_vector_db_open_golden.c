/*
 * tests/test_vector_db_open_golden.c — vector_db_open over generated database files: valid version-2
 * files (made with the library itself) and version-1 files (built by hand, no checksums) with zero
 * to six entries, each read intact, with the wrong embedding dimension, cut at random bytes and with
 * one to three random byte flips (header fields, checksums, lengths and payloads). A hash of the
 * result code and of everything loaded (format, valid end offset, every embedding and text),
 * recorded from the function before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "rag/vector_db.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xcc79b38401c25f8aull
#define DIM 4
#define N_SOURCES 120
#define VARIANTS 160
#define CAP 8192

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const k_text[] = {
    "a", "hello world", "the quick brown fox", "x y z",
    "a somewhat longer entry with several words in it, and a comma"};

static size_t read_all(const char *path, uint8_t *buf) {
    FILE *f  = fopen(path, "rb");
    size_t n = f ? fread(buf, 1, CAP, f) : 0;
    if (f) fclose(f);
    return n;
}

static void write_all(const char *path, const uint8_t *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    fwrite(buf, 1, n, f);
    fclose(f);
}

/* A version-2 file written by the library. */
static size_t make_v2(const char *path, uint8_t *out) {
    remove(path);
    VectorDB db;
    TEST_ASSERT(vector_db_open(&db, path, DIM) == TN_OK, "create");
    unsigned n = rnd(7);
    for (unsigned i = 0; i < n; i++) {
        float e[DIM];
        for (int k = 0; k < DIM; k++) e[k] = (float)rnd(1000) / 10.0f - 50.0f;
        TEST_ASSERT(vector_db_store(&db, e, k_text[rnd(5)]) == TN_OK, "store");
    }
    vector_db_close(&db);
    return read_all(path, out);
}

/* A version-1 file: 16-byte header, then [embedding][text_len][text with NUL] per entry. */
static size_t make_v1(uint8_t *out) {
    unsigned n      = rnd(7);
    uint32_t hdr[4] = {VRDB_MAGIC, VRDB_VERSION_V1, n, DIM};
    memcpy(out, hdr, sizeof hdr);
    size_t at = sizeof hdr;
    for (unsigned i = 0; i < n; i++) {
        float e[DIM];
        for (int k = 0; k < DIM; k++) e[k] = (float)rnd(1000) / 10.0f - 50.0f;
        memcpy(out + at, e, sizeof e);
        at += sizeof e;
        const char *t = k_text[rnd(5)];
        uint32_t tl   = (uint32_t)strlen(t) + 1;
        memcpy(out + at, &tl, 4);
        at += 4;
        memcpy(out + at, t, tl);
        at += tl;
    }
    return at;
}

static void open_and_hash(const char *path, int dim) {
    VectorDB db;
    int rc = (int)vector_db_open(&db, path, dim);
    mix(&rc, sizeof rc);
    if (rc != TN_OK) return;
    mix(&db.format, sizeof db.format);
    mix(&db.valid_end, sizeof db.valid_end);
    mix(&db.num_entries, sizeof db.num_entries);
    if (db.num_entries > 0) mix(db.embeddings, (size_t)db.num_entries * dim * sizeof(float));
    for (int i = 0; i < db.num_entries; i++) mix(db.texts[i], strlen(db.texts[i]) + 1);
    vector_db_close(&db);
}

/* One to three random byte flips. */
static void flip_bytes(uint8_t *buf, size_t n) {
    unsigned flips = 1 + rnd(3);
    for (unsigned k = 0; k < flips; k++) buf[rnd((unsigned)n)] ^= (uint8_t)(1 + rnd(255));
}

/* One damaged copy of `src` written to path and opened. v picks the damage. */
static void open_variant(const char *path, const uint8_t *src, size_t n, int v) {
    static uint8_t var[CAP];
    memcpy(var, src, n);
    size_t len   = n;
    int open_dim = DIM;
    if (v % 4 == 1) {
        open_dim = DIM + 1; /* wrong dimension */
    } else if (v % 4 == 2) {
        len = rnd((unsigned)n + 1); /* cut */
    } else if (v % 4 == 3) {
        flip_bytes(var, n);
    }
    write_all(path, var, len);
    open_and_hash(path, open_dim);
}

static void test_open_golden(void) {
    char path[] = "/tmp/pz_vdb_golden_XXXXXX";
    int fd      = mkstemp(path);
    TEST_ASSERT(fd >= 0, "temp file");
    close(fd);
    static uint8_t src[CAP];
    for (int s = 0; s < N_SOURCES; s++) {
        size_t n = (s % 2) ? make_v1(src) : make_v2(path, src);
        for (int v = 0; v < VARIANTS; v++) open_variant(path, src, n, v);
    }
    remove(path);
    if (getenv("VDB_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "opens match the recorded hash");
}

int main(void) {
    printf("=== vector_db_open golden ===\n");
    RUN_TEST(test_open_golden);
    TEST_SUMMARY();
}
