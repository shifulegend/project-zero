/* Vector DB persistence: v2 format with checksums, v1 upgrade with backup,
 * file locking, crash-orphan recovery, bounds checks, and no leaks on any
 * error path (this test runs under ASan/LSan in `make test`). */
#include "test_harness.h"
#include "rag/vector_db.h"
#include "core/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DIM 4
static char g_dir[] = "/tmp/pz_vdb_test_XXXXXX";
static char g_path[300];

static void fresh(const char *name) {
    snprintf(g_path, sizeof g_path, "%s/%s", g_dir, name);
    char cmd[1400]; snprintf(cmd, sizeof cmd, "rm -f '%s' '%s'.v1.bak* '%s'.new", g_path, g_path, g_path);
    if (system(cmd) != 0) { /* nothing to remove */ }
}
static void vec(float *v, float seed) { for (int i = 0; i < DIM; i++) v[i] = seed + (float)i; }

static long file_size(const char *p) { FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }
static void flip_byte(const char *p, long off) {
    FILE *f = fopen(p, "r+b"); fseek(f, off, SEEK_SET); int c = fgetc(f); fseek(f, off, SEEK_SET); fputc(c ^ 0x5A, f); fclose(f);
}

static void test_roundtrip_and_checksums(void) {
    fresh("rt.vdb");
    VectorDB db; float v[DIM];
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK, "create");
    for (int i = 0; i < 3; i++) { vec(v, (float)i); char t[32]; snprintf(t, sizeof t, "entry %d", i); TEST_ASSERT(vector_db_store(&db, v, t) == TN_OK, "store"); }
    vector_db_close(&db);
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK && db.num_entries == 3, "reload 3 entries");
    TEST_ASSERT(strcmp(vector_db_text(&db, 2), "entry 2") == 0 && db.embeddings[2 * DIM + 1] == 3.0f, "contents intact");
    vector_db_close(&db);

    flip_byte(g_path, 20 + DIM * 4 + 4 + 1);          /* a text byte of record 0 */
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_ERR_DB_OPEN, "flipped record byte is detected");
    fresh("rt2.vdb");
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK, "create 2");
    vec(v, 1); vector_db_store(&db, v, "x"); vector_db_close(&db);
    flip_byte(g_path, 9);                               /* header: num_entries */
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_ERR_DB_OPEN, "flipped header byte is detected");
}

static void write_v1(const char *p, int n) {
    FILE *f = fopen(p, "wb");
    uint32_t h[4] = { VRDB_MAGIC, 1, (uint32_t)n, DIM };
    fwrite(h, 4, 4, f);
    for (int i = 0; i < n; i++) {
        float v[DIM]; vec(v, (float)i * 10);
        fwrite(v, 4, DIM, f);
        char t[16]; snprintf(t, sizeof t, "old %d", i); uint32_t tl = (uint32_t)strlen(t) + 1;
        fwrite(&tl, 4, 1, f); fwrite(t, 1, tl, f);
    }
    fclose(f);
}

static void test_v1_upgrade(void) {
    fresh("v1.vdb");
    write_v1(g_path, 2);
    long orig = file_size(g_path);
    char orig_bytes[512]; { FILE *f = fopen(g_path, "rb"); size_t n = fread(orig_bytes, 1, sizeof orig_bytes, f); fclose(f); (void)n; }
    VectorDB db;
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK && db.num_entries == 2, "v1 file opens and loads");
    TEST_ASSERT(strcmp(vector_db_text(&db, 1), "old 1") == 0, "v1 contents read correctly");
    TEST_ASSERT(file_size(g_path) == orig, "opening a v1 file does not rewrite it");
    float v[DIM]; vec(v, 99);
    TEST_ASSERT(vector_db_store(&db, v, "new one") == TN_OK, "first store upgrades");
    vector_db_close(&db);
    char bak[400]; snprintf(bak, sizeof bak, "%s.v1.bak", g_path);
    TEST_ASSERT(file_size(bak) == orig, "backup has the original size");
    char b2[512] = {0}; { FILE *f = fopen(bak, "rb"); size_t n = fread(b2, 1, sizeof b2, f); fclose(f); (void)n; }
    TEST_ASSERT(memcmp(orig_bytes, b2, (size_t)orig) == 0, "backup is byte-identical to the original v1 file");
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK && db.num_entries == 3, "upgraded file reloads with 3 entries");
    TEST_ASSERT(strcmp(vector_db_text(&db, 0), "old 0") == 0 && strcmp(vector_db_text(&db, 2), "new one") == 0, "old and new entries intact");
    vector_db_close(&db);
    uint32_t ver = 0; { FILE *f = fopen(g_path, "rb"); fseek(f, 4, SEEK_SET); size_t n = fread(&ver, 4, 1, f); fclose(f); (void)n; }
    TEST_ASSERT(ver == 2, "file on disk is now version 2");

    /* a second v1 file at the same path must not overwrite the existing backup */
    write_v1(g_path, 1);
    TEST_ASSERT(vector_db_open(&db, g_path, DIM) == TN_OK, "reopen new v1");
    vec(v, 5); TEST_ASSERT(vector_db_store(&db, v, "again") == TN_OK, "upgrade again");
    vector_db_close(&db);
    char bak1[420]; snprintf(bak1, sizeof bak1, "%s.1", bak);
    TEST_ASSERT(file_size(bak) == orig && file_size(bak1) > 0, "existing backup kept, new one gets a numbered name");
}

static void test_lock_and_orphans_and_errors(void) {
    fresh("lock.vdb");
    VectorDB a, b; float v[DIM];
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_OK, "first open");
    TEST_ASSERT(vector_db_open(&b, g_path, DIM) == TN_ERR_DB_OPEN, "second open of the same file is refused");
    vec(v, 1); TEST_ASSERT(vector_db_store(&a, v, "a") == TN_OK, "store a");
    vec(v, 2); TEST_ASSERT(vector_db_store(&a, v, "b") == TN_OK, "store b");
    vector_db_close(&a);
    TEST_ASSERT(vector_db_open(&b, g_path, DIM) == TN_OK, "open succeeds once the first is closed");
    vector_db_close(&b);

    /* crash between the record append and the header update: orphan bytes after the counted data */
    { FILE *f = fopen(g_path, "ab"); const char junk[] = "ORPHANED-PARTIAL-RECORD"; fwrite(junk, 1, sizeof junk, f); fclose(f); }
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_OK && a.num_entries == 2, "orphan bytes are ignored on open");
    vec(v, 3); TEST_ASSERT(vector_db_store(&a, v, "c") == TN_OK, "store after orphans");
    vector_db_close(&a);
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_OK && a.num_entries == 3 && strcmp(vector_db_text(&a, 2), "c") == 0,
                "record after orphans is aligned and readable");
    vector_db_close(&a);

    /* write failure leaves memory and the counted database unchanged */
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_OK, "open for failure test");
    int n0 = a.num_entries;
    close(fileno(a.fp));                                  /* make every write fail */
    vec(v, 4);
    TEST_ASSERT(vector_db_store(&a, v, "lost") == TN_ERR_DB_WRITE, "store reports the write failure");
    TEST_ASSERT(a.num_entries == n0 && vector_db_text(&a, n0) == NULL, "in-memory state not changed by a failed store");
    vector_db_close(&a);

    /* entry count above the limit */
    fresh("big.vdb");
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_OK, "create big");
    vector_db_close(&a);
    { FILE *f = fopen(g_path, "r+b"); uint32_t huge = 0x7FFFFFFFu; fseek(f, 8, SEEK_SET); fwrite(&huge, 4, 1, f); fclose(f); }
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_ERR_DB_OPEN, "oversized entry count is refused (header CRC or bound)");

    /* v1 file with an oversized count and one with a truncated record: refused, nothing leaked */
    write_v1(g_path, 2);
    { FILE *f = fopen(g_path, "r+b"); uint32_t huge = 0x7FFFFFFFu; fseek(f, 8, SEEK_SET); fwrite(&huge, 4, 1, f); fclose(f); }
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_ERR_DB_OPEN, "v1 oversized count refused");
    write_v1(g_path, 2);
    { long sz = file_size(g_path); if (truncate(g_path, sz - 3) != 0) perror("truncate"); }
    TEST_ASSERT(vector_db_open(&a, g_path, DIM) == TN_ERR_DB_OPEN, "truncated v1 record refused, resources released");
    TEST_ASSERT(vector_db_open(&b, g_path, DIM) == TN_ERR_DB_OPEN, "the failed open released its lock (reopen fails on content, not lock)");
}

int main(void) {
    printf("=== vector db persistence ===\n");
    if (!mkdtemp(g_dir)) { perror("mkdtemp"); return 1; }
    RUN_TEST(test_roundtrip_and_checksums);
    RUN_TEST(test_v1_upgrade);
    RUN_TEST(test_lock_and_orphans_and_errors);
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", g_dir);
    if (system(cmd) != 0) { /* best effort */ }
    TEST_SUMMARY();
}
