/*
 * tests/test_startup_paths.c — the startup-cost and stop-token fixes found by
 * the 2026-10-19 head-to-head sweep:
 *  - core/bw_cache: host key, save/load round trip, stale / other-host /
 *    future / malformed entries rejected, default path from PZ_CACHE_DIR;
 *  - core/text_file: size limit is inclusive, empty and missing files fail;
 *  - tn_aligned_calloc: large (lazily zeroed) buffers read back as zero even
 *    when the allocator hands back dirty memory, at odd sizes/alignments;
 *  - tokenizer_collect_eog: declared ids first, end-of-turn names, special-only
 *    filter, duplicates dropped.
 */
#include "test_harness.h"
#include "core/bw_cache.h"
#include "core/text_file.h"
#include "memory/aligned_alloc.h"
#include "tokenizer/tokenizer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char g_dir[64];

static void path_in_dir(char *buf, size_t n, const char *name) {
    snprintf(buf, n, "%s/%s", g_dir, name);
}

static void test_bw_cache(void) {
    char key[256], other[256], path[256];
    TEST_ASSERT_EQ(tn_bw_cache_key(key, sizeof(key), "Xeon\t@ 2.8", 4, 4, 33u << 20,
                                   (size_t)16 << 30), 0, "key built");
    TEST_ASSERT(strchr(key, '\t') == NULL, "tabs in the CPU name are neutralised");
    TEST_ASSERT(strstr(key, "ram=16384MB") != NULL, "key carries total RAM");
    TEST_ASSERT_EQ(tn_bw_cache_key(key, 8, "Xeon", 4, 4, 0, 0), -1, "short buffer -> -1");
    tn_bw_cache_key(key, sizeof(key), "Xeon", 4, 4, 33u << 20, (size_t)16 << 30);
    tn_bw_cache_key(other, sizeof(other), "Xeon", 8, 8, 33u << 20, (size_t)16 << 30);

    path_in_dir(path, sizeof(path), "sub/membw");             /* parent created on save */
    double g = 0.0;
    TEST_ASSERT_EQ(tn_bw_cache_load(path, key, 1000, 100, &g), 0, "missing file -> miss");
    TEST_ASSERT_EQ(tn_bw_cache_save(path, key, 33.47, 1000), 0, "save");
    TEST_ASSERT_EQ(tn_bw_cache_load(path, key, 1050, 100, &g), 1, "fresh entry -> hit");
    TEST_ASSERT_FLOAT_EQ(g, 33.47, 1e-3, "value round-trips");
    TEST_ASSERT_EQ(tn_bw_cache_load(path, key, 1101, 100, &g), 0, "older than max age -> miss");
    TEST_ASSERT_EQ(tn_bw_cache_load(path, key, 999, 100, &g), 0, "entry from the future -> miss");
    TEST_ASSERT_EQ(tn_bw_cache_load(path, other, 1050, 100, &g), 0, "other host -> miss");
    TEST_ASSERT_EQ(tn_bw_cache_save(path, key, 0.0, 1000), -1, "non-positive value refused");
    TEST_ASSERT_EQ(tn_bw_cache_save(path, "a\tb", 1.0, 1000), -1, "tab in key refused");

    FILE *f = fopen(path, "w");
    fputs("pz-membw 1\tgarbage\n", f);
    fclose(f);
    TEST_ASSERT_EQ(tn_bw_cache_load(path, key, 1050, 100, &g), 0, "malformed -> miss");

    char def[256];
    setenv("PZ_CACHE_DIR", g_dir, 1);
    TEST_ASSERT_EQ(tn_bw_cache_path(def, sizeof(def)), 0, "default path");
    path_in_dir(path, sizeof(path), "membw");
    TEST_ASSERT(strcmp(def, path) == 0, "PZ_CACHE_DIR/membw");
    unsetenv("PZ_CACHE_DIR");
}

static void write_n(const char *path, size_t n) {
    FILE *f = fopen(path, "wb");
    for (size_t i = 0; i < n; i++) fputc('a' + (int)(i % 26), f);
    fclose(f);
}

static void test_text_file(void) {
    char path[256];
    size_t len = 0;
    path_in_dir(path, sizeof(path), "t.txt");
    write_n(path, 10000);
    char *s = tn_read_text_file(path, 10000, &len);
    TEST_ASSERT(s != NULL && len == 10000 && s[10000] == '\0', "exactly max_bytes accepted");
    TEST_ASSERT(s && s[9999] == 'a' + (9999 % 26), "content intact across buffer growth");
    free(s);
    TEST_ASSERT(tn_read_text_file(path, 9999, &len) == NULL, "max_bytes + 1 refused");
    TEST_ASSERT(tn_read_text_file(path, 100, &len) == NULL, "far over limit refused");
    write_n(path, 0);
    TEST_ASSERT(tn_read_text_file(path, 100, &len) == NULL, "empty file refused");
    path_in_dir(path, sizeof(path), "nope.txt");
    TEST_ASSERT(tn_read_text_file(path, 100, &len) == NULL && len == 0, "missing file -> NULL");
}

static void test_aligned_calloc_zero(void) {
    /* Dirty a block, free it, and ask again: the allocator likely reuses it,
     * so a skipped zeroing would show. Sizes straddle partial pages. */
    const size_t sizes[] = { 64 * 4096 + 123, 1u << 20, 3 * 4096 - 8, 17 };
    int ok = 1;
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        unsigned char *p = (unsigned char *)tn_aligned_alloc(sizes[k], 64);
        if (!p) { ok = 0; break; }
        memset(p, 0xAB, sizes[k]);
        tn_aligned_free(p);
        unsigned char *q = (unsigned char *)tn_aligned_calloc(sizes[k], 1, 64);
        if (!q || ((uintptr_t)q & 63) != 0) { ok = 0; tn_aligned_free(q); break; }
        for (size_t i = 0; i < sizes[k]; i++) if (q[i]) { ok = 0; break; }
        tn_aligned_free(q);
    }
    TEST_ASSERT(ok, "tn_aligned_calloc returns zeroed, 64-byte aligned memory at all sizes");
    TEST_ASSERT(tn_aligned_calloc((size_t)-1, 16, 64) == NULL, "overflowing count*size -> NULL");
}

static char **g_vocab;
static int cmp_idx(const void *a, const void *b) {
    return strcmp(g_vocab[*(const int *)a], g_vocab[*(const int *)b]);
}

static void test_collect_eog(void) {
    char *vocab[] = { "a", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "b", "<end_of_turn>" };
    int sorted[6] = { 0, 1, 2, 3, 4, 5 };
    int special[] = { 1, 2, 5 };                         /* <|im_end|> is a NORMAL token here */
    g_vocab = vocab;
    qsort(sorted, 6, sizeof(int), cmp_idx);
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    t.vocab = vocab;
    t.vocab_size = 6;
    t.sorted_vocab_indices = sorted;
    t.sorted = 1;
    t.special_token_ids = special;
    t.n_special = 3;

    int ids[3] = { 2, -1, 99 };                          /* eos, absent eot, out of range */
    tokenizer_collect_eog(&t, ids, 3, 1);
    TEST_ASSERT_EQ(t.n_eos, 3, "eos + <|eot_id|> + <end_of_turn>");
    TEST_ASSERT_EQ(t.eos_list[0], 2, "declared EOS first");
    int has1 = 0, has3 = 0, has5 = 0;
    for (int i = 0; i < t.n_eos; i++) {
        has1 |= t.eos_list[i] == 1; has3 |= t.eos_list[i] == 3; has5 |= t.eos_list[i] == 5;
    }
    TEST_ASSERT(has1 && has5, "end-of-turn control tokens stop generation");
    TEST_ASSERT(!has3, "a NORMAL token spelled like a marker does not (require_special)");

    tokenizer_collect_eog(&t, NULL, 0, 0);
    TEST_ASSERT_EQ(t.n_eos, 4, "without require_special: names only, no duplicates");
}

int main(void) {
    printf("=== Startup paths (bw cache, text file, lazy calloc, EOG) ===\n");
    snprintf(g_dir, sizeof(g_dir), "/tmp/pz_startup_test_XXXXXX");
    if (!mkdtemp(g_dir)) { printf("mkdtemp failed\n"); return 1; }
    RUN_TEST(test_bw_cache);
    RUN_TEST(test_text_file);
    RUN_TEST(test_aligned_calloc_zero);
    RUN_TEST(test_collect_eog);
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", g_dir);
    if (system(cmd) != 0) printf("(cleanup of %s failed)\n", g_dir);
    TEST_SUMMARY();
}
