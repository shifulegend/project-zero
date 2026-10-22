/*
 * tests/test_prefix_cache.c — Phase 21.5 KV prefix-reuse bookkeeping.
 */
#include "test_harness.h"
#include "kv_cache/prefix_cache.h"

static void test_match_and_record(void) {
    PrefixCache pc;
    TEST_ASSERT_EQ(prefix_cache_init(&pc, 8), 0, "init");
    int a[] = {1, 2, 3, 4};
    TEST_ASSERT_EQ(prefix_cache_match(&pc, a, 4), 0, "empty record matches nothing");
    for (int i = 0; i < 4; i++) prefix_cache_set(&pc, i, a[i]);
    int b[] = {1, 2, 9, 9, 9};
    TEST_ASSERT_EQ(prefix_cache_match(&pc, b, 5), 2, "LCP = 2");
    TEST_ASSERT_EQ(prefix_cache_match(&pc, a, 3), 3, "LCP bounded by query length");
    prefix_cache_set(&pc, 2, 9);
    TEST_ASSERT_EQ(pc.n, 3, "overwrite at pos drops the tail");
    TEST_ASSERT_EQ(prefix_cache_match(&pc, b, 5), 3, "record updated");
    prefix_cache_free(&pc);
}

static void test_invalidation(void) {
    PrefixCache pc;
    prefix_cache_init(&pc, 4);
    for (int i = 0; i < 4; i++) prefix_cache_set(&pc, i, i);
    TEST_ASSERT_EQ(pc.n, 4, "full");
    prefix_cache_set(&pc, 4, 7);  /* window would wrap */
    TEST_ASSERT_EQ(pc.n, 0, "wrap invalidates");
    prefix_cache_set(&pc, 2, 7);  /* gap */
    TEST_ASSERT_EQ(pc.n, 0, "gap invalidates");
    prefix_cache_set(&pc, 0, 5);
    prefix_cache_truncate(&pc, 0);
    TEST_ASSERT_EQ(pc.n, 0, "truncate to 0");
    prefix_cache_set(&pc, 0, 5);
    prefix_cache_reset(&pc);
    TEST_ASSERT_EQ(prefix_cache_match(&pc, (int[]){5}, 1), 0, "reset clears");
    prefix_cache_free(&pc);
}

static void test_inert_cache(void) {
    PrefixCache pc;
    TEST_ASSERT_EQ(prefix_cache_init(&pc, 0), -1, "capacity 0 rejected");
    prefix_cache_set(&pc, 0, 1);
    TEST_ASSERT_EQ(prefix_cache_match(&pc, (int[]){1}, 1), 0, "inert cache never matches");
    TEST_ASSERT_EQ(prefix_cache_match(NULL, (int[]){1}, 1), 0, "NULL safe");
    prefix_cache_free(&pc);
}

int main(void) {
    printf("=== Prefix Cache (Phase 21.5) Tests ===\n");
    RUN_TEST(test_match_and_record);
    RUN_TEST(test_invalidation);
    RUN_TEST(test_inert_cache);
    TEST_SUMMARY();
}
