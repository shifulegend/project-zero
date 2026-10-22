/*
 * tests/test_safe_alloc.c — checked allocation + size/bit helpers.
 *
 * Covers the pure helpers (pow2/align/size-overflow) and the ABSURD-size
 * guard. The trap paths (tn_safe_* on failure) abort by design and are
 * verified in the process-isolation audit, not here.
 */
#include "memory/safe_alloc.h"
#include "test_harness.h"
#include <limits.h>
#include <string.h>

static void test_pow2(void) {
    TEST_ASSERT(tn_is_pow2(1), "1 is a power of two");
    TEST_ASSERT(tn_is_pow2(64), "64 is a power of two");
    TEST_ASSERT(!tn_is_pow2(0), "0 is not a power of two");
    TEST_ASSERT(!tn_is_pow2(48), "48 is not a power of two");
}

static void test_next_pow2(void) {
    TEST_ASSERT_EQ(tn_next_pow2(1), (size_t)1, "next_pow2(1) = 1");
    TEST_ASSERT_EQ(tn_next_pow2(2), (size_t)2, "next_pow2(2) = 2");
    TEST_ASSERT_EQ(tn_next_pow2(3), (size_t)4, "next_pow2(3) = 4");
    TEST_ASSERT_EQ(tn_next_pow2(64), (size_t)64, "next_pow2(64) = 64");
    TEST_ASSERT_EQ(tn_next_pow2(65), (size_t)128, "next_pow2(65) = 128");
}

static void test_align_up(void) {
    TEST_ASSERT_EQ(tn_align_up(0, 16), (size_t)0, "align_up(0,16) = 0");
    TEST_ASSERT_EQ(tn_align_up(1, 16), (size_t)16, "align_up(1,16) = 16");
    TEST_ASSERT_EQ(tn_align_up(16, 16), (size_t)16, "align_up(16,16) = 16");
    TEST_ASSERT_EQ(tn_align_up(17, 16), (size_t)32, "align_up(17,16) = 32");
}

static void test_size_mul_overflow(void) {
    size_t out = 0;
    TEST_ASSERT_EQ(tn_size_mul_overflow(4, 8, &out), 0, "4*8 does not overflow");
    TEST_ASSERT_EQ(out, (size_t)32, "4*8 = 32");
    TEST_ASSERT(tn_size_mul_overflow(SIZE_MAX, 2, &out) != 0, "SIZE_MAX*2 overflows");
    TEST_ASSERT(tn_size_mul3(SIZE_MAX, 2, 2, &out) != 0, "SIZE_MAX*2*2 overflows");
}

static void test_alloc_too_large(void) {
    /* A buffer dwarfing RAM must be flagged; something modest must not.
     * (Uses the shared tn_get_free_ram(); if RAM is unknown the guard
     * returns 0 for the modest case, which is the assertion anyway.) */
    TEST_ASSERT(tn_alloc_too_large(SIZE_MAX / 4, 8) == 1, "absurd buffer flagged");
    TEST_ASSERT(tn_alloc_too_large(64, sizeof(double)) == 0, "1 KiB not flagged");
}

static void test_safe_alloc_returns_valid(void) {
    void *p = tn_safe_malloc(1024, "test");
    TEST_ASSERT(p != NULL, "tn_safe_malloc returns a buffer");
    memset(p, 0xA5, 1024);
    void *q = tn_safe_realloc(p, 2048, "test");
    TEST_ASSERT(q != NULL, "tn_safe_realloc returns a buffer");
    TEST_ASSERT(((unsigned char *)q)[0] == 0xA5, "realloc preserved contents");
    free(q);

    int *z = (int *)tn_safe_calloc(16, sizeof(int), "test");
    TEST_ASSERT(z != NULL, "tn_safe_calloc returns a buffer");
    for (int i = 0; i < 16; i++) TEST_ASSERT_EQ(z[i], 0, "calloc zero-filled");
    free(z);

    /* size 0 -> NULL (an explicitly empty buffer), no trap */
    TEST_ASSERT(tn_safe_malloc(0, "empty") == NULL, "malloc(0) is NULL");
}

int main(void) {
    RUN_TEST(test_pow2);
    RUN_TEST(test_next_pow2);
    RUN_TEST(test_align_up);
    RUN_TEST(test_size_mul_overflow);
    RUN_TEST(test_alloc_too_large);
    RUN_TEST(test_safe_alloc_returns_valid);
    TEST_SUMMARY();
}
