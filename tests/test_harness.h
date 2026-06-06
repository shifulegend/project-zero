#ifndef TN_TEST_HARNESS_H
#define TN_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tn_tests_run = 0;
static int tn_tests_passed = 0;
static int tn_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tn_tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, (msg)); \
        tn_tests_failed++; \
    } else { \
        tn_tests_passed++; \
    } \
} while(0)

#define TEST_ASSERT_EQ(a, b, msg) \
    TEST_ASSERT((a) == (b), msg)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    TEST_ASSERT(fabsf((float)(a) - (float)(b)) < (eps), msg)

#define RUN_TEST(fn) do { \
    printf("Running %s...\n", #fn); \
    fn(); \
} while(0)

#define TEST_SUMMARY() do { \
    printf("\n=== %d/%d tests passed", tn_tests_passed, tn_tests_run); \
    if (tn_tests_failed > 0) printf(", %d FAILED", tn_tests_failed); \
    printf(" ===\n"); \
    return tn_tests_failed > 0 ? 1 : 0; \
} while(0)

#endif /* TN_TEST_HARNESS_H */
