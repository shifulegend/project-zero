/*
 * tests/test_program_name.c — Phase 39 rename: basename parsing and the
 * legacy-name detection behind the one-release deprecation notice.
 */
#include "test_harness.h"
#include "cli/program_name.h"
#include <string.h>

static void test_basename(void) {
    TEST_ASSERT(strcmp(pz_program_basename("./pz"), "pz") == 0, "./pz -> pz");
    TEST_ASSERT(strcmp(pz_program_basename("/usr/local/bin/pz"), "pz") == 0, "absolute path");
    TEST_ASSERT(strcmp(pz_program_basename("C:\\pz\\pz.exe"), "pz.exe") == 0, "windows path");
    TEST_ASSERT(strcmp(pz_program_basename("pz"), "pz") == 0, "bare name");
    TEST_ASSERT(strcmp(pz_program_basename(NULL), "") == 0, "NULL -> empty");
}

static void test_legacy(void) {
    TEST_ASSERT(pz_is_legacy_invocation("./adaptive_ai_engine"), "legacy name detected");
    TEST_ASSERT(pz_is_legacy_invocation("/opt/x/adaptive_ai_engine"), "legacy via path");
    TEST_ASSERT(pz_is_legacy_invocation("adaptive_ai_engine.exe"), "legacy .exe");
    TEST_ASSERT(!pz_is_legacy_invocation("./pz"), "new name is not legacy");
    TEST_ASSERT(!pz_is_legacy_invocation("adaptive_ai_engine2"), "prefix only is not legacy");
    TEST_ASSERT(!pz_is_legacy_invocation("/adaptive_ai_engine/pz"), "directory name ignored");
    TEST_ASSERT(!pz_is_legacy_invocation(NULL), "NULL is not legacy");
}

int main(void) {
    printf("=== Program name (Phase 39) ===\n");
    RUN_TEST(test_basename);
    RUN_TEST(test_legacy);
    TEST_SUMMARY();
}
