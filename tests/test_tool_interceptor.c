#include "test_harness.h"
#include "agent/tool_interceptor.h"
#include <string.h>

static void test_passthrough(void) {
    ti_init();
    char out[128];
    TI_Tag t = ti_process_piece("Hello world", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_NONE, "no tag in plain text");
}

static void test_exec_tag(void) {
    ti_init();
    char out[128];
    TI_Tag t = ti_process_piece("Before <exec>ls -la</exec> After", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_EXEC, "exec tag detected");
    TEST_ASSERT(strcmp(out, "ls -la") == 0, "exec content captured");
}

static void test_think_tag(void) {
    ti_init();
    char out[128];
    TI_Tag t = ti_process_piece("<think>ponder</think>", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_THINK, "think tag detected");
    TEST_ASSERT(strcmp(out, "ponder") == 0, "think content captured");
}

static void test_unclosed_tag(void) {
    ti_init();
    char out[128];
    TI_Tag t = ti_process_piece("<exec>partial", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_NONE, "partial tag not returned until closed");
}

int main(void) {
    RUN_TEST(test_passthrough);
    RUN_TEST(test_exec_tag);
    RUN_TEST(test_think_tag);
    RUN_TEST(test_unclosed_tag);
    TEST_SUMMARY();
}
