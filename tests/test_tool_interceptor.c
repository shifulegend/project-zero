#include <string.h>

#include "agent/tool_interceptor.h"
#include "test_harness.h"

static void test_passthrough(void) {
    TiParser ti;
    ti_init(&ti);
    char out[128];
    TI_Tag t = ti_process_piece(&ti, "Hello world", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_NONE, "no tag in plain text");
}

static void test_exec_tag(void) {
    TiParser ti;
    ti_init(&ti);
    char out[128];
    TI_Tag t = ti_process_piece(&ti, "Before <exec>ls -la</exec> After", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_EXEC, "exec tag detected");
    TEST_ASSERT(strcmp(out, "ls -la") == 0, "exec content captured");
}

static void test_think_tag(void) {
    TiParser ti;
    ti_init(&ti);
    char out[128];
    TI_Tag t = ti_process_piece(&ti, "<think>ponder</think>", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_THINK, "think tag detected");
    TEST_ASSERT(strcmp(out, "ponder") == 0, "think content captured");
}

static void test_unclosed_tag(void) {
    TiParser ti;
    ti_init(&ti);
    char out[128];
    TI_Tag t = ti_process_piece(&ti, "<exec>partial", out, sizeof(out));
    TEST_ASSERT(t == TI_TAG_NONE, "partial tag not returned until closed");
}

static void test_two_parsers_do_not_mix(void) {
    TiParser a, b;
    ti_init(&a);
    ti_init(&b);
    char out[128];
    TEST_ASSERT(ti_process_piece(&a, "<exec>ls", out, sizeof(out)) == TI_TAG_NONE,
                "parser a holds a partial tag");
    TEST_ASSERT(ti_process_piece(&b, "hello </exec>", out, sizeof(out)) == TI_TAG_NONE,
                "a closing tag fed to parser b does not complete a's tag");
    TEST_ASSERT(ti_process_piece(&a, " -la</exec>", out, sizeof(out)) == TI_TAG_EXEC &&
                    strcmp(out, "ls -la") == 0,
                "parser a still completes its own tag");
}

int main(void) {
    RUN_TEST(test_passthrough);
    RUN_TEST(test_exec_tag);
    RUN_TEST(test_think_tag);
    RUN_TEST(test_unclosed_tag);
    RUN_TEST(test_two_parsers_do_not_mix);
    TEST_SUMMARY();
}
