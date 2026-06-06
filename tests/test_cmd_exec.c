#include "test_harness.h"
#include "agent/cmd_exec.h"
#include <string.h>

static void test_policy_allows_basic(void) {
    TEST_ASSERT(exec_policy_allows("ls"), "ls is allowed");
}

static void test_policy_blocks_rm(void) {
    TEST_ASSERT(!exec_policy_allows("rm"), "rm is blocked");
}

static void test_exec_echo(void) {
    char *argv[] = {"echo", "hello", NULL};
    char out[256] = {0};
    ExecResult r = execute_command(argv, 2, out, sizeof(out));
    TEST_ASSERT(r.exit_code == 0, "echo exited 0");
    TEST_ASSERT(strstr(out, "hello") != NULL, "echo output seen");
}

static void test_blocked_command(void) {
    char *argv[] = {"rm", "-rf", "/tmp/nonexistent", NULL};
    char out[256] = {0};
    ExecResult r = execute_command(argv, 1, out, sizeof(out));
    TEST_ASSERT(r.exit_code == 127, "blocked command returns 127 (policy)");
}

int main(void) {
    RUN_TEST(test_policy_allows_basic);
    RUN_TEST(test_policy_blocks_rm);
    RUN_TEST(test_exec_echo);
    RUN_TEST(test_blocked_command);
    TEST_SUMMARY();
}
