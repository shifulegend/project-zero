/*
 * tests/test_cmd_exec_buffer.c — execute_command() when the output does not fit: the buffer fills,
 * the call returns what fits and the child must not be left behind as a zombie (before 2026-11-21
 * the parent stopped reading without waiting for it, so every such call leaked a process-table
 * entry). Also pins the exit code, length and text for ordinary, failing and truncated runs, and
 * for a zero-length buffer.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "agent/cmd_exec.h"
#include "test_harness.h"

/* No child of this test process may be left: true when waitpid has nothing to report. */
static int no_children_left(void) {
    int status;
    pid_t r = waitpid(-1, &status, WNOHANG);
    return r == -1 && errno == ECHILD;
}

static void test_ordinary_run(void) {
    char *argv[] = {"echo", "hello", NULL};
    char buf[64];
    ExecResult r = execute_command(argv, 5, buf, sizeof buf);
    TEST_ASSERT_EQ(r.exit_code, 0, "echo exits 0");
    TEST_ASSERT_EQ((int)r.stdout_len, 6, "length of \"hello\\n\"");
    TEST_ASSERT(strcmp(buf, "hello\n") == 0, "output text");
    TEST_ASSERT(no_children_left(), "no child left after an ordinary run");
}

/* (the policy may refuse the path before running it: either way the code is non-zero) */
static void test_failing_command(void) {
    char *argv[] = {"ls", "/nonexistent_pz_dir_for_test", NULL};
    char buf[256];
    ExecResult r = execute_command(argv, 5, buf, sizeof buf);
    TEST_ASSERT(r.exit_code != 0 && r.exit_code != -1, "ls of a missing directory fails");
    TEST_ASSERT(no_children_left(), "no child left after a failing run");
}

static void test_output_larger_than_buffer(void) {
    char arg[400];
    memset(arg, 'x', sizeof arg - 1);
    arg[sizeof arg - 1] = '\0';
    char *argv[]        = {"echo", arg, NULL};
    char buf[16];
    ExecResult r = execute_command(argv, 5, buf, sizeof buf);
    TEST_ASSERT_EQ((int)r.stdout_len, 15, "the buffer is filled to size - 1");
    TEST_ASSERT(strcmp(buf, "xxxxxxxxxxxxxxx") == 0, "the text that fits");
    TEST_ASSERT_EQ(r.timed_out, 0, "not a timeout");
    TEST_ASSERT(no_children_left(), "no zombie left after a truncated run");
}

static void test_repeated_truncated_runs_leave_nothing(void) {
    char arg[300];
    memset(arg, 'y', sizeof arg - 1);
    arg[sizeof arg - 1] = '\0';
    char *argv[]        = {"echo", arg, NULL};
    for (int i = 0; i < 20; i++) {
        char buf[8];
        execute_command(argv, 5, buf, sizeof buf);
    }
    TEST_ASSERT(no_children_left(), "twenty truncated runs leave no process behind");
}

int main(void) {
    printf("=== execute_command buffer handling ===\n");
    RUN_TEST(test_ordinary_run);
    RUN_TEST(test_failing_command);
    RUN_TEST(test_output_larger_than_buffer);
    RUN_TEST(test_repeated_truncated_runs_leave_nothing);
    TEST_SUMMARY();
}
