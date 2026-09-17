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

/* ── Hardening: argument-level policy (exec_policy_allows_args) ─────────── */

static void test_args_policy_allows_relative_cat_ls(void) {
    char *cat_argv[] = {"cat", "README.md", NULL};
    char *ls_argv[]  = {"ls", "-la", "src", NULL};
    TEST_ASSERT(exec_policy_allows_args(cat_argv), "cat on a relative path is allowed");
    TEST_ASSERT(exec_policy_allows_args(ls_argv),  "ls -la on a relative path is allowed");
}

static void test_args_policy_blocks_absolute_path(void) {
    char *cat_etc[]  = {"cat", "/etc/shadow", NULL};
    char *ls_home[]  = {"ls", "/root", NULL};
    char *cat_tilde[] = {"cat", "~/.ssh/id_rsa", NULL};
    TEST_ASSERT(!exec_policy_allows_args(cat_etc),   "cat on an absolute path is blocked");
    TEST_ASSERT(!exec_policy_allows_args(ls_home),   "ls on an absolute path is blocked");
    TEST_ASSERT(!exec_policy_allows_args(cat_tilde), "cat on a ~-relative path is blocked");
}

static void test_args_policy_blocks_dotdot_traversal(void) {
    char *up_one[]   = {"cat", "../secrets.env", NULL};
    char *nested[]   = {"cat", "a/b/../../secrets.env", NULL};
    char *bare[]     = {"ls", "..", NULL};
    TEST_ASSERT(!exec_policy_allows_args(up_one), "cat ../x traversal is blocked");
    TEST_ASSERT(!exec_policy_allows_args(nested), "cat a/b/../../x traversal is blocked");
    TEST_ASSERT(!exec_policy_allows_args(bare),   "ls .. traversal is blocked");
}

static void test_args_policy_allows_dots_that_are_not_traversal(void) {
    /* "..." and "foo..bar" contain the substring ".." but are not a ".."
     * path segment (not bounded by '/' or start/end) -- must not be
     * confused with real traversal. */
    char *ellipsis[] = {"cat", "...", NULL};
    char *dotdotword[] = {"cat", "foo..bar.txt", NULL};
    char *hidden[]     = {"ls", ".hidden", NULL};
    TEST_ASSERT(exec_policy_allows_args(ellipsis),   "'...' is not traversal");
    TEST_ASSERT(exec_policy_allows_args(dotdotword), "'foo..bar.txt' is not traversal");
    TEST_ASSERT(exec_policy_allows_args(hidden),     "'.hidden' (single dot) is not traversal");
}

static void test_args_policy_date_readonly_only(void) {
    char *plain[]   = {"date", NULL};
    char *fmt[]     = {"date", "+%Y-%m-%d", NULL};
    char *set_short[] = {"date", "-s", "12:00", NULL};
    char *set_long[]  = {"date", "--set=12:00", NULL};
    TEST_ASSERT(exec_policy_allows_args(plain), "bare 'date' is allowed");
    TEST_ASSERT(exec_policy_allows_args(fmt),   "'date +FORMAT' is allowed");
    TEST_ASSERT(!exec_policy_allows_args(set_short), "'date -s' (clock-setting) is blocked");
    TEST_ASSERT(!exec_policy_allows_args(set_long),  "'date --set=' (clock-setting) is blocked");
}

static void test_args_policy_other_commands_unrestricted_by_path_rule(void) {
    /* echo/pwd/uname/id don't take meaningful path arguments -- the
     * path-confinement rule is specific to cat/ls and must not accidentally
     * block unrelated allow-listed commands. */
    char *echo_argv[] = {"echo", "/etc/shadow", NULL}; /* just text to echo, not a file read */
    TEST_ASSERT(exec_policy_allows_args(echo_argv), "echo is not subject to the path-confinement rule");
}

static void test_exec_blocks_path_traversal_end_to_end(void) {
    /* Confirms execute_command() itself (not just the policy function in
     * isolation) enforces the new argument-level check. */
    char *argv[] = {"cat", "/etc/hostname", NULL};
    char out[256] = {0};
    ExecResult r = execute_command(argv, 1, out, sizeof(out));
    TEST_ASSERT(r.exit_code == 127, "execute_command rejects cat on an absolute path");
}

int main(void) {
    RUN_TEST(test_policy_allows_basic);
    RUN_TEST(test_policy_blocks_rm);
    RUN_TEST(test_exec_echo);
    RUN_TEST(test_blocked_command);
    RUN_TEST(test_args_policy_allows_relative_cat_ls);
    RUN_TEST(test_args_policy_blocks_absolute_path);
    RUN_TEST(test_args_policy_blocks_dotdot_traversal);
    RUN_TEST(test_args_policy_allows_dots_that_are_not_traversal);
    RUN_TEST(test_args_policy_date_readonly_only);
    RUN_TEST(test_args_policy_other_commands_unrestricted_by_path_rule);
    RUN_TEST(test_exec_blocks_path_traversal_end_to_end);
    TEST_SUMMARY();
}
