/*
 * test_cmd_exec_adversarial.c — TS-3.1: adversarial argument corpus for the
 * agent sandbox (src/agent/cmd_exec.c). Table-driven per
 * docs/reports/TEST_PLAN_2026-09-18.md §TS-3.1: every row asserts
 * exec_policy_allows_args() AND execute_command() agree (a denied argv
 * must both fail the policy check and get exit_code == 127 from
 * execute_command, never actually exec).
 */
#include "test_harness.h"
#include "agent/cmd_exec.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>

/* Both layers must agree: the policy function directly, and
 * execute_command()'s own enforcement (exit_code 127 = policy denied). */
static void assert_blocked(char *const argv[], const char *label) {
    char msg[160];
    snprintf(msg, sizeof(msg), "%s: exec_policy_allows_args blocks", label);
    TEST_ASSERT(!exec_policy_allows_args(argv), msg);

    char out[64] = {0};
    ExecResult r = execute_command(argv, 2, out, sizeof(out));
    snprintf(msg, sizeof(msg), "%s: execute_command returns 127 (policy denied)", label);
    TEST_ASSERT(r.exit_code == 127, msg);
}

static void assert_allowed_by_policy(char *const argv[], const char *label) {
    char msg[160];
    snprintf(msg, sizeof(msg), "%s: exec_policy_allows_args allows", label);
    TEST_ASSERT(exec_policy_allows_args(argv), msg);
}

/* ── Absolute paths ──────────────────────────────────────────────────────── */
static void test_absolute_paths_blocked(void) {
    char *a[] = {"cat", "/etc/shadow", NULL};            assert_blocked(a, "cat /etc/shadow");
    char *b[] = {"cat", "/etc/passwd", NULL};             assert_blocked(b, "cat /etc/passwd");
    char *c[] = {"cat", "/proc/self/environ", NULL};      assert_blocked(c, "cat /proc/self/environ");
    char *d[] = {"cat", "/root/.ssh/id_rsa", NULL};       assert_blocked(d, "cat /root/.ssh/id_rsa");
    char *e[] = {"cat", "//etc/passwd", NULL};            assert_blocked(e, "cat //etc/passwd (double slash)");
    char *f[] = {"ls",  "/etc", NULL};                    assert_blocked(f, "ls /etc");
}

/* ── Home-dir shortcuts ──────────────────────────────────────────────────── */
static void test_home_shortcuts_blocked(void) {
    char *a[] = {"cat", "~/.ssh/id_rsa", NULL};  assert_blocked(a, "cat ~/.ssh/id_rsa");
    char *b[] = {"cat", "~root/.bashrc", NULL};  assert_blocked(b, "cat ~root/.bashrc");
}

/* ── Traversal ───────────────────────────────────────────────────────────── */
static void test_traversal_blocked(void) {
    char *a[] = {"cat", "../x", NULL};                          assert_blocked(a, "../x");
    char *b[] = {"cat", "../../x", NULL};                       assert_blocked(b, "../../x");
    char *c[] = {"cat", "a/../../x", NULL};                     assert_blocked(c, "a/../../x");
    char *d[] = {"cat", "a/b/../../../etc/passwd", NULL};       assert_blocked(d, "a/b/../../../etc/passwd");
    char *e[] = {"ls",  "..", NULL};                            assert_blocked(e, "bare ..");
    char *f[] = {"cat", "./../x", NULL};                        assert_blocked(f, "./../x");
}

/* ── Traversal look-alikes: must NOT be over-blocked ────────────────────── */
static void test_traversal_lookalikes_allowed(void) {
    char *a[] = {"cat", "...", NULL};          assert_allowed_by_policy(a, "'...'");
    char *b[] = {"cat", "foo..bar.txt", NULL}; assert_allowed_by_policy(b, "'foo..bar.txt'");
    char *c[] = {"ls",  ".hidden", NULL};      assert_allowed_by_policy(c, "'.hidden'");
    char *d[] = {"cat", "a..b/c", NULL};       assert_allowed_by_policy(d, "'a..b/c'");
}

/* ── "Encoded" traversal: this engine does no percent/backslash decoding --
 * execvp() takes argv bytes literally, so these strings are (at most) odd
 * but harmless literal filenames, not an escape. Each case is asserted
 * individually against the real policy function rather than assumed, since
 * some of them (e.g. a bare "../") ARE real ASCII ".." segments and must
 * still be blocked by the existing lexical check -- "no decoding happens"
 * doesn't mean "no literal .. can appear". ────────────────────────────────*/
static void test_encoded_traversal_documented(void) {
    /* No literal '.' characters at all -- not traversal, not decoded either. */
    char *a[] = {"cat", "%2e%2e%2f", NULL};
    assert_allowed_by_policy(a, "'%2e%2e%2f' (no decoding -> literal filename)");

    /* ".." followed by '%' (not '/' or end-of-string) is not a clean ".."
     * path segment per path_arg_is_safe's own rule -- allowed as literal. */
    char *b[] = {"cat", "..%2f", NULL};
    assert_allowed_by_policy(b, "'..%2f' (.. not segment-terminated -> literal)");

    /* ".." followed by a literal backslash: same reasoning, backslash is
     * not a path separator on this POSIX target. */
    char *c[] = {"cat", "..\\", NULL};
    assert_allowed_by_policy(c, "'..\\\\' (backslash is not a separator here)");

    /* This one IS a real ".." path segment (dot-dot then a genuine '/') --
     * correctly caught by the ordinary traversal check, not an evasion. */
    char *d[] = {"cat", "../", NULL};
    assert_blocked(d, "'../' (real traversal, not an encoding bypass)");
}

/* ── Null/odd bytes: must never crash, must block or fail cleanly ────────── */
static void test_null_and_odd_bytes(void) {
    /* An "embedded NUL" cannot exist inside a C string argv element -- the
     * string simply ends there. Exercise the resulting truncated value
     * (a bare ".", which is a hidden-dir-style relative name, ALLOWED). */
    char embedded_nul_arg[] = "abc\0def";
    char *a[] = {"cat", embedded_nul_arg, NULL};
    TEST_ASSERT(exec_policy_allows_args(a) == 1, "truncated-at-NUL arg ('abc') does not crash and is allowed");

    /* Very long path (> PATH_MAX): must not crash or overflow. */
    char long_path[PATH_MAX * 2];
    memset(long_path, 'a', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = '\0';
    char *b[] = {"cat", long_path, NULL};
    int allowed = exec_policy_allows_args(b); /* no crash is the actual assertion */
    TEST_ASSERT(allowed == 0 || allowed == 1, "very long path (> PATH_MAX) does not crash exec_policy_allows_args");
    char out[64] = {0};
    ExecResult r = execute_command(b, 2, out, sizeof(out));
    TEST_ASSERT(r.exit_code == 127 || r.exit_code >= 0, "very long path does not crash execute_command");

    /* UTF-8 homoglyph for '.' (U+FF0E FULLWIDTH FULL STOP) is not the ASCII
     * byte 0x2E, so it can never match the lexical ".." check -- allowed as
     * a literal (and harmless, since the OS doesn't treat it as ".." either). */
    char *c[] = {"cat", "\xef\xbc\x8e\xef\xbc\x8e/etc/passwd", NULL};
    assert_allowed_by_policy(c, "UTF-8 fullwidth-dot homoglyph is not ASCII '.' -- literal, not traversal");
}

/* ── Symlink escape: FIXED 2026-09-18 (was a documented known gap) ───────
 * path_escapes_cwd_via_symlink() resolves the arg with realpath() and
 * requires the result stay within getcwd()'s subtree, so a relative name
 * that is itself a symlink pointing outside CWD is now caught even though
 * it has no ".." in its own text. */
static void test_symlink_escape_now_blocked(void) {
    /* The outside-CWD target must actually exist: realpath() fails on a
     * dangling symlink, and path_escapes_cwd_via_symlink() correctly treats
     * that as "not this check's problem" (a broken symlink can't leak
     * anything -- cat/ls fails on it harmlessly regardless), which would
     * silently skip this test's real assertion rather than exercise it. A
     * hardcoded system path is not portable enough to rely on for that: this
     * test previously used /etc/hostname, which doesn't exist on macOS/XNU
     * (no such file there, unlike Linux) -- confirmed via a real macOS CI
     * run, 2026-09-21. mkstemp() under the system tmp dir gives a target
     * that's guaranteed to exist and guaranteed outside CWD on every POSIX
     * platform this project targets. */
    char target_template[] = "/tmp/pz_symlink_escape_target_XXXXXX";
    int target_fd = mkstemp(target_template);
    if (target_fd < 0) {
        printf("  SKIP: could not create outside-CWD target file (%s)\n", strerror(errno));
        return;
    }
    close(target_fd);

    const char *link_name = "test_adv_symlink_escape_tmp";
    unlink(link_name); /* best-effort cleanup from a previous crashed run */
    if (symlink(target_template, link_name) != 0) {
        printf("  SKIP: could not create test symlink (no write access to CWD?)\n");
        unlink(target_template);
        return;
    }

    char *a[] = {"cat", (char *)link_name, NULL};
    assert_blocked(a, "cat <symlink pointing outside CWD>");

    unlink(link_name);
    unlink(target_template);
}

static void test_symlink_within_cwd_still_allowed(void) {
    /* A symlink that points to a file INSIDE the CWD subtree must still be
     * usable -- this is a confinement check, not a blanket symlink ban. */
    const char *link_name = "test_adv_symlink_internal_tmp";
    unlink(link_name);
    if (symlink("README.md", link_name) != 0) {
        printf("  SKIP: could not create test symlink (no write access to CWD, or README.md missing)\n");
        return;
    }
    char *a[] = {"cat", (char *)link_name, NULL};
    assert_allowed_by_policy(a, "cat <symlink pointing inside CWD>");
    unlink(link_name);
}

/* ── Flag injection ──────────────────────────────────────────────────────── */
static void test_flag_injection(void) {
    char *a[] = {"ls", "--help", NULL};             assert_allowed_by_policy(a, "ls --help (flag, not a path)");
    char *b[] = {"cat", "-v", "/etc/passwd", NULL};
    TEST_ASSERT(!exec_policy_allows_args(b), "cat -v /etc/passwd: absolute path arg still blocks despite flag");
    char *c[] = {"date", "-s", "00:00", NULL};       assert_blocked(c, "date -s 00:00 (clock-setting)");
    char *d[] = {"date", "--set=00:00", NULL};       assert_blocked(d, "date --set=00:00 (clock-setting)");
    char *e[] = {"echo", "-e", "hi", NULL};          assert_allowed_by_policy(e, "echo -e hi (echo has no path-confinement rule)");
}

/* ── Shell metacharacters: fork()+execvp() has no shell -- these must land
 * as a single literal (failed) filename argument, never get interpreted. ── */
static void test_shell_metacharacters_are_literal(void) {
    char *a[] = {"ls", "; rm -rf /", NULL};
    assert_allowed_by_policy(a, "'; rm -rf /' is policy-allowed as a literal filename candidate");
    /* Proof it's inert: run it for real and confirm no shell side effect. */
    char marker[] = "/tmp/pz_adv_test_should_not_exist_marker";
    unlink(marker);
    char injected[128];
    snprintf(injected, sizeof(injected), "; touch %s ;", marker);
    char *b[] = {"ls", injected, NULL};
    char out[256] = {0};
    ExecResult r = execute_command(b, 2, out, sizeof(out));
    TEST_ASSERT(access(marker, F_OK) != 0, "shell metacharacter payload did not create a marker file (no shell exists)");
    TEST_ASSERT(r.exit_code != 0, "ls on the literal metacharacter string fails (no such file), not '0' from a shell no-op");
    unlink(marker);

    char *c[] = {"ls", "a && cat /etc/passwd", NULL};
    assert_allowed_by_policy(c, "'a && cat /etc/passwd' is policy-allowed as a literal filename candidate");
    char *d[] = {"ls", "`whoami`", NULL};
    assert_allowed_by_policy(d, "backtick payload is policy-allowed as a literal filename candidate");
    char *e[] = {"ls", "$(id)", NULL};
    assert_allowed_by_policy(e, "$() payload is policy-allowed as a literal filename candidate");
    char *f[] = {"ls", "a | nc", NULL};
    assert_allowed_by_policy(f, "pipe payload is policy-allowed as a literal filename candidate");
}

/* ── argv[0] tricks ──────────────────────────────────────────────────────── */
static void test_argv0_tricks_blocked(void) {
    char *a[] = {"./ls", NULL};      assert_blocked(a, "./ls (not in name allow-list)");
    char *b[] = {"/bin/ls", NULL};   assert_blocked(b, "/bin/ls (not in name allow-list)");
    char *c[] = {"ls/../ls", NULL};  assert_blocked(c, "ls/../ls (not in name allow-list)");
    char *d[] = {"", NULL};          assert_blocked(d, "empty string argv[0]");
    TEST_ASSERT(!exec_policy_allows_args(NULL), "NULL argv is blocked, not a crash");
    /* argv[0] == NULL hits execute_command's early defensive `!argv[0]`
     * return (exit_code stays at its -1 init value, distinct from the
     * ordinary "-127 policy denied" path reached via exec_policy_allows_args)
     * -- both are non-success/non-crash, just not the same sentinel. */
    char *e[] = {NULL};
    TEST_ASSERT(!exec_policy_allows_args(e), "argv[0] itself NULL: policy blocks");
    char out[64] = {0};
    ExecResult r = execute_command(e, 2, out, sizeof(out));
    TEST_ASSERT(r.exit_code != 0, "argv[0] itself NULL: execute_command does not report success");
}

int main(void) {
    RUN_TEST(test_absolute_paths_blocked);
    RUN_TEST(test_home_shortcuts_blocked);
    RUN_TEST(test_traversal_blocked);
    RUN_TEST(test_traversal_lookalikes_allowed);
    RUN_TEST(test_encoded_traversal_documented);
    RUN_TEST(test_null_and_odd_bytes);
    RUN_TEST(test_symlink_escape_now_blocked);
    RUN_TEST(test_symlink_within_cwd_still_allowed);
    RUN_TEST(test_flag_injection);
    RUN_TEST(test_shell_metacharacters_are_literal);
    RUN_TEST(test_argv0_tricks_blocked);
    TEST_SUMMARY();
}
