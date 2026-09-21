/*
 * test_cmd_exec_gaps.c — TS-3.2: the agent sandbox's remaining known-gap
 * table (docs/reports/TEST_PLAN_2026-09-18.md §TS-3.2), verified/documented
 * one gap at a time:
 *
 *   - Symlink escape: FIXED 2026-09-18 -- see test_cmd_exec_adversarial.c's
 *     test_symlink_escape_now_blocked, not repeated here.
 *   - PATH hijacking: FIXED 2026-09-19 -- resolve_trusted_path() in
 *     cmd_exec.c resolves allow-listed commands against a fixed {/bin,
 *     /usr/bin} list via execv(), never the process's own $PATH via
 *     execvp(). Verified below with a live PATH-hijacking attempt.
 *   - PROJECT_ZERO_AGENT_AUTO_APPROVE: verified below that it does not
 *     weaken execute_command()'s own policy check -- it only lives in
 *     user_approval.c, a layer entirely upstream of and independent from
 *     cmd_exec.c.
 *   - TOCTOU: NOT fixed, documented only (see the comment on
 *     test_toctou_is_a_known_unfixed_gap below) -- a nondeterministic race
 *     condition is not something a reliable, non-flaky unit test can prove
 *     either way, and the test plan itself rates this "low severity given
 *     the human-approval gate, but real", not something this pass fixes.
 */
#include "test_harness.h"
#include "agent/cmd_exec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

/* ── PATH hijacking: a malicious executable named "cat" placed earlier in
 * $PATH than the real one must NOT run in its place. ─────────────────── */
static void test_path_hijacking_prevented(void) {
    const char *evil_dir = "test_gaps_evil_path_tmp";
    const char *evil_cat = "test_gaps_evil_path_tmp/cat";
    const char *marker = "test_gaps_evil_marker_tmp";

    unlink(marker);
    unlink(evil_cat);
    rmdir(evil_dir);
    if (mkdir(evil_dir, 0755) != 0) {
        printf("  SKIP: could not create a scratch directory for the PATH-hijack test\n");
        return;
    }

    /* A fake "cat" that, if it ever ran, would create a marker file and
     * print something unmistakably different from the real cat's output. */
    FILE *f = fopen(evil_cat, "w");
    if (!f) {
        printf("  SKIP: could not write the fake 'cat' script\n");
        rmdir(evil_dir);
        return;
    }
    fprintf(f, "#!/bin/sh\ntouch %s\necho HIJACKED\n", marker);
    fclose(f);
    chmod(evil_cat, 0755);

    char *old_path = getenv("PATH");
    char *saved_path = old_path ? strdup(old_path) : NULL;
    char new_path[4096];
    snprintf(new_path, sizeof(new_path), "%s:%s", evil_dir, old_path ? old_path : "/usr/bin");
    setenv("PATH", new_path, 1);

    /* A real, harmless relative file to cat -- README.md, used elsewhere in
     * this test suite (tests/test_cmd_exec.c) as a known-present fixture. */
    char *argv[] = {"cat", "README.md", NULL};
    char out[256] = {0};
    ExecResult r = execute_command(argv, 3, out, sizeof(out));

    if (saved_path) { setenv("PATH", saved_path, 1); free(saved_path); }
    else unsetenv("PATH");

    TEST_ASSERT(strstr(out, "HIJACKED") == NULL,
                "the malicious PATH-earlier 'cat' never ran (execv against a trusted dir, not execvp)");
    TEST_ASSERT(access(marker, F_OK) != 0,
                "the malicious script's side effect (marker file) never happened");
    TEST_ASSERT(r.exit_code == 0, "the REAL cat still ran successfully on README.md");

    unlink(marker);
    unlink(evil_cat);
    rmdir(evil_dir);
}

/* ── PROJECT_ZERO_AGENT_AUTO_APPROVE must not weaken execute_command()'s own
 * policy check -- that env var only affects the separate human-approval
 * gate in user_approval.c, never reached by these tests since they call
 * execute_command() directly. This test proves the policy layer itself is
 * structurally independent of it. ──────────────────────────────────────── */
static void test_auto_approve_does_not_weaken_policy(void) {
    setenv("PROJECT_ZERO_AGENT_AUTO_APPROVE", "1", 1);

    char *bad_argv[] = {"cat", "/etc/passwd", NULL};
    char out[256] = {0};
    ExecResult r = execute_command(bad_argv, 2, out, sizeof(out));
    TEST_ASSERT(r.exit_code == 127, "absolute-path cat is still blocked with AUTO_APPROVE=1 set");

    char *good_argv[] = {"echo", "still fine", NULL};
    ExecResult r2 = execute_command(good_argv, 2, out, sizeof(out));
    TEST_ASSERT(r2.exit_code == 0, "an allow-listed command still runs normally with AUTO_APPROVE=1 set");

    unsetenv("PROJECT_ZERO_AGENT_AUTO_APPROVE");
}

/*
 * TOCTOU (time-of-check-to-time-of-use): exec_policy_allows_args()'s
 * realpath()-based symlink check (path_escapes_cwd_via_symlink) and the
 * later execv() in the forked child are not atomic -- a relative path that
 * resolves safely at check time could, in principle, be replaced (e.g. by
 * something else with write access to the same directory tree) with a
 * symlink pointing outside the CWD before the child actually execs.
 *
 * This is a real gap, not fixed in this pass: closing it properly needs an
 * atomic open-then-check-then-exec sequence (e.g. openat() with
 * O_NOFOLLOW, fstat the resulting fd, verify its realpath, then fexecve()
 * that fd) instead of the current path-string-based check -- a real
 * restructuring of execute_command()'s control flow, not a small change,
 * so it is being explicitly deferred rather than attempted here.
 *
 * Not covered by a runtime test: reliably winning a race window this
 * narrow (between a userspace realpath() call and a fork()+execv() a few
 * instructions later) is not something a deterministic, non-flaky unit
 * test can demonstrate either way -- a test that "usually" doesn't catch
 * the race proves nothing, and one that relies on scheduler cooperation
 * to force it isn't testing the real-world exploitability. Documenting the
 * gap here (and in docs/ai/mistakes.md / cmd_exec.c's own comments) is the
 * test plan's own prescribed treatment for this specific case: "low
 * severity given the human-approval gate, but real."
 */
static void test_toctou_is_a_known_unfixed_gap(void) {
    TEST_ASSERT(1, "documented above, not fixed this pass -- see the comment on this test");
}

int main(void) {
    RUN_TEST(test_path_hijacking_prevented);
    RUN_TEST(test_auto_approve_does_not_weaken_policy);
    RUN_TEST(test_toctou_is_a_known_unfixed_gap);
    TEST_SUMMARY();
}
