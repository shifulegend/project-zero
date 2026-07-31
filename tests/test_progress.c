/*
 * tests/test_progress.c
 *
 * Phase 22.3 — model-load progress line formatting (pure functions only,
 * no real TTY needed).
 */

#include <stdio.h>
#include <string.h>
#include "cli/progress.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_non_tty_format(void) {
    char buf[128];
    size_t n = tn_progress_format(buf, sizeof(buf), 2, 4, "Loading weights...", 0);
    if (n == 0) { FAIL("non_tty_format", "expected non-empty output"); return; }
    if (!strstr(buf, "[2/4]")) { FAIL("non_tty_format", "missing stage counter"); return; }
    if (!strstr(buf, "Loading weights...")) { FAIL("non_tty_format", "missing label"); return; }
    if (buf[n-1] != '\n') { FAIL("non_tty_format", "non-tty output should end with newline"); return; }
    if (strstr(buf, "\r")) { FAIL("non_tty_format", "non-tty output should not use \\r"); return; }
    PASS("non_tty_format");
}

static void test_tty_format(void) {
    char buf[128];
    size_t n = tn_progress_format(buf, sizeof(buf), 1, 4, "Opening model file...", 1);
    if (n == 0) { FAIL("tty_format", "expected non-empty output"); return; }
    if (buf[0] != '\r') { FAIL("tty_format", "tty output should start with \\r"); return; }
    if (!strstr(buf, "[1/4]")) { FAIL("tty_format", "missing stage counter"); return; }
    PASS("tty_format");
}

static void test_tiny_buffer_stays_terminated(void) {
    char buf[6];
    size_t n = tn_progress_format(buf, sizeof(buf), 4, 4, "Ready.", 0);
    if (n >= sizeof(buf)) { FAIL("tiny_buffer_stays_terminated", "return exceeds cap"); return; }
    PASS("tiny_buffer_stays_terminated");
}

static void test_null_label_no_crash(void) {
    char buf[64];
    size_t n = tn_progress_format(buf, sizeof(buf), 1, 1, NULL, 0);
    if (n == 0) { FAIL("null_label_no_crash", "expected some output even with NULL label"); return; }
    PASS("null_label_no_crash");
}

int main(void) {
    printf("=== Phase 22.3 Progress Tests ===\n");
    test_non_tty_format();
    test_tty_format();
    test_tiny_buffer_stays_terminated();
    test_null_label_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All progress tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
