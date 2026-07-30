/*
 * tests/test_banner.c
 *
 * CLI startup banner — plain (non-animated) formatter tests only (pure
 * function, no real TTY needed). The animated/TTY path in tn_banner_print
 * is covered by manual pty verification (tools/screenshots/cli/capture.mjs),
 * matching the convention in tests/test_progress.c.
 */

#include <stdio.h>
#include <string.h>
#include "cli/banner.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_no_escape_bytes(void) {
    char buf[TN_BANNER_PLAIN_BUF_CAP];
    size_t n = tn_banner_format_plain(buf, sizeof(buf));
    if (n == 0) { FAIL("no_escape_bytes", "expected non-empty output"); return; }
    if (memchr(buf, '\x1b', n)) { FAIL("no_escape_bytes", "plain output must not contain ESC bytes"); return; }
    PASS("no_escape_bytes");
}

static void test_contains_glyph_content(void) {
    char buf[TN_BANNER_PLAIN_BUF_CAP];
    tn_banner_format_plain(buf, sizeof(buf));
    if (!strchr(buf, '#')) { FAIL("contains_glyph_content", "expected at least one '#' glyph pixel"); return; }
    PASS("contains_glyph_content");
}

static void test_five_rows_newline_terminated(void) {
    char buf[TN_BANNER_PLAIN_BUF_CAP];
    size_t n = tn_banner_format_plain(buf, sizeof(buf));
    int newlines = 0;
    for (size_t i = 0; i < n; i++) if (buf[i] == '\n') newlines++;
    if (newlines != 5) { FAIL("five_rows_newline_terminated", "expected exactly 5 newlines (one per glyph row)"); return; }
    if (n == 0 || buf[n - 1] != '\n') { FAIL("five_rows_newline_terminated", "output must end with a newline"); return; }
    PASS("five_rows_newline_terminated");
}

static void test_tiny_buffer_no_crash(void) {
    char buf[4];
    size_t n = tn_banner_format_plain(buf, sizeof(buf));
    if (n >= sizeof(buf)) { FAIL("tiny_buffer_no_crash", "return must not exceed cap"); return; }
    PASS("tiny_buffer_no_crash");
}

static void test_null_buf_no_crash(void) {
    size_t n = tn_banner_format_plain(NULL, 0);
    if (n != 0) { FAIL("null_buf_no_crash", "expected 0 for NULL buf"); return; }
    PASS("null_buf_no_crash");
}

static void test_zero_cap_no_crash(void) {
    char buf[8];
    size_t n = tn_banner_format_plain(buf, 0);
    if (n != 0) { FAIL("zero_cap_no_crash", "expected 0 for zero cap"); return; }
    PASS("zero_cap_no_crash");
}

int main(void) {
    printf("=== Banner Tests ===\n");
    test_no_escape_bytes();
    test_contains_glyph_content();
    test_five_rows_newline_terminated();
    test_tiny_buffer_no_crash();
    test_null_buf_no_crash();
    test_zero_cap_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All banner tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
