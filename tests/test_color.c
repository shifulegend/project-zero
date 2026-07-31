/*
 * tests/test_color.c
 *
 * Phase 22.3 — CLI color mode parsing/resolution.
 */

#include <stdio.h>
#include <string.h>
#include "cli/color.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_mode_parse(void) {
    if (tn_color_mode_parse("always") != TN_COLOR_ALWAYS) { FAIL("mode_parse", "always"); return; }
    if (tn_color_mode_parse("never")  != TN_COLOR_NEVER)  { FAIL("mode_parse", "never"); return; }
    if (tn_color_mode_parse("auto")   != TN_COLOR_AUTO)   { FAIL("mode_parse", "auto"); return; }
    if (tn_color_mode_parse("ALWAYS") != TN_COLOR_ALWAYS) { FAIL("mode_parse", "case-insensitive"); return; }
    if (tn_color_mode_parse("bogus")  != TN_COLOR_AUTO)   { FAIL("mode_parse", "unrecognized -> auto"); return; }
    if (tn_color_mode_parse(NULL)     != TN_COLOR_AUTO)   { FAIL("mode_parse", "NULL -> auto"); return; }
    PASS("mode_parse");
}

static void test_resolve_always_never(void) {
    if (!tn_color_resolve(TN_COLOR_ALWAYS, 0, "1")) { FAIL("resolve_always_never", "ALWAYS must win even non-tty+NO_COLOR"); return; }
    if (tn_color_resolve(TN_COLOR_NEVER, 1, NULL)) { FAIL("resolve_always_never", "NEVER must win even tty"); return; }
    PASS("resolve_always_never");
}

static void test_resolve_auto_tty(void) {
    if (!tn_color_resolve(TN_COLOR_AUTO, 1, NULL)) { FAIL("resolve_auto_tty", "tty+no NO_COLOR -> enabled"); return; }
    if (tn_color_resolve(TN_COLOR_AUTO, 0, NULL)) { FAIL("resolve_auto_tty", "non-tty -> disabled"); return; }
    PASS("resolve_auto_tty");
}

static void test_resolve_auto_no_color_env(void) {
    if (tn_color_resolve(TN_COLOR_AUTO, 1, "1")) { FAIL("resolve_auto_no_color_env", "NO_COLOR=1 must disable"); return; }
    if (tn_color_resolve(TN_COLOR_AUTO, 1, "anything")) { FAIL("resolve_auto_no_color_env", "any non-empty NO_COLOR disables"); return; }
    if (!tn_color_resolve(TN_COLOR_AUTO, 1, "")) { FAIL("resolve_auto_no_color_env", "empty NO_COLOR must NOT disable"); return; }
    PASS("resolve_auto_no_color_env");
}

static void test_tn_color_helper(void) {
    if (strcmp(tn_color(1, TN_ANSI_BOLD), TN_ANSI_BOLD) != 0) { FAIL("tn_color_helper", "enabled should return code"); return; }
    if (strcmp(tn_color(0, TN_ANSI_BOLD), "") != 0) { FAIL("tn_color_helper", "disabled should return empty string"); return; }
    PASS("tn_color_helper");
}

int main(void) {
    printf("=== Phase 22.3 Color Tests ===\n");
    test_mode_parse();
    test_resolve_always_never();
    test_resolve_auto_tty();
    test_resolve_auto_no_color_env();
    test_tn_color_helper();

    printf("\n");
    if (g_failures == 0) { printf("=== All color tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
