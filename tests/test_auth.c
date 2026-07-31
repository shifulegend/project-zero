/*
 * tests/test_auth.c
 *
 * Phase 22 — API-key auth check.
 */

#include <stdio.h>
#include <string.h>
#include "api/auth.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_disabled_always_passes(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    if (!auth_check_request(&cfg, NULL)) { FAIL("disabled_always_passes", "NULL header, no key set"); return; }
    if (!auth_check_request(&cfg, "Bearer whatever")) { FAIL("disabled_always_passes", "any header, no key set"); return; }
    PASS("disabled_always_passes");
}

static void test_correct_key(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.api_key = "sekret123";
    if (!auth_check_request(&cfg, "Bearer sekret123")) { FAIL("correct_key", "should pass"); return; }
    PASS("correct_key");
}

static void test_incorrect_key(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.api_key = "sekret123";
    if (auth_check_request(&cfg, "Bearer wrong")) { FAIL("incorrect_key", "should fail"); return; }
    PASS("incorrect_key");
}

static void test_missing_header(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.api_key = "sekret123";
    if (auth_check_request(&cfg, NULL)) { FAIL("missing_header", "should fail when key required"); return; }
    PASS("missing_header");
}

static void test_missing_bearer_prefix(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.api_key = "sekret123";
    if (auth_check_request(&cfg, "sekret123")) { FAIL("missing_bearer_prefix", "should require 'Bearer ' prefix"); return; }
    PASS("missing_bearer_prefix");
}

static void test_length_mismatch_no_crash(void) {
    AuthConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.api_key = "a-fairly-long-api-key-value";
    if (auth_check_request(&cfg, "Bearer short")) { FAIL("length_mismatch_no_crash", "should fail"); return; }
    if (auth_check_request(&cfg, "Bearer ")) { FAIL("length_mismatch_no_crash", "empty presented key should fail"); return; }
    PASS("length_mismatch_no_crash");
}

int main(void) {
    printf("=== Phase 22 Auth Tests ===\n");
    test_disabled_always_passes();
    test_correct_key();
    test_incorrect_key();
    test_missing_header();
    test_missing_bearer_prefix();
    test_length_mismatch_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All auth tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
