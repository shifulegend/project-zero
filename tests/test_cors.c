/*
 * tests/test_cors.c
 *
 * Phase 22 — CORS origin matching + header formatting.
 */

#include <stdio.h>
#include <string.h>
#include "api/cors.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_disabled_by_default(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    if (cors_is_origin_allowed(&cfg, "http://example.com")) {
        FAIL("disabled_by_default", "should be disallowed when cors.enabled == 0");
        return;
    }
    PASS("disabled_by_default");
}

static void test_exact_match(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = 1;
    cfg.origins[0] = "http://localhost:5173";
    cfg.num_origins = 1;

    if (!cors_is_origin_allowed(&cfg, "http://localhost:5173")) {
        FAIL("exact_match", "expected allowed"); return;
    }
    if (cors_is_origin_allowed(&cfg, "http://localhost:9999")) {
        FAIL("exact_match", "unexpected allow for different origin"); return;
    }
    PASS("exact_match");
}

static void test_wildcard(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = 1;
    cfg.origins[0] = "*";
    cfg.num_origins = 1;

    if (!cors_is_origin_allowed(&cfg, "http://anything.example")) {
        FAIL("wildcard", "expected wildcard to allow any origin"); return;
    }
    PASS("wildcard");
}

static void test_null_origin_header(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = 1;
    cfg.origins[0] = "*";
    cfg.num_origins = 1;

    if (cors_is_origin_allowed(&cfg, NULL)) {
        FAIL("null_origin_header", "NULL Origin header must never be allowed"); return;
    }
    PASS("null_origin_header");
}

static void test_multi_origin_parse(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = 1;
    cfg.origins[0] = "http://a.example";
    cfg.origins[1] = "http://b.example";
    cfg.num_origins = 2;

    if (!cors_is_origin_allowed(&cfg, "http://b.example")) {
        FAIL("multi_origin_parse", "second origin should match"); return;
    }
    if (cors_is_origin_allowed(&cfg, "http://c.example")) {
        FAIL("multi_origin_parse", "unlisted origin must not match"); return;
    }
    PASS("multi_origin_parse");
}

static void test_format_headers_disabled_writes_nothing(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    char buf[256];
    size_t n = cors_format_headers(&cfg, "http://example.com", buf, sizeof(buf));
    if (n != 0 || buf[0] != '\0') {
        FAIL("format_headers_disabled", "expected empty output when disabled"); return;
    }
    PASS("format_headers_disabled_writes_nothing");
}

static void test_format_headers_allowed(void) {
    CorsConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = 1;
    cfg.origins[0] = "*";
    cfg.num_origins = 1;

    char buf[256];
    size_t n = cors_format_headers(&cfg, "http://example.com", buf, sizeof(buf));
    if (n == 0) { FAIL("format_headers_allowed", "expected non-empty headers"); return; }
    if (!strstr(buf, "Access-Control-Allow-Origin: http://example.com")) {
        FAIL("format_headers_allowed", "missing Allow-Origin header"); return;
    }
    if (!strstr(buf, "\r\n")) { FAIL("format_headers_allowed", "lines must be CRLF-terminated"); return; }
    PASS("format_headers_allowed");
}

int main(void) {
    printf("=== Phase 22 CORS Tests ===\n");
    test_disabled_by_default();
    test_exact_match();
    test_wildcard();
    test_null_origin_header();
    test_multi_origin_parse();
    test_format_headers_disabled_writes_nothing();
    test_format_headers_allowed();

    printf("\n");
    if (g_failures == 0) { printf("=== All CORS tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
