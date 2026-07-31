/*
 * tests/test_static_assets.c
 *
 * Phase 22.2 — web UI asset manifest lookup/SPA-fallback logic.
 * Uses a hand-written 2-entry fake manifest, NOT the real embedded webui
 * bundle — keeps this test independent of whatever happens to be built
 * into src/api/webui_bundle_generated.c at any given time, and keeps
 * `make test` Node-free.
 */

#include <stdio.h>
#include <string.h>
#include "api/static_assets.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static const unsigned char FAKE_HTML[] = "<html>fake index</html>";
static const unsigned char FAKE_JS[]   = "console.log('fake');";

static const WebuiAsset FAKE_MANIFEST[] = {
    { "/", FAKE_HTML, sizeof(FAKE_HTML) - 1, "text/html" },
    { "/assets/app.js", FAKE_JS, sizeof(FAKE_JS) - 1, "application/javascript" },
};
static const size_t FAKE_COUNT = sizeof(FAKE_MANIFEST) / sizeof(FAKE_MANIFEST[0]);

static void test_exact_match_root(void) {
    const WebuiAsset *a = static_assets_find_in(FAKE_MANIFEST, FAKE_COUNT, "/");
    if (!a || strcmp(a->mime_type, "text/html") != 0) { FAIL("exact_match_root", "expected index.html"); return; }
    PASS("exact_match_root");
}

static void test_exact_match_asset(void) {
    const WebuiAsset *a = static_assets_find_in(FAKE_MANIFEST, FAKE_COUNT, "/assets/app.js");
    if (!a || strcmp(a->mime_type, "application/javascript") != 0) { FAIL("exact_match_asset", "expected app.js"); return; }
    PASS("exact_match_asset");
}

static void test_unknown_asset_returns_null(void) {
    const WebuiAsset *a = static_assets_find_in(FAKE_MANIFEST, FAKE_COUNT, "/assets/missing.css");
    if (a != NULL) { FAIL("unknown_asset_returns_null", "expected 404 (NULL), not a fallback"); return; }
    PASS("unknown_asset_returns_null");
}

static void test_unknown_top_level_falls_back_to_index(void) {
    /* Client-side SPA routes (e.g. a future "/chat/123") aren't real files —
     * must fall back to "/" so the app's own router can handle them. */
    const WebuiAsset *a = static_assets_find_in(FAKE_MANIFEST, FAKE_COUNT, "/some/spa/route");
    if (!a || strcmp(a->mime_type, "text/html") != 0) { FAIL("unknown_top_level_falls_back_to_index", "expected index.html fallback"); return; }
    PASS("unknown_top_level_falls_back_to_index");
}

static void test_empty_manifest_no_crash(void) {
    const WebuiAsset *a = static_assets_find_in(FAKE_MANIFEST, 0, "/");
    if (a != NULL) { FAIL("empty_manifest_no_crash", "expected NULL with zero-length manifest"); return; }
    PASS("empty_manifest_no_crash");
}

int main(void) {
    printf("=== Phase 22.2 Static Assets Tests ===\n");
    test_exact_match_root();
    test_exact_match_asset();
    test_unknown_asset_returns_null();
    test_unknown_top_level_falls_back_to_index();
    test_empty_manifest_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All static assets tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
