/*
 * tests/test_openapi.c
 *
 * Phase 22 — Static OpenAPI JSON + docs page content.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/openapi.h"

#define PASS(name) printf("[PASS] %s\n", name)
#define FAIL(name, msg)                                                                            \
    do {                                                                                           \
        printf("[FAIL] %s: %s\n", name, msg);                                                      \
        g_failures++;                                                                              \
    } while (0)

static int g_failures = 0;

static void test_openapi_json_has_routes(void) {
    char *json = openapi_json();
    if (!json) {
        FAIL("openapi_json_has_routes", "NULL json");
        return;
    }
    const char *routes[] = {"/v1/models",
                            "/v1/chat/completions",
                            "/v1/chat/completions/cancel",
                            "/health",
                            "/metrics",
                            "/docs",
                            "/v1/config",
                            "/v1/watermark/detect",
                            "/v1/prompt_cache"};
    for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); i++) {
        if (!strstr(json, routes[i])) {
            char msg[128];
            snprintf(msg, sizeof(msg), "missing route %s", routes[i]);
            FAIL("openapi_json_has_routes", msg);
            free(json);
            return;
        }
    }
    free(json);
    PASS("openapi_json_has_routes");
}

static void test_openapi_json_is_valid_shape(void) {
    char *json  = openapi_json();
    char *again = openapi_json();
    if (!json || !again || strcmp(json, again) != 0 || json == again) {
        FAIL("openapi_json_is_valid_shape", "each call must return its own identical copy");
    } else if (!strstr(json, "\"openapi\":\"3.0.0\"")) {
        FAIL("openapi_json_is_valid_shape", "missing openapi version field");
    } else if (!strstr(json, "\"paths\":{")) {
        FAIL("openapi_json_is_valid_shape", "missing paths object");
    } else {
        PASS("openapi_json_is_valid_shape");
    }
    free(json);
    free(again);
}

static void test_docs_html_references_openapi(void) {
    const char *html = openapi_docs_html();
    if (!html) {
        FAIL("docs_html_references_openapi", "NULL html");
        return;
    }
    if (!strstr(html, "/openapi.json")) {
        FAIL("docs_html_references_openapi", "docs page must fetch /openapi.json");
        return;
    }
    if (!strstr(html, "<html>")) {
        FAIL("docs_html_references_openapi", "not a valid HTML page");
        return;
    }
    PASS("docs_html_references_openapi");
}

int main(void) {
    printf("=== Phase 22 OpenAPI Tests ===\n");
    test_openapi_json_has_routes();
    test_openapi_json_is_valid_shape();
    test_docs_html_references_openapi();

    printf("\n");
    if (g_failures == 0) {
        printf("=== All openapi tests passed ===\n");
        return 0;
    }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
