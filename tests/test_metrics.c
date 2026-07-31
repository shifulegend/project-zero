/*
 * tests/test_metrics.c
 *
 * Phase 22 — Prometheus metrics rendering + counter tracking.
 */

#include <stdio.h>
#include <string.h>
#include "api/metrics.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_render_shape(void) {
    MetricsState m;
    metrics_init(&m);
    char buf[2048];
    size_t n = metrics_render(&m, buf, sizeof(buf));
    if (n == 0) { FAIL("render_shape", "expected non-empty output"); return; }
    if (!strstr(buf, "# HELP")) { FAIL("render_shape", "missing HELP line"); return; }
    if (!strstr(buf, "# TYPE")) { FAIL("render_shape", "missing TYPE line"); return; }
    if (!strstr(buf, "project_zero_requests_total")) { FAIL("render_shape", "missing requests_total metric"); return; }
    PASS("render_shape");
}

static void test_counters_reflect_increments(void) {
    MetricsState m;
    metrics_init(&m);
    metrics_record_request(&m, 200);
    metrics_record_request(&m, 200);
    metrics_record_request(&m, 404);
    metrics_record_request(&m, 500);
    metrics_add_tokens(&m, 42);
    metrics_connection_opened(&m);
    metrics_connection_opened(&m);
    metrics_connection_closed(&m);
    metrics_record_cancel(&m);

    char buf[2048];
    metrics_render(&m, buf, sizeof(buf));
    if (!strstr(buf, "status=\"2xx\"} 2")) { FAIL("counters_reflect_increments", "2xx count wrong"); return; }
    if (!strstr(buf, "status=\"4xx\"} 1")) { FAIL("counters_reflect_increments", "4xx count wrong"); return; }
    if (!strstr(buf, "status=\"5xx\"} 1")) { FAIL("counters_reflect_increments", "5xx count wrong"); return; }
    if (!strstr(buf, "tokens_generated_total 42")) { FAIL("counters_reflect_increments", "tokens count wrong"); return; }
    if (!strstr(buf, "active_connections 1")) { FAIL("counters_reflect_increments", "active_connections wrong"); return; }
    if (!strstr(buf, "generations_cancelled_total 1")) { FAIL("counters_reflect_increments", "cancel count wrong"); return; }
    PASS("counters_reflect_increments");
}

static void test_many_increments_no_overflow(void) {
    MetricsState m;
    metrics_init(&m);
    for (int i = 0; i < 100000; i++) metrics_record_request(&m, 200);
    char buf[2048];
    metrics_render(&m, buf, sizeof(buf));
    if (!strstr(buf, "status=\"2xx\"} 100000")) { FAIL("many_increments_no_overflow", "expected 100000"); return; }
    PASS("many_increments_no_overflow");
}

static void test_tiny_buffer_stays_terminated(void) {
    MetricsState m;
    metrics_init(&m);
    char buf[8];
    size_t n = metrics_render(&m, buf, sizeof(buf));
    if (n >= sizeof(buf)) { FAIL("tiny_buffer_stays_terminated", "return value exceeds cap"); return; }
    if (buf[sizeof(buf)-1] != '\0' && strlen(buf) >= sizeof(buf)) {
        FAIL("tiny_buffer_stays_terminated", "not NUL-terminated within cap"); return;
    }
    PASS("tiny_buffer_stays_terminated");
}

int main(void) {
    printf("=== Phase 22 Metrics Tests ===\n");
    test_render_shape();
    test_counters_reflect_increments();
    test_many_increments_no_overflow();
    test_tiny_buffer_stays_terminated();

    printf("\n");
    if (g_failures == 0) { printf("=== All metrics tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
