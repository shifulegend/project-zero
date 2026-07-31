/*
 * Phase 22 — src/api/metrics.c
 * See include/api/metrics.h for responsibility.
 */

#include "api/metrics.h"
#include <stdio.h>
#include <string.h>

void metrics_init(MetricsState *m) {
    if (!m) return;
    atomic_init(&m->requests_total, 0);
    atomic_init(&m->requests_2xx, 0);
    atomic_init(&m->requests_4xx, 0);
    atomic_init(&m->requests_5xx, 0);
    atomic_init(&m->tokens_generated_total, 0);
    atomic_init(&m->active_connections, 0);
    atomic_init(&m->generations_cancelled_total, 0);
    m->start_time = time(NULL);
}

void metrics_record_request(MetricsState *m, int status_code) {
    if (!m) return;
    atomic_fetch_add(&m->requests_total, 1);
    if (status_code >= 200 && status_code < 300) atomic_fetch_add(&m->requests_2xx, 1);
    else if (status_code >= 400 && status_code < 500) atomic_fetch_add(&m->requests_4xx, 1);
    else if (status_code >= 500) atomic_fetch_add(&m->requests_5xx, 1);
}

void metrics_add_tokens(MetricsState *m, long long n) {
    if (!m) return;
    atomic_fetch_add(&m->tokens_generated_total, n);
}

void metrics_connection_opened(MetricsState *m) {
    if (!m) return;
    atomic_fetch_add(&m->active_connections, 1);
}

void metrics_connection_closed(MetricsState *m) {
    if (!m) return;
    atomic_fetch_sub(&m->active_connections, 1);
}

void metrics_record_cancel(MetricsState *m) {
    if (!m) return;
    atomic_fetch_add(&m->generations_cancelled_total, 1);
}

size_t metrics_render(const MetricsState *m, char *buf, size_t cap) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    if (!m) return 0;

    long long uptime = (long long)(time(NULL) - m->start_time);
    int n = snprintf(buf, cap,
        "# HELP project_zero_uptime_seconds Seconds since the API server started.\n"
        "# TYPE project_zero_uptime_seconds counter\n"
        "project_zero_uptime_seconds %lld\n"
        "# HELP project_zero_requests_total Total HTTP requests handled, by status class.\n"
        "# TYPE project_zero_requests_total counter\n"
        "project_zero_requests_total{status=\"2xx\"} %lld\n"
        "project_zero_requests_total{status=\"4xx\"} %lld\n"
        "project_zero_requests_total{status=\"5xx\"} %lld\n"
        "# HELP project_zero_tokens_generated_total Total tokens generated across all requests.\n"
        "# TYPE project_zero_tokens_generated_total counter\n"
        "project_zero_tokens_generated_total %lld\n"
        "# HELP project_zero_active_connections Currently open HTTP connections.\n"
        "# TYPE project_zero_active_connections gauge\n"
        "project_zero_active_connections %d\n"
        "# HELP project_zero_generations_cancelled_total Total generations stopped via the cancel endpoint.\n"
        "# TYPE project_zero_generations_cancelled_total counter\n"
        "project_zero_generations_cancelled_total %lld\n",
        uptime,
        (long long)atomic_load(&m->requests_2xx),
        (long long)atomic_load(&m->requests_4xx),
        (long long)atomic_load(&m->requests_5xx),
        (long long)atomic_load(&m->tokens_generated_total),
        atomic_load(&m->active_connections),
        (long long)atomic_load(&m->generations_cancelled_total));
    if (n < 0) { buf[0] = '\0'; return 0; }
    if ((size_t)n >= cap) return cap - 1; /* truncated but still NUL-terminated by snprintf */
    return (size_t)n;
}
