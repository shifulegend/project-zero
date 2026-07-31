#ifndef TN_METRICS_H
#define TN_METRICS_H

/*
 * Phase 22 — Prometheus-style metrics for GET /metrics.
 * Off by default (nothing increments the counters' visibility, but tracking
 * itself is always-on and cheap; --metrics only gates whether the route
 * exists and returns data).
 */

#include <stdatomic.h>
#include <stddef.h>
#include <time.h>

typedef struct {
    atomic_llong requests_total;
    atomic_llong requests_2xx;
    atomic_llong requests_4xx;
    atomic_llong requests_5xx;
    atomic_llong tokens_generated_total;
    atomic_int   active_connections;
    atomic_llong generations_cancelled_total;
    time_t       start_time;
} MetricsState;

void metrics_init(MetricsState *m);
void metrics_record_request(MetricsState *m, int status_code);
void metrics_add_tokens(MetricsState *m, long long n);
void metrics_connection_opened(MetricsState *m);
void metrics_connection_closed(MetricsState *m);
void metrics_record_cancel(MetricsState *m);

/* Renders Prometheus text-exposition format into buf (truncates safely,
 * always NUL-terminated if cap > 0). Returns the number of bytes written
 * (excluding the NUL terminator). */
size_t metrics_render(const MetricsState *m, char *buf, size_t cap);

#endif /* TN_METRICS_H */
