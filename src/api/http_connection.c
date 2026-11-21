/* One accepted connection: read, gate, dispatch, record, close (see api/http_connection.h). */

#include "api/http_connection.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api/auth.h"
#include "api/config_endpoint.h"
#include "api/cors.h"
#include "api/http_headers.h"
#include "api/http_request.h"
#include "api/http_response.h"
#include "api/http_routes.h"
#include "api/http_status.h"
#include "api/metrics.h"
#include "api/rate_limit.h"
#include "api/static_assets.h"
#include "api/update_endpoint.h"
#include "core/debug.h"
#include "core/log.h"

/* Fixed buffer capacities (bytes). */
#define HTTP_EXTRA_HDRS_BUF 640 /* CORS headers + X-Request-ID added to a response */
#define HTTP_RETRY_HDRS_BUF (HTTP_EXTRA_HDRS_BUF + 128) /* extra headers + Retry-After */

/* Refuses the request: records the status in the metrics and sends the error. */
static void refuse(int fd, ApiContext *ctx, int status, const char *message, const char *headers) {
    metrics_record_request(&ctx->metrics, status);
    http_send_error_ex(fd, status, message, headers);
}

/* Each gate returns 1 to let the request on, 0 after it has sent a refusal. */

/* DNS-rebinding guard (everything except /health, which monitors probe). */
static int gate_host(int fd, ApiContext *ctx, const HttpRequest *req, const char *headers) {
    if (api_host_allowed(req->host, &ctx->server_config.cors)) return 1;
    refuse(fd, ctx, HTTP_FORBIDDEN,
           "Host not allowed (this server answers to localhost / 127.0.0.1; "
           "behind a proxy add its address with --cors --cors-origin)",
           headers);
    return 0;
}

static int gate_api_key(int fd, ApiContext *ctx, const HttpRequest *req, const char *headers) {
    if (auth_check_request(&ctx->server_config.auth,
                           req->has_authorization ? req->authorization : NULL))
        return 1;
    refuse(fd, ctx, HTTP_UNAUTHORIZED, "Unauthorized", headers);
    return 0;
}

/* Rate limit the cheap non-generation API routes (generation has its own
 * queue; /health, /metrics and the web UI's static files are not limited). */
static int gate_rate_limit(int fd, ApiContext *ctx, const HttpRequest *req, const char *headers) {
    if (strncmp(req->path, "/v1/", 4) != 0 || strncmp(req->path, "/v1/chat/completions", 20) == 0)
        return 1;
    int retry = 1;
    if (rate_limit_take(&ctx->rate_limiter, rate_limit_now(), &retry)) return 1;
    char hdrs[HTTP_RETRY_HDRS_BUF];
    snprintf(hdrs, sizeof hdrs, "%sRetry-After: %d\r\n", headers, retry);
    refuse(fd, ctx, HTTP_TOO_MANY_REQUESTS, "Too many requests; slow down", hdrs);
    return 0;
}

/* Cross-site guard for every mutating route (the web UI and API clients
 * pass; a foreign web page cannot, because browsers always attach Origin). */
static int gate_cross_site(int fd, ApiContext *ctx, const HttpRequest *req, const char *headers) {
    if (strcmp(req->method, "POST") != 0) return 1;
    if (api_post_allowed(req->content_type, req->has_origin ? req->origin : NULL, req->host,
                         &ctx->server_config.cors, req->body_len))
        return 1;
    refuse(fd, ctx, HTTP_FORBIDDEN,
           "Cross-site request refused (the Origin is not this server or "
           "an allowed --cors-origin)",
           headers);
    return 0;
}

/* The gates, in order. Returns 1 when the request may be routed; 0 when a reply (refusal or
 * preflight answer) has already been sent. /health is exempt from the Host and API-key gates. */
static int pass_gates(int fd, ApiContext *ctx, const HttpRequest *req, const char *headers) {
    int is_health = strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0;
    if (!is_health && !gate_host(fd, ctx, req, headers)) return 0;
    if (strcmp(req->method, "OPTIONS") == 0) {
        http_send_response_ex(fd, HTTP_NO_CONTENT, "text/plain", NULL, 0, headers);
        return 0;
    }
    if (!is_health && !gate_api_key(fd, ctx, req, headers)) return 0;
    return gate_rate_limit(fd, ctx, req, headers) && gate_cross_site(fd, ctx, req, headers);
}

/* ── Routes ──────────────────────────────────────────────────────────────── */
/* Every handler takes the same arguments and returns the status to record. */

typedef struct {
    int fd;
    ApiContext *ctx;
    const HttpRequest *req;
    const char *extra_headers;
} RouteCall;

static int route_models(const RouteCall *c) {
    http_handle_get_models(c->fd, c->ctx, c->extra_headers);
    return HTTP_OK;
}
static int route_config(const RouteCall *c) {
    http_handle_get_config(c->fd, c->ctx, c->extra_headers);
    return HTTP_OK;
}
static int route_config_save(const RouteCall *c) {
    const HttpRequest *r = c->req;
    return http_handle_config_save(c->fd, r->body, r->body_len, r->content_type,
                                   r->has_origin ? r->origin : NULL, r->host, c->ctx,
                                   c->extra_headers);
}
static int route_update(const RouteCall *c) {
    http_send_json_heap(c->fd, HTTP_OK, api_update_json(c->ctx->update, 0), c->extra_headers);
    return HTTP_OK;
}
static int route_update_check(const RouteCall *c) {
    http_send_json_heap(c->fd, HTTP_OK, api_update_json(c->ctx->update, 1), c->extra_headers);
    return HTTP_OK;
}
static int route_cache_stats(const RouteCall *c) {
    http_handle_prompt_cache_stats(c->fd, c->ctx, c->extra_headers);
    return HTTP_OK;
}
static int route_cache_clear(const RouteCall *c) {
    return http_handle_prompt_cache_clear(c->fd, c->ctx, c->extra_headers);
}
static int route_selfcheck(const RouteCall *c) {
    return http_handle_selfcheck(c->fd, c->req->body, c->req->body_len, c->ctx, c->extra_headers);
}
static int route_watermark(const RouteCall *c) {
    return http_handle_watermark_detect(c->fd, c->req->body, c->req->body_len, c->ctx,
                                        c->extra_headers);
}
static int route_health(const RouteCall *c) {
    /* Always available, regardless of --web-ui; "/" serves the web UI (Phase 22.2). */
    http_handle_health(c->fd, c->extra_headers);
    return HTTP_OK;
}
static int route_metrics(const RouteCall *c) {
    if (c->ctx->server_config.metrics.enabled) {
        http_handle_get_metrics(c->fd, c->ctx, c->extra_headers);
        return HTTP_OK;
    }
    http_send_error_ex(c->fd, HTTP_NOT_FOUND, "Metrics disabled (start with --metrics)",
                       c->extra_headers);
    return HTTP_NOT_FOUND;
}
static int route_docs(const RouteCall *c) {
    http_handle_get_docs(c->fd, c->extra_headers);
    return HTTP_OK;
}
static int route_openapi(const RouteCall *c) {
    http_handle_get_openapi(c->fd, c->extra_headers);
    return HTTP_OK;
}
static int route_chat(const RouteCall *c) {
    http_handle_chat_completions(c->fd, c->req->body, c->req->body_len, c->ctx, c->extra_headers);
    return HTTP_OK;
}
static int route_cancel(const RouteCall *c) {
    http_handle_cancel(c->fd, c->req->body, c->req->body_len, c->ctx, c->extra_headers);
    return HTTP_OK;
}

static const struct {
    const char *method;
    const char *path;
    int (*handler)(const RouteCall *);
} k_routes[] = {
    {"GET", "/v1/models", route_models},
    {"GET", "/v1/config", route_config},
    {"POST", "/v1/config/defaults", route_config_save},
    {"GET", "/v1/update", route_update},
    {"POST", "/v1/update/check", route_update_check},
    {"GET", "/v1/prompt_cache", route_cache_stats},
    {"POST", "/v1/prompt_cache/clear", route_cache_clear},
    {"POST", "/v1/selfcheck", route_selfcheck},
    {"POST", "/v1/watermark/detect", route_watermark},
    {"GET", "/health", route_health},
    {"GET", "/metrics", route_metrics},
    {"GET", "/docs", route_docs},
    {"GET", "/openapi.json", route_openapi},
    {"POST", "/v1/chat/completions", route_chat},
    {"POST", "/v1/chat/completions/cancel", route_cancel},
};

/* Runs the matching route and returns the status to record. After every API route, an unmatched
 * GET goes to the web UI (Phase 22.2: "/" and files under "/assets/", also client-side SPA
 * routes) unless it is off; anything else is a 404. */
static int dispatch(int fd, ApiContext *ctx, const HttpRequest *req, const char *extra_headers) {
    RouteCall call = {fd, ctx, req, extra_headers};
    for (size_t i = 0; i < sizeof(k_routes) / sizeof(k_routes[0]); i++)
        if (strcmp(req->method, k_routes[i].method) == 0 &&
            strcmp(req->path, k_routes[i].path) == 0)
            return k_routes[i].handler(&call);
    if (strcmp(req->method, "GET") == 0 && ctx->server_config.ui.web_ui != WEBUI_MODE_OFF)
        return static_assets_serve(fd, req->path, &ctx->server_config, extra_headers);
    http_send_error_ex(fd, HTTP_NOT_FOUND, "Not found", extra_headers);
    return HTTP_NOT_FOUND;
}

void api_handle_connection(int fd, ApiContext *ctx) {
    char *recv_buf       = NULL;
    size_t total         = 0;
    const char *read_err = NULL;
    int read_status      = http_request_read(fd, &recv_buf, &total, &read_err);
    if (read_status != 0) { /* framing refused (or the client went away) */
        if (read_status > 0) http_send_error(fd, read_status, read_err);
        close(fd);
        return;
    }
    HttpRequest req;
    if (!http_parse(recv_buf, total, &req)) {
        http_send_error(fd, HTTP_BAD_REQUEST, "Malformed HTTP request");
        free(recv_buf);
        close(fd);
        return;
    }

    /* CORS headers and the request ID go on every reply. */
    char extra_headers[HTTP_EXTRA_HDRS_BUF];
    extra_headers[0] = '\0';
    if (ctx->server_config.cors.enabled)
        cors_format_headers(&ctx->server_config.cors, req.has_origin ? req.origin : NULL,
                            extra_headers, sizeof(extra_headers));
    http_request_attach_id(&ctx->request_seq, &req, extra_headers, sizeof(extra_headers));

    if (pass_gates(fd, ctx, &req, extra_headers)) {
        int status = dispatch(fd, ctx, &req, extra_headers);
        metrics_record_request(&ctx->metrics, status);
        if (g_tn_verbose) TN_LOG_INF("api", "[API] request %s -> %d\n", req.request_id, status);
    }
    free(recv_buf);
    close(fd);
}
