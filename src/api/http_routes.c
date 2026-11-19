/*
 * src/api/http_routes.c
 *
 * The small route handlers of the HTTP server (models, config, prompt cache, self-check,
 * watermark, health, metrics, docs, OpenAPI, cancel). Split out of http_server.c; the code
 * is unchanged apart from the `http_` name prefix on the functions it now exports.
 */

#include "api/http_routes.h"
#include "api/http_response.h"
#include "api/http_status.h"
#include "api/config_endpoint.h"
#include "api/api_info.h"
#include "api/batch_scheduler.h"
#include "api/cancel.h"
#include "api/gen_queue.h"
#include "api/metrics.h"
#include "api/openapi.h"
#include "api/selfcheck_endpoint.h"
#include "api/watermark_endpoint.h"
#include "kv_cache/prefix_cache.h"
#include "runtime/prompt_store.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Fixed buffer capacity: rendered /metrics text. */
#define HTTP_METRICS_BUF 4096

/* 2026-10-18: previously a hardcoded {"id":"local-adaptive-engine"} with no
 * "vision" flag, so the web UI's image-upload control (gated on
 * modelInfo.vision) could never appear even with --vision/--proj set. */
void http_handle_get_models(int fd, ApiContext *ctx, const char *extra_headers) {
    http_send_json_heap(fd, HTTP_OK, api_models_json(ctx), extra_headers);
}

void http_handle_get_config(int fd, ApiContext *ctx, const char *extra_headers) {
    http_send_json_heap(fd, HTTP_OK, api_config_json(ctx), extra_headers);
}

/* Phase 23 stage 2b: a batched chat request holds one gen_queue unit (one
 * scheduler slot); anything that needs the whole engine — the sequential
 * generation path, image input, best-of, tree search, cache reset — holds
 * all of them plus generation_mutex, so the scheduler is idle meanwhile. */
int http_engine_units(ApiContext *ctx) {
    return gen_queue_capacity(&ctx->gen_queue);
}

/* Releases what handle_chat_completions() / prompt-cache clear took:
 * `units` gen_queue units, and generation_mutex when it held the engine. */
void http_release_generation_slot(ApiContext *ctx, int units, int exclusive) {
    if (exclusive) pthread_mutex_unlock(&ctx->generation_mutex);
    gen_queue_release_n(&ctx->gen_queue, units);
}

/* GET /v1/prompt_cache — Phase 26 store stats. Reads counters without the
 * generation mutex (word-sized, informational only). */
void http_handle_prompt_cache_stats(int fd, ApiContext *ctx, const char *extra_headers) {
    http_send_json_heap(fd, HTTP_OK, api_prompt_cache_json(ctx), extra_headers);
}

/* POST /v1/prompt_cache/clear — drops every snapshot and the live prefix
 * record. Mutates generation state, so it takes generation_mutex (429 while
 * a generation is running, same as a second chat request). */
int http_handle_prompt_cache_clear(int fd, ApiContext *ctx, const char *extra_headers) {
    int units = http_engine_units(ctx);
    if (gen_queue_try_acquire_n(&ctx->gen_queue, units) != GEN_QUEUE_OK) {
        http_send_error_ex(fd, HTTP_TOO_MANY_REQUESTS, "A generation is in flight", extra_headers);
        return HTTP_TOO_MANY_REQUESTS;
    }
    pthread_mutex_lock(&ctx->generation_mutex);
    prompt_store_clear(&ctx->prompt_store);
    prefix_cache_reset(&ctx->prefix_cache);
    batch_scheduler_reset_caches(ctx->sched); /* every parallel slot's prefix too */
    http_release_generation_slot(ctx, units, 1);
    http_send_json_heap(fd, HTTP_OK, api_prompt_cache_json(ctx), extra_headers);
    return HTTP_OK;
}

/* POST /v1/selfcheck — the accuracy probe on the loaded model. It runs forward passes on the
 * engine's own state, so like a cache reset it takes every generation unit plus the mutex
 * (429 while a generation is running) and drops the prefix caches that the probe overwrites. */
int http_handle_selfcheck(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                          const char *extra_headers) {
    char *body = (char *)malloc(body_len + 1);
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return HTTP_INTERNAL_ERROR;
    }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';
    int units      = http_engine_units(ctx);
    if (gen_queue_try_acquire_n(&ctx->gen_queue, units) != GEN_QUEUE_OK) {
        free(body);
        http_send_error_ex(fd, HTTP_TOO_MANY_REQUESTS, "A generation is in flight", extra_headers);
        return HTTP_TOO_MANY_REQUESTS;
    }
    pthread_mutex_lock(&ctx->generation_mutex);
    int status = HTTP_INTERNAL_ERROR;
    char *json = api_selfcheck(ctx, body, &status);
    prompt_store_clear(&ctx->prompt_store);
    prefix_cache_reset(&ctx->prefix_cache);
    batch_scheduler_reset_caches(ctx->sched);
    http_release_generation_slot(ctx, units, 1);
    free(body);
    http_send_json_heap(fd, status, json, extra_headers);
    return status;
}

int http_handle_watermark_detect(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                                 const char *extra_headers) {
    char *body = (char *)malloc(body_len + 1);
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return HTTP_INTERNAL_ERROR;
    }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';
    int status     = HTTP_INTERNAL_ERROR;
    char *json     = api_watermark_detect(ctx, body, &status);
    free(body);
    http_send_json_heap(fd, status, json, extra_headers);
    return status;
}

/* Phase 40.3: save a server default to the config file (never takes the
 * generation mutex). */
int http_handle_config_save(int fd, const char *body_start, size_t body_len,
                            const char *content_type, const char *origin, const char *host,
                            ApiContext *ctx, const char *extra_headers) {
    if (!api_config_write_allowed(content_type, origin, host, &ctx->server_config.cors)) {
        http_send_error_ex(fd, HTTP_FORBIDDEN,
                           "Cross-site request refused (send Content-Type: application/json "
                           "from the server's own origin)",
                           extra_headers);
        return HTTP_FORBIDDEN;
    }
    char *body = (char *)malloc(body_len + 1);
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return HTTP_INTERNAL_ERROR;
    }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';
    int status     = HTTP_INTERNAL_ERROR;
    char *json     = api_config_save(ctx, body, &status);
    free(body);
    http_send_json_heap(fd, status, json, extra_headers);
    return status;
}

void http_handle_health(int fd, const char *extra_headers) {
    http_send_response_ex(fd, HTTP_OK, "application/json", "{\"status\":\"ok\"}", 0, extra_headers);
}

void http_handle_get_metrics(int fd, ApiContext *ctx, const char *extra_headers) {
    char buf[HTTP_METRICS_BUF];
    size_t n = metrics_render(&ctx->metrics, buf, sizeof(buf));
    /* Phase 23: generation slots (gen_queue units) and the parallel scheduler. */
    BatchSchedulerStats bst;
    batch_scheduler_stats(ctx->sched, &bst);
    if (n < sizeof(buf))
        snprintf(
            buf + n, sizeof(buf) - n,
            "# HELP project_zero_generation_slots Requests the server generates at once "
            "(--parallel).\n"
            "# TYPE project_zero_generation_slots gauge\n"
            "project_zero_generation_slots %d\n"
            "# HELP project_zero_generation_slots_busy Generation slots in use.\n"
            "# TYPE project_zero_generation_slots_busy gauge\n"
            "project_zero_generation_slots_busy %d\n"
            "# HELP project_zero_queue_waiting Chat requests waiting for a slot.\n"
            "# TYPE project_zero_queue_waiting gauge\n"
            "project_zero_queue_waiting %d\n"
            "# HELP project_zero_parallel_kv_tokens KV-cache positions held by parallel slots.\n"
            "# TYPE project_zero_parallel_kv_tokens gauge\n"
            "project_zero_parallel_kv_tokens %lld\n",
            ctx->parallel_slots > 0 ? ctx->parallel_slots : 1, gen_queue_in_use(&ctx->gen_queue),
            gen_queue_waiting(&ctx->gen_queue), bst.kv_committed_tokens);
    http_send_response_ex(fd, HTTP_OK, "text/plain; version=0.0.4", buf, 0, extra_headers);
}

void http_handle_get_docs(int fd, const char *extra_headers) {
    http_send_response_ex(fd, HTTP_OK, "text/html", openapi_docs_html(), 0, extra_headers);
}

void http_handle_get_openapi(int fd, const char *extra_headers) {
    http_send_response_ex(fd, HTTP_OK, "application/json", openapi_json(), 0, extra_headers);
}

/* Minimal extraction of {"id":"..."} — the cancel endpoint's only field.
 * Deliberately not routed through json_parse.c (whose helpers are file-
 * static and scoped to the fuller ChatRequest grammar); this mirrors the
 * lightweight hand-rolled scanning style already used by http_parse(). */
static int extract_id_field(const char *json_body, char *out, size_t out_cap) {
    if (!json_body || !out || out_cap == 0) return 0;
    const char *key = strstr(json_body, "\"id\"");
    if (!key) return 0;
    const char *colon = strchr(key, ':');
    if (!colon) return 0;
    const char *p = colon + 1;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < out_cap - 1) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

void http_handle_cancel(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                        const char *extra_headers) {
    char *body = (char *)malloc(body_len + 1);
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return;
    }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';

    char id[32];
    if (!extract_id_field(body, id, sizeof(id))) {
        free(body);
        http_send_error_ex(fd, HTTP_BAD_REQUEST, "Missing 'id' field", extra_headers);
        return;
    }
    free(body);

    int matched = cancel_registry_request(&ctx->cancel, id);
    if (matched) {
        metrics_record_cancel(&ctx->metrics);
        http_send_response_ex(fd, HTTP_OK, "application/json", "{\"cancelled\":true}", 0,
                              extra_headers);
    } else {
        http_send_error_ex(fd, HTTP_NOT_FOUND, "No matching in-flight request", extra_headers);
    }
}
