#ifndef TN_API_HTTP_ROUTES_H
#define TN_API_HTTP_ROUTES_H

/* The small route handlers of the HTTP server (src/api/http_routes.c). The ones returning int
 * return the HTTP status they sent; `extra_headers` is the CORS block or NULL. */

#include <stddef.h>

#include "api/api_server.h"

void http_handle_get_models(int fd, ApiContext *ctx, const char *extra_headers);
void http_handle_get_config(int fd, ApiContext *ctx, const char *extra_headers);
void http_handle_prompt_cache_stats(int fd, ApiContext *ctx, const char *extra_headers);
int http_handle_prompt_cache_clear(int fd, ApiContext *ctx, const char *extra_headers);
int http_handle_selfcheck(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                          const char *extra_headers);
int http_handle_watermark_detect(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                                 const char *extra_headers);
int http_handle_config_save(int fd, const char *body_start, size_t body_len,
                            const char *content_type, const char *origin, const char *host,
                            ApiContext *ctx, const char *extra_headers);
/* POST /v1/chat/completions: the whole generation request, queue wait included. */
void http_handle_chat_completions(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                                  const char *extra_headers);
void http_handle_health(int fd, const char *extra_headers);
void http_handle_get_metrics(int fd, ApiContext *ctx, const char *extra_headers);
void http_handle_get_docs(int fd, const char *extra_headers);
void http_handle_get_openapi(int fd, const char *extra_headers);
void http_handle_cancel(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                        const char *extra_headers);

/* Generation admission: units a request that needs the whole engine must hold, and the release
 * that pairs with gen_queue acquisition (also unlocks generation_mutex when `exclusive`). */
int http_engine_units(ApiContext *ctx);
void http_release_generation_slot(ApiContext *ctx, int units, int exclusive);

#endif
