/*
 * Phase 21 — OpenAI-Compatible API Layer
 * Phase 22 — CORS/auth/metrics/OpenAPI/cancel hardening + concurrency
 *            rearchitecture
 * src/api/http_server.c
 *
 * Minimal POSIX HTTP/1.1 server using only standard system calls (socket,
 * bind, listen, accept, read, write).  No external dependencies beyond
 * what the rest of the engine already uses.
 *
 * Supported routes
 * ────────────────
 *   GET  /v1/models                       → JSON model list
 *   POST /v1/chat/completions             → Chat completion (streaming + non-streaming;
 *                                            image_url content parts run the vision
 *                                            pipeline when --vision/--proj are set)
 *   POST /v1/chat/completions/cancel      → Cancel the in-flight generation
 *   GET  /health                          → {"status":"ok"} (never gated by auth)
 *   GET  /metrics                         → Prometheus text exposition (if --metrics)
 *   GET  /docs                            → Hand-rolled API docs page
 *   GET  /openapi.json                    → Static OpenAPI 3.0 description
 *   GET  / and files under /assets/        → Web chat UI (Phase 22.2, unless --web-ui off)
 *   OPTIONS *                             → CORS preflight (if --cors)
 *
 * Threading model (Phase 22)
 * ───────────────────────────
 * The listener thread calls accept() in a loop and spawns a detached
 * per-connection thread for each accepted socket, so static/metrics/docs/
 * cancel requests are never blocked behind an in-flight generation. Only
 * one generate_with_callback() call may run at a time — routes that
 * generate acquire ctx->generation_mutex via trylock and return 429
 * immediately if it's already held (no queueing).
 *
 * Security notes
 * ──────────────
 *   • Listens only on 127.0.0.1 (loopback) by default — not exposed to LAN.
 *   • Request body size is capped at HTTP_MAX_BODY_BYTES.
 *   • Buffer overflows are guarded with explicit size checks.
 *   • CORS and API-key auth are both OFF by default (opt-in via --cors /
 *     --api-key) so existing loopback workflows are unaffected.
 */

/* _GNU_SOURCE for strcasestr / strncasecmp on Linux —
 * defined before any system header to take effect. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "api/api_server.h"
#include "api/chat_request.h"
#include "api/chat_compile.h"
#include "api/sse_stream.h"
#include "api/cors.h"
#include "api/auth.h"
#include "api/metrics.h"
#include "api/openapi.h"
#include "api/cancel.h"
#include "api/static_assets.h"
#include "api/data_url.h"
#include "multimodal/vision_pipeline.h"
#include "transformer/generate.h"
#include "core/debug.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>

/* ── Tuning constants ────────────────────────────────────────────────────── */
#define HTTP_RECV_BUF        8192       /* receive buffer per connection */
#define HTTP_MAX_BODY_BYTES  (8*1024*1024) /* 8 MiB max request body — raised from
                                             * 512 KiB (Phase 21) to fit base64
                                             * image uploads (Phase 22.2). */
#define HTTP_MAX_PROMPT_BYTES (65536)   /* compiled prompt size cap       */
#define HTTP_BACKLOG         16

/* ── Simple unique ID generator ─────────────────────────────────────────── */
static void make_id(char *buf, size_t cap) {
    static pthread_mutex_t id_lock = PTHREAD_MUTEX_INITIALIZER;
    static unsigned long counter = 0;
    pthread_mutex_lock(&id_lock);
    unsigned long n = ++counter;
    pthread_mutex_unlock(&id_lock);
    snprintf(buf, cap, "%lx-%lx", (unsigned long)time(NULL), n);
}

/* ── HTTP response helpers ───────────────────────────────────────────────── */

/* extra_headers, if non-NULL, must already be "\r\n"-terminated per line
 * (e.g. from cors_format_headers()) and is spliced in ahead of the blank
 * line that ends the header block. */
static void send_response_ex(int fd, int status, const char *content_type,
                              const char *body, int streaming,
                              const char *extra_headers) {
    char hdr[1024];
    size_t body_len = body ? strlen(body) : 0;
    const char *extra = extra_headers ? extra_headers : "";
    int n;
    if (streaming) {
        /* 2026-07-15: pre-existing bug — this claimed "Transfer-Encoding:
         * chunked" but the SSE writers (sse_write_token/done) write raw
         * unframed bytes, not HTTP chunk-size-prefixed framing. A strict
         * client (browser fetch/EventSource; curl --raw) sees a protocol
         * violation. Since the response always closes the connection when
         * done, RFC 7230 §3.3.3 case 7 already lets EOF delimit the body —
         * no Transfer-Encoding header is needed at all. */
        n = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "%s"
            "\r\n", status, content_type, extra);
    } else {
        n = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "%s"
            "\r\n", status, content_type, body_len, extra);
    }
    if (n > 0) { ssize_t r = write(fd, hdr, (size_t)n); (void)r; }
    if (body && !streaming) { ssize_t r = write(fd, body, body_len); (void)r; }
}

static void send_error_ex(int fd, int status, const char *msg, const char *extra_headers) {
    char body[256];
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"server_error\"}}", msg);
    const char *extra = extra_headers ? extra_headers : "";
    char hdr[1024];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d Error\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "%s"
        "\r\n", status, strlen(body), extra);
    if (n > 0) {
        ssize_t r;
        r = write(fd, hdr, (size_t)n); (void)r;
        r = write(fd, body, strlen(body)); (void)r;
    }
}

static void send_error(int fd, int status, const char *msg) {
    send_error_ex(fd, status, msg, NULL);
}

/* ── Token streaming callback ────────────────────────────────────────────── */

typedef struct {
    int   conn_fd;       /* accepted socket */
    char  id[32];        /* raw request ID — passed to sse_write_* (which
                           * itself prepends "chatcmpl-") */
    char  public_id[48]; /* "chatcmpl-"+id — what the client actually sees
                           * in the streamed JSON "id" field, and therefore
                           * what a client will send back to the cancel
                           * endpoint. Must match cancel_registry_begin(). */
    int   is_streaming;  /* SSE mode flag   */
    /* For non-streaming mode, accumulate full text */
    char *accum_buf;
    size_t accum_len;
    size_t accum_cap;
    long long tokens_emitted;
    /* Cancellation */
    CancelState *cancel;
} StreamState;

static int streaming_token_callback(const char *piece, void *userdata) {
    StreamState *st = (StreamState *)userdata;
    if (!piece) return 0;

    if (st->is_streaming) {
        sse_write_token(st->conn_fd, st->id, piece);
    } else {
        /* Accumulate into dynamic buffer */
        size_t plen = strlen(piece);
        if (st->accum_len + plen + 1 > st->accum_cap) {
            size_t new_cap = (st->accum_cap == 0) ? 4096 : st->accum_cap * 2;
            while (new_cap < st->accum_len + plen + 1) new_cap *= 2;
            char *new_buf = (char *)realloc(st->accum_buf, new_cap);
            if (!new_buf) return 1; /* OOM — stop generating */
            st->accum_buf = new_buf;
            st->accum_cap = new_cap;
        }
        memcpy(st->accum_buf + st->accum_len, piece, plen);
        st->accum_len += plen;
        st->accum_buf[st->accum_len] = '\0';
    }
    st->tokens_emitted++;

    return cancel_registry_check(st->cancel, st->public_id);
}

/* ── Route handlers ──────────────────────────────────────────────────────── */

static void handle_get_models(int fd, const char *extra_headers) {
    const char *body =
        "{\"object\":\"list\","
         "\"data\":[{"
            "\"id\":\"local-adaptive-engine\","
            "\"object\":\"model\","
            "\"owned_by\":\"project-zero\"}]}";
    send_response_ex(fd, 200, "application/json", body, 0, extra_headers);
}

static void handle_health(int fd, const char *extra_headers) {
    send_response_ex(fd, 200, "application/json", "{\"status\":\"ok\"}", 0, extra_headers);
}

static void handle_get_metrics(int fd, ApiContext *ctx, const char *extra_headers) {
    char buf[2048];
    metrics_render(&ctx->metrics, buf, sizeof(buf));
    send_response_ex(fd, 200, "text/plain; version=0.0.4", buf, 0, extra_headers);
}

static void handle_get_docs(int fd, const char *extra_headers) {
    send_response_ex(fd, 200, "text/html", openapi_docs_html(), 0, extra_headers);
}

static void handle_get_openapi(int fd, const char *extra_headers) {
    send_response_ex(fd, 200, "application/json", openapi_json(), 0, extra_headers);
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

static void handle_cancel(int fd, const char *body_start, size_t body_len,
                           ApiContext *ctx, const char *extra_headers) {
    char *body = (char *)malloc(body_len + 1);
    if (!body) { send_error_ex(fd, 500, "OOM", extra_headers); return; }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';

    char id[32];
    if (!extract_id_field(body, id, sizeof(id))) {
        free(body);
        send_error_ex(fd, 400, "Missing 'id' field", extra_headers);
        return;
    }
    free(body);

    int matched = cancel_registry_request(&ctx->cancel, id);
    if (matched) {
        metrics_record_cancel(&ctx->metrics);
        send_response_ex(fd, 200, "application/json",
                          "{\"cancelled\":true}", 0, extra_headers);
    } else {
        send_error_ex(fd, 404, "No matching in-flight request", extra_headers);
    }
}

static void handle_chat_completions(int fd, const char *body_start, size_t body_len,
                                    ApiContext *ctx, const char *extra_headers) {
    /* Only one generation runs at a time — try to acquire the mutex without
     * blocking so unrelated routes (metrics/docs/static/cancel) are never
     * stalled behind an in-flight generation. */
    if (pthread_mutex_trylock(&ctx->generation_mutex) != 0) {
        send_error_ex(fd, 429, "Another generation is already in flight", extra_headers);
        return;
    }

    /* Null-terminate the body for parsing */
    char *body = (char *)malloc(body_len + 1);
    if (!body) { pthread_mutex_unlock(&ctx->generation_mutex); send_error_ex(fd, 500, "OOM", extra_headers); return; }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';

    ChatRequest req;
    chat_request_init(&req);

    if (chat_request_parse(body, &req) != TN_OK) {
        free(body);
        pthread_mutex_unlock(&ctx->generation_mutex);
        send_error_ex(fd, 400, "Invalid JSON in request body", extra_headers);
        return;
    }
    free(body);

    if (req.num_messages == 0) {
        chat_request_free(&req);
        pthread_mutex_unlock(&ctx->generation_mutex);
        send_error_ex(fd, 400, "No messages in request", extra_headers);
        return;
    }

    /* Phase 22.2: image upload via the web UI's image_url content parts —
     * inject the vision pipeline into the KV cache before compiling/
     * generating the prompt, mirroring the CLI's --image handling.
     * Best-effort: an image sent to a server with no --vision/--proj
     * configured is logged and skipped (text-only fallback), matching the
     * CLI's own behavior when --image is given without --vision/--proj. */
    for (int i = req.num_messages - 1; i >= 0; i--) {
        if (strcmp(req.messages[i].role, "user") != 0 || !req.messages[i].image_data_url) continue;

        if (!ctx->server_config.vision_path || !ctx->server_config.proj_path) {
            fprintf(stderr, "[vision] image_url provided but server has no --vision/--proj configured; ignoring image\n");
            break;
        }

        unsigned char *img_bytes = NULL;
        size_t img_len = 0;
        if (data_url_decode(req.messages[i].image_data_url, &img_bytes, &img_len) != TN_OK) {
            fprintf(stderr, "[vision] failed to decode image_url data\n");
            break;
        }

        char tmp_path[] = "/tmp/pz-vision-upload-XXXXXX";
        int tmp_fd = mkstemp(tmp_path);
        if (tmp_fd < 0) {
            fprintf(stderr, "[vision] failed to create temp file for uploaded image\n");
            free(img_bytes);
            break;
        }
        ssize_t written = write(tmp_fd, img_bytes, img_len);
        close(tmp_fd);
        free(img_bytes);
        if (written < 0 || (size_t)written != img_len) {
            fprintf(stderr, "[vision] failed to write temp image file\n");
            unlink(tmp_path);
            break;
        }

        /* static: safe because only one generation runs at a time
         * (generation_mutex, held for the whole of this function). */
        static char stripped_prompt_buf[8192];
        const char *stripped_prompt = req.messages[i].content;
        /* cfg/weights are declared const in ApiContext (text generation
         * never mutates them); the vision pipeline's KV-cache injection
         * needs non-const pointers even though it doesn't rewrite either —
         * same underlying objects main.c's CLI path passes non-const. */
        vision_pipeline_run(tmp_path, ctx->server_config.vision_path, ctx->server_config.proj_path,
                            (Config *)ctx->cfg, (TransformerWeights *)ctx->weights, ctx->run_state,
                            ctx->moe_cfg, ctx->tok, ctx->tp,
                            req.messages[i].content, stripped_prompt_buf, sizeof(stripped_prompt_buf),
                            &stripped_prompt);
        unlink(tmp_path);

        if (stripped_prompt != req.messages[i].content) {
            char *new_content = strdup(stripped_prompt);
            if (new_content) {
                free(req.messages[i].content);
                req.messages[i].content = new_content;
            }
        }
        break;
    }

    /* Compile messages into a single prompt.
     * If the tokenizer has an embedded chat template (GGUF), generate()
     * handles template application internally.  For legacy tokenizers without
     * a built-in template, we apply a template here. */
    char *prompt = NULL;
    int free_prompt = 0;

    if (ctx->tok->chat_template) {
        /* GGUF tokenizer: just pass the last user message content directly;
         * generate_with_callback will apply the Jinja template. */
        const char *last_content = NULL;
        for (int i = req.num_messages - 1; i >= 0; i--) {
            if (strcmp(req.messages[i].role, "user") == 0 &&
                req.messages[i].content) {
                last_content = req.messages[i].content;
                break;
            }
        }
        if (!last_content) last_content = req.messages[req.num_messages-1].content;
        prompt = (char *)last_content; /* borrow — do NOT free */
    } else {
        /* Legacy tokenizer: compile all messages into a formatted prompt */
        char *compiled = (char *)malloc(HTTP_MAX_PROMPT_BYTES);
        if (!compiled) { chat_request_free(&req); pthread_mutex_unlock(&ctx->generation_mutex); send_error_ex(fd, 500, "OOM", extra_headers); return; }
        ChatTemplateType tmpl = chat_template_detect(req.model);
        if (chat_compile_prompt(&req, tmpl, compiled, HTTP_MAX_PROMPT_BYTES) != TN_OK) {
            free(compiled);
            chat_request_free(&req);
            pthread_mutex_unlock(&ctx->generation_mutex);
            send_error_ex(fd, 500, "Prompt compilation failed", extra_headers);
            return;
        }
        prompt     = compiled;
        free_prompt = 1;
    }

    /* Clamp sampling parameters to valid ranges */
    float temperature = req.temperature;
    float top_p       = req.top_p;
    int   max_tokens  = req.max_tokens;
    if (temperature < 0.0f) temperature = 0.0f;
    if (temperature > 2.0f) temperature = 2.0f;
    if (top_p < 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (max_tokens <= 0 || max_tokens > 4096) max_tokens = 512;

    /* Generate a request ID */
    char id[32];
    make_id(id, sizeof(id));

    /* Set up streaming state */
    StreamState st;
    memset(&st, 0, sizeof(st));
    st.conn_fd      = fd;
    st.is_streaming = req.stream;
    st.cancel       = &ctx->cancel;
    snprintf(st.id, sizeof(st.id), "%s", id);
    snprintf(st.public_id, sizeof(st.public_id), "chatcmpl-%s", id);

    if (req.stream) {
        /* Send SSE headers before starting generation */
        send_response_ex(fd, 200, "text/event-stream", NULL, 1, extra_headers);
    }

    /* Register under public_id — the same "chatcmpl-<id>" string the client
     * observes in the stream, and therefore what it will send back to
     * POST /v1/chat/completions/cancel. */
    cancel_registry_begin(&ctx->cancel, st.public_id);

    /* Run inference */
    generate_with_callback(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg,
                           ctx->tok, ctx->tp,
                           prompt, max_tokens, temperature, top_p,
                           streaming_token_callback, &st);

    cancel_registry_end(&ctx->cancel);
    metrics_add_tokens(&ctx->metrics, st.tokens_emitted);

    if (req.stream) {
        sse_write_done(fd, id);
    } else {
        /* 2026-07-15: build the JSON body FIRST so Content-Length reflects
         * its real size. The previous code sent "Content-Length: 0" (NULL
         * body) and only then wrote the actual JSON via a second write() —
         * conformant HTTP clients stop reading at 0 bytes and never see it. */
        const char *text = st.accum_buf ? st.accum_buf : "";
        char *json = sse_format_full_response(id, text);
        send_response_ex(fd, 200, "application/json", json, 0, extra_headers);
        free(json);
    }

    free(st.accum_buf);
    if (free_prompt) free(prompt);
    chat_request_free(&req);
    pthread_mutex_unlock(&ctx->generation_mutex);
}

/* ── HTTP request parser ─────────────────────────────────────────────────── */

typedef struct {
    char method[16];
    char path[256];
    char *body;          /* pointer into recv_buf or separately allocated */
    size_t body_len;
    int  content_length; /* from Content-Length header, -1 if not present */
    int  is_chunked;
    char origin[256];
    char authorization[512];
    int  has_origin;
    int  has_authorization;
} HttpRequest;

/* Copies a header value starting at *p up to (not including) \r or \n into
 * dst (NUL-terminated, truncated to dst_cap - 1). */
static void copy_header_value(const char *p, char *dst, size_t dst_cap) {
    size_t i = 0;
    while (*p == ' ') p++;
    while (*p && *p != '\r' && *p != '\n' && i < dst_cap - 1) dst[i++] = *p++;
    dst[i] = '\0';
}

/*
 * Parse a minimal HTTP/1.1 request from a NUL-terminated buffer.
 * Returns 1 on success, 0 on parse failure.
 */
static int http_parse(const char *buf, size_t buf_len, HttpRequest *req) {
    memset(req, 0, sizeof(*req));
    req->content_length = -1;

    /* Method */
    const char *p = buf;
    size_t i = 0;
    while (*p && *p != ' ' && i < sizeof(req->method) - 1) req->method[i++] = *p++;
    req->method[i] = '\0';
    if (*p != ' ') return 0;
    p++;

    /* Path */
    i = 0;
    while (*p && *p != ' ' && *p != '\r' && i < sizeof(req->path) - 1) req->path[i++] = *p++;
    req->path[i] = '\0';
    if (*p != ' ') return 0;

    /* Skip to end of request line */
    while (*p && *p != '\n') p++;
    if (*p) p++;

    /* Parse headers until blank line */
    while (*p) {
        if (*p == '\r' && *(p+1) == '\n') { p += 2; break; }
        if (*p == '\n') { p++; break; }

        if (strncasecmp(p, "content-length:", 15) == 0) {
            const char *v = p + 15;
            while (*v == ' ') v++;
            req->content_length = atoi(v);
        } else if (strncasecmp(p, "origin:", 7) == 0) {
            copy_header_value(p + 7, req->origin, sizeof(req->origin));
            req->has_origin = 1;
        } else if (strncasecmp(p, "authorization:", 14) == 0) {
            copy_header_value(p + 14, req->authorization, sizeof(req->authorization));
            req->has_authorization = 1;
        }
        /* Skip to next line */
        while (*p && *p != '\n') p++;
        if (*p) p++;
    }

    /* Body starts at p */
    req->body = (char *)p;
    req->body_len = (buf_len > (size_t)(p - buf))
                    ? buf_len - (size_t)(p - buf) : 0;

    return 1;
}

/* ── Per-connection handler ──────────────────────────────────────────────── */

static void handle_connection(int fd, ApiContext *ctx) {
    /* Receive the full request (including body) */
    char   *recv_buf = (char *)malloc(HTTP_RECV_BUF);
    if (!recv_buf) { close(fd); return; }
    size_t  total    = 0;
    size_t  cap      = HTTP_RECV_BUF;

    /* Read until we have headers + full body */
    int headers_done = 0;
    int content_length = -1;
    size_t header_end = 0;

    while (total < HTTP_MAX_BODY_BYTES + 4096) {
        ssize_t n = recv(fd, recv_buf + total, cap - total - 1, 0);
        if (n <= 0) break;
        total += (size_t)n;
        recv_buf[total] = '\0';

        if (!headers_done) {
            char *hdrs_end = strstr(recv_buf, "\r\n\r\n");
            if (!hdrs_end) hdrs_end = strstr(recv_buf, "\n\n");
            if (hdrs_end) {
                headers_done = 1;
                header_end = (size_t)(hdrs_end - recv_buf)
                             + (strstr(recv_buf, "\r\n\r\n") ? 4 : 2);

                /* Extract Content-Length from headers */
                char *cl = strcasestr(recv_buf, "content-length:");
                if (cl) content_length = atoi(cl + 15);

                /* 2026-07-15: pre-existing bug — a request with no
                 * Content-Length (the normal case for GET/HEAD/OPTIONS, and
                 * any bodyless POST) has content_length == -1 forever, so
                 * the old "stop once we have the full body" check below
                 * never fired and recv() blocked indefinitely waiting for
                 * bytes that were never coming (client already sent its
                 * full request and is waiting on OUR response). No
                 * Content-Length means no body is expected — stop reading
                 * as soon as headers are complete. */
                if (content_length < 0) break;

                /* If we already have the full body, stop reading */
                if (total >= header_end + (size_t)content_length) break;
            }
        } else {
            if (content_length >= 0 &&
                total >= header_end + (size_t)content_length) break;
        }

        /* Grow buffer if needed */
        if (cap - total < 1024) {
            size_t new_cap = cap * 2;
            if (new_cap > HTTP_MAX_BODY_BYTES + 8192) {
                send_error(fd, 413, "Request body too large");
                free(recv_buf);
                close(fd);
                return;
            }
            char *new_buf = (char *)realloc(recv_buf, new_cap);
            if (!new_buf) break;
            recv_buf = new_buf;
            cap = new_cap;
        }
    }
    recv_buf[total] = '\0';

    HttpRequest req;
    if (!http_parse(recv_buf, total, &req)) {
        send_error(fd, 400, "Malformed HTTP request");
        free(recv_buf);
        close(fd);
        return;
    }

    /* ── CORS + auth gate (applied ahead of dispatch; /health always open) ── */
    char extra_headers[512];
    extra_headers[0] = '\0';
    if (ctx->server_config.cors.enabled) {
        cors_format_headers(&ctx->server_config.cors,
                             req.has_origin ? req.origin : NULL,
                             extra_headers, sizeof(extra_headers));
    }

    int is_health = (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0);
    int is_preflight = (strcmp(req.method, "OPTIONS") == 0);

    if (is_preflight) {
        send_response_ex(fd, 204, "text/plain", NULL, 0, extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    if (!is_health &&
        !auth_check_request(&ctx->server_config.auth,
                             req.has_authorization ? req.authorization : NULL)) {
        metrics_record_request(&ctx->metrics, 401);
        send_error_ex(fd, 401, "Unauthorized", extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    /* ── Route dispatch ──────────────────────────────────────────────────── */
    int status = 200;
    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_get_models(fd, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        /* Always available, regardless of --web-ui — distinct from "/",
         * which now serves the web UI (Phase 22.2). */
        handle_health(fd, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/metrics") == 0) {
        if (ctx->server_config.metrics.enabled) {
            handle_get_metrics(fd, ctx, extra_headers);
        } else {
            status = 404;
            send_error_ex(fd, 404, "Metrics disabled (start with --metrics)", extra_headers);
        }
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/docs") == 0) {
        handle_get_docs(fd, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/openapi.json") == 0) {
        handle_get_openapi(fd, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions") == 0) {
        handle_chat_completions(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions/cancel") == 0) {
        handle_cancel(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 &&
               ctx->server_config.web_ui != WEBUI_MODE_OFF) {
        /* Web UI (Phase 22.2): "/" and files under "/assets/", placed after
         * every API route so nothing above is ever shadowed. Any other
         * unmatched GET also falls through here for client-side SPA routing. */
        status = static_assets_serve(fd, req.path, &ctx->server_config, extra_headers);
    } else {
        status = 404;
        send_error_ex(fd, 404, "Not found", extra_headers);
    }
    metrics_record_request(&ctx->metrics, status);

    free(recv_buf);
    close(fd);
}

/* ── Per-connection thread trampoline (Phase 22 concurrency rearchitecture) ── */

typedef struct {
    int         conn_fd;
    ApiContext *ctx;
} ConnArgs;

static void *connection_thread_fn(void *arg) {
    ConnArgs *ca = (ConnArgs *)arg;
    metrics_connection_opened(&ca->ctx->metrics);
    handle_connection(ca->conn_fd, ca->ctx);
    metrics_connection_closed(&ca->ctx->metrics);
    free(ca);
    return NULL;
}

/* ── Listener thread ─────────────────────────────────────────────────────── */

static void *listener_thread_fn(void *arg) {
    ApiContext *ctx = (ApiContext *)arg;

    while (ctx->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int conn_fd = accept(ctx->server_fd,
                             (struct sockaddr *)&client_addr, &addr_len);
        if (conn_fd < 0) {
            if (!ctx->running) break;
            continue;
        }

        if (g_tn_verbose)
            fprintf(stderr, "[API] Connection from %s\n",
                    inet_ntoa(client_addr.sin_addr));

        ConnArgs *ca = (ConnArgs *)malloc(sizeof(ConnArgs));
        if (!ca) { close(conn_fd); continue; }
        ca->conn_fd = conn_fd;
        ca->ctx     = ctx;

        pthread_t tid;
        if (pthread_create(&tid, NULL, connection_thread_fn, ca) != 0) {
            close(conn_fd);
            free(ca);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void api_context_init(ApiContext *ctx,
                      const Config *cfg,
                      const TransformerWeights *weights,
                      RunState *run_state,
                      const MoEConfig *moe_cfg,
                      Tokenizer *tok,
                      ThreadPool *tp) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg        = cfg;
    ctx->weights    = weights;
    ctx->run_state  = run_state;
    ctx->moe_cfg    = moe_cfg;
    ctx->tok        = tok;
    ctx->tp         = tp;
    ctx->server_fd  = -1;
    ctx->running    = 0;
    server_config_init(&ctx->server_config);
    metrics_init(&ctx->metrics);
    cancel_state_init(&ctx->cancel);
    pthread_mutex_init(&ctx->generation_mutex, NULL);
}

TernaryError api_server_start(int port, ApiContext *ctx) {
    if (!ctx) return TN_ERR_INVALID_ARGS;

    /* 2026-07-15: without this, any client that disconnects mid-response
     * (e.g. a short curl --max-time, or a browser navigating away during
     * streaming) delivers SIGPIPE to write()/send() on that socket, whose
     * default disposition kills the WHOLE process — not just the one
     * connection. Standard for any long-running socket server. */
    signal(SIGPIPE, SIG_IGN);

    /* Create TCP socket */
    ctx->server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (ctx->server_fd < 0) return TN_ERR_SOCKET_CREATE;

    /* Allow immediate reuse of the port after restart */
    int opt = 1;
    setsockopt(ctx->server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    /* Bind to loopback only — not exposed to the network */
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); /* 127.0.0.1 */

    if (bind(ctx->server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(ctx->server_fd);
        ctx->server_fd = -1;
        return TN_ERR_SOCKET_BIND;
    }

    if (listen(ctx->server_fd, HTTP_BACKLOG) < 0) {
        close(ctx->server_fd);
        ctx->server_fd = -1;
        return TN_ERR_SOCKET_LISTEN;
    }

    ctx->running = 1;

    /* Spawn listener thread */
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t));
    if (!tid) {
        close(ctx->server_fd);
        ctx->server_fd = -1;
        ctx->running = 0;
        return TN_ERR_THREAD_CREATE;
    }
    if (pthread_create(tid, NULL, listener_thread_fn, ctx) != 0) {
        free(tid);
        close(ctx->server_fd);
        ctx->server_fd = -1;
        ctx->running = 0;
        return TN_ERR_THREAD_CREATE;
    }
    ctx->listener_thread = tid;

    printf("[API] Server listening on http://127.0.0.1:%d\n", port);
    printf("[API] Endpoints:\n");
    printf("[API]   GET  http://127.0.0.1:%d/v1/models\n", port);
    printf("[API]   POST http://127.0.0.1:%d/v1/chat/completions\n", port);
    printf("[API]   POST http://127.0.0.1:%d/v1/chat/completions/cancel\n", port);
    printf("[API]   GET  http://127.0.0.1:%d/health\n", port);
    printf("[API]   GET  http://127.0.0.1:%d/docs\n", port);
    if (ctx->server_config.web_ui != WEBUI_MODE_OFF)
        printf("[API]   GET  http://127.0.0.1:%d/  (web chat UI%s)\n", port,
               ctx->server_config.static_dir ? ", served from disk" : "");
    if (ctx->server_config.metrics.enabled)
        printf("[API]   GET  http://127.0.0.1:%d/metrics\n", port);
    if (ctx->server_config.cors.enabled)
        printf("[API]   CORS enabled (%d allowed origin%s)\n", ctx->server_config.cors.num_origins,
               ctx->server_config.cors.num_origins == 1 ? "" : "s");
    if (ctx->server_config.auth.api_key)
        printf("[API]   Auth enabled (Authorization: Bearer <key> required)\n");
    return TN_OK;
}

void api_server_stop(ApiContext *ctx) {
    if (!ctx || ctx->server_fd < 0) return;
    ctx->running = 0;
    /* Wake the blocking accept() in listener_thread_fn. close() alone is
     * *not* reliable here: POSIX explicitly leaves "another thread blocked
     * in accept() on an fd this thread closes" unspecified, and in practice
     * on Linux a bare close() from a different thread frequently does not
     * unblock a concurrent accept() at all — confirmed by this exact hang
     * the first time api_server_stop() became reachable (previously dead
     * code, since main()'s pause() had no signal handler installed and so
     * never returned). shutdown(SHUT_RDWR) first reliably makes the
     * blocked accept() return (ECONNABORTED/EINVAL, both handled by the
     * `if (!ctx->running) break;` check below) before the fd is closed. */
    shutdown(ctx->server_fd, SHUT_RDWR);
    close(ctx->server_fd);
    ctx->server_fd = -1;
    if (ctx->listener_thread) {
        pthread_join(*(pthread_t *)ctx->listener_thread, NULL);
        free(ctx->listener_thread);
        ctx->listener_thread = NULL;
    }
    /* Deliberately NOT freeing server_config/cancel/generation_mutex here:
     * detached per-connection threads spawned before the stop request may
     * still be running and reference them. There is no drain/barrier to
     * wait for them (this is a local single-operator dev server, not a
     * service that needs clean reload), so these small, process-lifetime
     * resources are left for the OS to reclaim at exit rather than risking
     * a use-after-free race. */
}
