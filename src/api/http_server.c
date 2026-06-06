/*
 * Phase 21 — OpenAI-Compatible API Layer
 * src/api/http_server.c
 *
 * Minimal POSIX HTTP/1.1 server using only standard system calls (socket,
 * bind, listen, accept, read, write).  No external dependencies beyond
 * what the rest of the engine already uses.
 *
 * Supported routes
 * ────────────────
 *   GET  /v1/models                 → JSON model list
 *   POST /v1/chat/completions       → Chat completion (streaming + non-streaming)
 *   GET  /health                    → {"status":"ok"}
 *
 * Threading model
 * ───────────────
 * A single listener thread (spawned by api_server_start) calls accept() in
 * a loop.  Each accepted connection is handled inline on the same thread
 * (sequential, not parallel).  This keeps the design simple and avoids
 * thread-safety issues with the inference engine RunState.
 *
 * Security notes
 * ──────────────
 *   • Listens only on 127.0.0.1 (loopback) by default — not exposed to LAN.
 *   • Request body size is capped at HTTP_MAX_BODY_BYTES.
 *   • Buffer overflows are guarded with explicit size checks.
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

/* ── Tuning constants ────────────────────────────────────────────────────── */
#define HTTP_RECV_BUF        8192      /* receive buffer per connection */
#define HTTP_MAX_BODY_BYTES  (512*1024) /* 512 KiB max request body      */
#define HTTP_MAX_PROMPT_BYTES (65536)  /* compiled prompt size cap       */
#define HTTP_BACKLOG         8

/* ── Simple unique ID generator ─────────────────────────────────────────── */
static void make_id(char *buf, size_t cap) {
    snprintf(buf, cap, "%lx", (unsigned long)time(NULL));
}

/* ── HTTP response helpers ───────────────────────────────────────────────── */

static void send_response(int fd, int status, const char *content_type,
                           const char *body, int streaming) {
    char hdr[512];
    size_t body_len = body ? strlen(body) : 0;
    int n;
    if (streaming) {
        n = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Connection: close\r\n"
            "\r\n", status, content_type);
    } else {
        n = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "\r\n", status, content_type, body_len);
    }
    if (n > 0) { ssize_t r = write(fd, hdr, (size_t)n); (void)r; }
    if (body && !streaming) { ssize_t r = write(fd, body, body_len); (void)r; }
}

static void send_error(int fd, int status, const char *msg) {
    char body[256];
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"server_error\"}}", msg);
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d Error\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n", status, strlen(body));
    if (n > 0) {
        ssize_t r;
        r = write(fd, hdr, (size_t)n); (void)r;
        r = write(fd, body, strlen(body)); (void)r;
    }
}

/* ── Token streaming callback ────────────────────────────────────────────── */

typedef struct {
    int   conn_fd;       /* accepted socket */
    char  id[32];        /* request ID      */
    int   is_streaming;  /* SSE mode flag   */
    /* For non-streaming mode, accumulate full text */
    char *accum_buf;
    size_t accum_len;
    size_t accum_cap;
} StreamState;

static void streaming_token_callback(const char *piece, void *userdata) {
    StreamState *st = (StreamState *)userdata;
    if (!piece) return;

    if (st->is_streaming) {
        sse_write_token(st->conn_fd, st->id, piece);
    } else {
        /* Accumulate into dynamic buffer */
        size_t plen = strlen(piece);
        if (st->accum_len + plen + 1 > st->accum_cap) {
            size_t new_cap = (st->accum_cap == 0) ? 4096 : st->accum_cap * 2;
            while (new_cap < st->accum_len + plen + 1) new_cap *= 2;
            char *new_buf = (char *)realloc(st->accum_buf, new_cap);
            if (!new_buf) return;
            st->accum_buf = new_buf;
            st->accum_cap = new_cap;
        }
        memcpy(st->accum_buf + st->accum_len, piece, plen);
        st->accum_len += plen;
        st->accum_buf[st->accum_len] = '\0';
    }
}

/* ── Route handlers ──────────────────────────────────────────────────────── */

static void handle_get_models(int fd) {
    const char *body =
        "{\"object\":\"list\","
         "\"data\":[{"
            "\"id\":\"local-adaptive-engine\","
            "\"object\":\"model\","
            "\"owned_by\":\"project-zero\"}]}";
    send_response(fd, 200, "application/json", body, 0);
}

static void handle_health(int fd) {
    send_response(fd, 200, "application/json", "{\"status\":\"ok\"}", 0);
}

static void handle_chat_completions(int fd, const char *body_start, size_t body_len,
                                    ApiContext *ctx) {
    /* Null-terminate the body for parsing */
    char *body = (char *)malloc(body_len + 1);
    if (!body) { send_error(fd, 500, "OOM"); return; }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';

    ChatRequest req;
    chat_request_init(&req);

    if (chat_request_parse(body, &req) != TN_OK) {
        free(body);
        send_error(fd, 400, "Invalid JSON in request body");
        return;
    }
    free(body);

    if (req.num_messages == 0) {
        chat_request_free(&req);
        send_error(fd, 400, "No messages in request");
        return;
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
        if (!compiled) { chat_request_free(&req); send_error(fd, 500, "OOM"); return; }
        ChatTemplateType tmpl = chat_template_detect(req.model);
        if (chat_compile_prompt(&req, tmpl, compiled, HTTP_MAX_PROMPT_BYTES) != TN_OK) {
            free(compiled);
            chat_request_free(&req);
            send_error(fd, 500, "Prompt compilation failed");
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
    snprintf(st.id, sizeof(st.id), "%s", id);

    if (req.stream) {
        /* Send SSE headers before starting generation */
        send_response(fd, 200, "text/event-stream", NULL, 1);
    }

    /* Run inference */
    generate_with_callback(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg,
                           ctx->tok, ctx->tp,
                           prompt, max_tokens, temperature, top_p,
                           streaming_token_callback, &st);

    if (req.stream) {
        sse_write_done(fd, id);
    } else {
        /* Send buffered response */
        const char *text = st.accum_buf ? st.accum_buf : "";
        send_response(fd, 200, "application/json", NULL, 0);
        sse_write_full_response(fd, id, text);
    }

    free(st.accum_buf);
    if (free_prompt) free(prompt);
    chat_request_free(&req);
}

/* ── HTTP request parser ─────────────────────────────────────────────────── */

typedef struct {
    char method[16];
    char path[256];
    char *body;          /* pointer into recv_buf or separately allocated */
    size_t body_len;
    int  content_length; /* from Content-Length header, -1 if not present */
    int  is_chunked;
} HttpRequest;

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

        /* Look for Content-Length header */
        if (strncasecmp(p, "content-length:", 15) == 0) {
            p += 15;
            while (*p == ' ') p++;
            req->content_length = atoi(p);
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

                /* If we already have the full body, stop reading */
                if (content_length >= 0 &&
                    total >= header_end + (size_t)content_length) break;
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

    /* Route dispatch */
    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_get_models(fd);
    } else if (strcmp(req.method, "GET") == 0 &&
               (strcmp(req.path, "/health") == 0 || strcmp(req.path, "/") == 0)) {
        handle_health(fd);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions") == 0) {
        handle_chat_completions(fd, req.body, req.body_len, ctx);
    } else {
        send_error(fd, 404, "Not found");
    }

    free(recv_buf);
    close(fd);
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

        handle_connection(conn_fd, ctx);
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
}

TernaryError api_server_start(int port, ApiContext *ctx) {
    if (!ctx) return TN_ERR_INVALID_ARGS;

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
    printf("[API]   GET  http://127.0.0.1:%d/health\n", port);
    return TN_OK;
}

void api_server_stop(ApiContext *ctx) {
    if (!ctx || ctx->server_fd < 0) return;
    ctx->running = 0;
    /* Wake the accept() call by closing the socket */
    close(ctx->server_fd);
    ctx->server_fd = -1;
    if (ctx->listener_thread) {
        pthread_join(*(pthread_t *)ctx->listener_thread, NULL);
        free(ctx->listener_thread);
        ctx->listener_thread = NULL;
    }
}
