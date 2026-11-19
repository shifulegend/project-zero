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
 *   GET  /v1/models                       → JSON model list (+ model metadata)
 *   GET  /v1/config                       → Server generation defaults, valid
 *                                            ranges and feature flags (drives
 *                                            every web UI control)
 *   POST /v1/config/defaults              → Phase 40.3 save one default to the
 *                                            `pz config` file (next start)
 *   POST /v1/watermark/detect             → Phase 33 watermark z-score for text
 *   POST /v1/selfcheck                    → accuracy probe of the loaded model and settings
 *   GET  /v1/prompt_cache                 → Phase 26 prompt-cache stats
 *   POST /v1/prompt_cache/clear           → drop all cached prompt snapshots
 *   GET  /v1/update                       → Phase 38 update-check state (notify only)
 *   POST /v1/update/check                 → re-check now (rate-limited)
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
 * generate take the slot through ctx->gen_queue (Phase 23 stage 1): 429
 * immediately when busy by default, or wait in FIFO order with
 * --queue-size N; generation_mutex is then locked uncontended.
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

#include "api/config_endpoint.h"
#include "api/http_response.h"
#include "api/http_routes.h"
#include "api/http_request.h"
#include "api/chat_image.h"
#include "api/api_server.h"
#include "api/net_io.h"
#include "api/rate_limit.h"
#include "api/http_headers.h"
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
#include "api/api_info.h"
#include "api/update_endpoint.h"
#include "api/request_params.h"
#include "api/watermark_endpoint.h"
#include "api/selfcheck_endpoint.h"
#include "reasoning/best_of.h"
#include "reasoning/mcts.h"
#include "api/json_write.h"
#include "api/gen_queue.h"
#include "api/batch_scheduler.h"
#include "api/http_status.h"
#include "kv_cache/kv_strategy.h"
#include "multimodal/vision_pipeline.h"
#include "transformer/generate.h"
#include "core/debug.h"

#include <limits.h>
#include <sys/socket.h>
#include <sys/time.h>
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
#define HTTP_MAX_PROMPT_BYTES (65536) /* compiled prompt size cap       */
#define HTTP_BACKLOG 16

/* Fixed buffer capacities (bytes). Each is the size of one stack/static array below. */
#define HTTP_SSE_EVENT_BUF 160  /* one queue-failure SSE event */
#define HTTP_EXTRA_HDRS_BUF 640 /* CORS headers + X-Request-ID added to a response */
#define HTTP_RETRY_HDRS_BUF (HTTP_EXTRA_HDRS_BUF + 128) /* extra headers + Retry-After */
#define HTTP_BUSY_REPLY_BUF 320 /* complete 503 reply to a refused connection */

/* ── Token streaming callback ────────────────────────────────────────────── */

typedef struct {
    int conn_fd;        /* accepted socket */
    char id[32];        /* raw request ID — passed to sse_write_* (which
                           * itself prepends "chatcmpl-") */
    char public_id[48]; /* "chatcmpl-"+id — what the client actually sees
                           * in the streamed JSON "id" field, and therefore
                           * what a client will send back to the cancel
                           * endpoint. Must match cancel_registry_begin(). */
    int is_streaming;   /* SSE mode flag   */
    /* For non-streaming mode, accumulate full text */
    char *accum_buf;
    size_t accum_len;
    size_t accum_cap;
    long long tokens_emitted;
    /* Cancellation */
    CancelState *cancel;
} StreamState;

/* Phase 30: adds "best_of": {"n", "chosen", "scores": [...], "completion_tokens_total"}
 * to a finished response object (consumes and returns the heap string). */
static char *append_best_of_json(char *json, const BestOfResult *r) {
    size_t len = strlen(json);
    if (len < 2 || json[len - 1] != '}') return json;
    JsonBuf jb;
    jb_init(&jb);
    json[len - 1] = '\0';
    jb_raw(&jb, json);
    jb_raw(&jb, ",\"best_of\":{\"n\":");
    jb_int(&jb, r->n);
    jb_raw(&jb, ",\"chosen\":");
    jb_int(&jb, r->chosen);
    jb_raw(&jb, ",\"completion_tokens_total\":");
    jb_int(&jb, r->tokens_total);
    jb_raw(&jb, ",\"scores\":[");
    for (int i = 0; i < r->n && i < BEST_OF_N_MAX; i++) {
        if (i) jb_raw(&jb, ",");
        jb_num(&jb, r->scores[i]);
    }
    jb_raw(&jb, "]}}");
    char *out = jb_finish(&jb);
    if (!out) {
        json[len - 1] = '}';
        return json;
    }
    free(json);
    return out;
}

/* Phase 30: adds "mcts": {"simulations", "nodes", "steps", "path_value",
 * "scorer", "search_tokens"} to a finished response object. */
static char *append_mcts_json(char *json, const MctsResult *r) {
    size_t len = strlen(json);
    if (len < 2 || json[len - 1] != '}') return json;
    JsonBuf jb;
    jb_init(&jb);
    json[len - 1] = '\0';
    jb_raw(&jb, json);
    jb_raw(&jb, ",\"mcts\":{\"simulations\":");
    jb_int(&jb, r->simulations);
    jb_raw(&jb, ",\"nodes\":");
    jb_int(&jb, r->nodes);
    jb_raw(&jb, ",\"steps\":");
    jb_int(&jb, r->steps);
    jb_raw(&jb, ",\"path_value\":");
    jb_num(&jb, r->path_value);
    jb_raw(&jb, ",\"scorer\":");
    jb_str(&jb, r->used_prm ? "prm" : "self");
    jb_raw(&jb, ",\"search_tokens\":");
    jb_int(&jb, r->tokens_total);
    jb_raw(&jb, "}}");
    char *out = jb_finish(&jb);
    if (!out) {
        json[len - 1] = '}';
        return json;
    }
    free(json);
    return out;
}

/* Phase 30: lets best-of-N abandon its candidates when the client cancels. */
static int best_of_abort_check(void *userdata) {
    StreamState *st = (StreamState *)userdata;
    return cancel_registry_check(st->cancel, st->public_id);
}

static int streaming_token_callback(const char *piece, void *userdata) {
    StreamState *st = (StreamState *)userdata;
    if (!piece) return 0;

    if (st->is_streaming) {
        /* 2026-10-22: a failed write means the client left — stop instead of
         * generating the rest of the reply for nobody (and holding a slot). */
        if (sse_write_token(st->conn_fd, st->id, piece) < 0) return 1;
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

/* Phase 23 stage 1: while a streaming request waits for the generation slot,
 * send the SSE headers once and then ": queued <position>" comments (also a
 * keep-alive). A failed write means the client left — abandon the wait. */
typedef struct {
    int fd;
    int stream;
    int headers_sent;
    const char *extra_headers;
} QueueWaitCtx;

static int queue_wait_notify(int position, void *userdata) {
    QueueWaitCtx *w = (QueueWaitCtx *)userdata;
    if (!w->stream) return 0;
    if (!w->headers_sent) {
        http_send_response_ex(w->fd, HTTP_OK, "text/event-stream", NULL, 1, w->extra_headers);
        w->headers_sent = 1;
    }
    char line[64];
    int n = snprintf(line, sizeof(line), ": queued %d\n\n", position);
    return write(w->fd, line, (size_t)n) != n;
}

static void handle_chat_completions(int fd, const char *body_start, size_t body_len,
                                    ApiContext *ctx, const char *extra_headers) {
    /* Parse first — no shared state is touched — so the request's own
     * settings (stream) are known while it waits for the slot. */
    char *body = (char *)malloc(body_len + 1);
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return;
    }
    memcpy(body, body_start, body_len);
    body[body_len] = '\0';

    ChatRequest req;
    chat_request_init(&req);

    TernaryError parse_err = chat_request_parse(body, &req);
    if (parse_err != TN_OK) {
        free(body);
        chat_request_free(&req);
        http_send_error_ex(fd, HTTP_BAD_REQUEST, chat_request_error_message(parse_err),
                           extra_headers);
        return;
    }
    free(body);

    if (req.num_messages == 0) {
        chat_request_free(&req);
        http_send_error_ex(fd, HTTP_BAD_REQUEST, "No messages in request", extra_headers);
        return;
    }

    /* Phase 30: best-of-N (server default --best-of, capped by --best-of-max)
     * and step-level tree search (server --mcts*, request "mcts"). */
    int best_of = (req.set_mask & CHAT_SET_BEST_OF) ? req.best_of : ctx->server_config.best_of;
    if (best_of < 1) best_of = 1;
    if (best_of > ctx->server_config.best_of_max) best_of = ctx->server_config.best_of_max;
    MctsOptions mopt;
    request_mcts_resolve(&req, &ctx->server_config.mcts, ctx->server_config.mcts_max, &mopt);
    int wants_image = 0;
    if (ctx->server_config.vision_path && ctx->server_config.proj_path)
        for (int i = 0; i < req.num_messages; i++)
            if (strcmp(req.messages[i].role, "user") == 0 && req.messages[i].image_data_url)
                wants_image = 1;
    /* Phase 23 stage 2b: plain chat requests share the engine through the
     * parallel scheduler (one unit each); the rest need all of it. */
    int batched = ctx->sched && best_of <= 1 && mopt.simulations <= 0 && !wants_image;
    int units   = batched ? 1 : http_engine_units(ctx);

    /* With --parallel N, N batched requests run at once; otherwise one
     * generation at a time. With --queue-size N, up to N requests wait their
     * turn in arrival order (Phase 23 stage 1); otherwise (the default) a
     * busy server answers 429 immediately. Unrelated routes
     * (metrics/docs/static/cancel) never touch the queue. */
    QueueWaitCtx qw   = {fd, req.stream ? 1 : 0, 0, extra_headers};
    GenQueueResult qr = gen_queue_acquire_n(
        &ctx->gen_queue, units, ctx->server_config.queue_timeout_s * 1000, queue_wait_notify, &qw);
    if (qr == GEN_QUEUE_OK && !atomic_load(&ctx->running)) { /* shutting down: don't start */
        gen_queue_release_n(&ctx->gen_queue, units);
        chat_request_free(&req);
        if (!qw.headers_sent)
            http_send_error_ex(fd, HTTP_SERVICE_UNAVAILABLE, "Server is shutting down",
                               extra_headers);
        return;
    }
    if (qr != GEN_QUEUE_OK) {
        chat_request_free(&req);
        if (qr == GEN_QUEUE_ABORTED) return; /* client already gone */
        const char *why = qr == GEN_QUEUE_TIMEOUT ? "Timed out waiting for the generation queue"
                          : ctx->server_config.queue_size > 0
                              ? "Generation queue is full"
                              : "Another generation is already in flight";
        if (qw.headers_sent) {
            char ev[HTTP_SSE_EVENT_BUF];
            int n = snprintf(ev, sizeof(ev),
                             "data: {\"error\":{\"message\":\"%s\",\"type\":\"server_error\"}}\n\n",
                             why);
            if (n > 0) (void)net_write_all(fd, ev, (size_t)n); /* peer gone: stream already ended */
        } else {
            http_send_error_ex(fd, HTTP_TOO_MANY_REQUESTS, why, extra_headers);
        }
        return;
    }
    int had_image = 0;
    if (batched && !(ctx->server_config.vision_path && ctx->server_config.proj_path))
        for (int i = 0; i < req.num_messages; i++)
            if (strcmp(req.messages[i].role, "user") == 0 && req.messages[i].image_data_url) {
                fprintf(stderr, "[vision] image_url provided but server has no --vision/--proj "
                                "configured; ignoring image\n");
                break;
            }
    if (!batched) {
        pthread_mutex_lock(&ctx->generation_mutex); /* uncontended: the queue granted the engine */
        /* Every request is a fresh conversation rendered from its full message history, so start
         * from KV position 0 (2026-10-18: a stale image prefix leaked into later text requests). */
        ctx->run_state->current_pos = 0;

        /* Phase 22.2: image_url parts go through the vision pipeline (api/chat_image.h). */
        had_image = chat_apply_image(ctx, &req);
    } /* !batched */

    /* Compile messages into a single prompt.
     * If the tokenizer has an embedded chat template (GGUF), generate()
     * handles template application internally.  For legacy tokenizers without
     * a built-in template, we apply a template here. */
    char *prompt    = NULL;
    int free_prompt = 0;

    /* Resolve generation settings: server defaults (CLI flags) overlaid
     * with whatever the client sent, clamped to valid ranges. */
    GenerationParams gp;
    request_params_resolve(&req, &ctx->server_config.gen_defaults, api_max_tokens_limit(ctx), &gp);

    const char *roles[CHAT_MAX_MESSAGES];
    const char *contents[CHAT_MAX_MESSAGES];

    /* Per-request templates only when the operator allows them. */
    if (!ctx->server_config.allow_request_template) gp.chat_template = NULL;

    if (ctx->tok->chat_template || gp.chat_template) {
        /* GGUF tokenizer: the model's own Jinja template renders the whole
         * conversation (system/user/assistant turns). 2026-10-18: this used
         * to forward only the last user message, silently dropping all
         * earlier turns and any system message — multi-turn chat in the web
         * UI had no memory. */
        int n = 0;
        for (int i = 0; i < req.num_messages && n < CHAT_MAX_MESSAGES; i++) {
            if (!req.messages[i].content) continue;
            roles[n]      = req.messages[i].role[0] ? req.messages[i].role : "user";
            contents[n++] = req.messages[i].content;
        }
        gp.roles      = roles;
        gp.contents   = contents;
        gp.n_messages = n;
        prompt        = (char *)(n > 0 ? contents[n - 1] : "");
    } else {
        /* Legacy tokenizer: compile all messages into a formatted prompt */
        char *compiled = (char *)malloc(HTTP_MAX_PROMPT_BYTES);
        if (!compiled) {
            chat_request_free(&req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
            return;
        }
        ChatTemplateType tmpl = ctx->server_config.chat_format >= 0
                                    ? (ChatTemplateType)ctx->server_config.chat_format
                                    : chat_template_detect_vocab(ctx->tok);
        /* Message text is content, never chat markup (prompt-injection
         * defence, 2026-10-22): neutralise control tokens before the
         * built-in formats wrap it in their own. */
        int neutral_ok = 1;
        for (int i = 0; i < req.num_messages && neutral_ok; i++) {
            if (!req.messages[i].content) continue;
            char *safe = tokenizer_neutralize_control(ctx->tok, req.messages[i].content);
            if (!safe) {
                neutral_ok = 0;
                break;
            } /* fail closed */
            free(req.messages[i].content);
            req.messages[i].content = safe;
        }
        if (!neutral_ok) {
            free(compiled);
            chat_request_free(&req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
            return;
        }
        if (chat_compile_prompt(&req, tmpl, compiled, HTTP_MAX_PROMPT_BYTES) != TN_OK) {
            free(compiled);
            chat_request_free(&req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "Prompt compilation failed", extra_headers);
            return;
        }
        prompt      = compiled;
        free_prompt = 1;
    }

    /* Generate a request ID */
    char id[32];
    http_new_id(id, sizeof(id));

    /* Set up streaming state */
    StreamState st;
    memset(&st, 0, sizeof(st));
    st.conn_fd      = fd;
    st.is_streaming = req.stream;
    st.cancel       = &ctx->cancel;
    snprintf(st.id, sizeof(st.id), "%s", id);
    snprintf(st.public_id, sizeof(st.public_id), "chatcmpl-%s", id);

    if (req.stream && !qw.headers_sent) {
        /* Send SSE headers before starting generation */
        http_send_response_ex(fd, HTTP_OK, "text/event-stream", NULL, 1, extra_headers);
    }

    /* Register under public_id — the same "chatcmpl-<id>" string the client
     * observes in the stream, and therefore what it will send back to
     * POST /v1/chat/completions/cancel. */
    cancel_registry_begin(&ctx->cancel, st.public_id);

    /* Run inference */
    /* Phase 21.5: reuse the KV prefix shared with the previous request unless
     * disabled server-wide (--prompt-cache off) or for this request. */
    int use_cache =
        (req.set_mask & CHAT_SET_CACHE_PROMPT) ? req.cache_prompt : ctx->server_config.prompt_cache;
    if (!batched) {
        if (had_image) prefix_cache_reset(&ctx->prefix_cache); /* vision prefill rewrote KV */
        if (use_cache && ctx->prefix_cache.tokens) {
            gp.prefix_cache = &ctx->prefix_cache;
            gp.prompt_store = &ctx->prompt_store;
        } else
            prefix_cache_reset(&ctx->prefix_cache); /* KV is about to be overwritten */
    }

    GenerationStats gstats;
    memset(&gstats, 0, sizeof(gstats));
    gp.stats = &gstats;
    BestOfResult bres;
    memset(&bres, 0, sizeof(bres));
    MctsResult mres;
    memset(&mres, 0, sizeof(mres));
    if (batched) {
        batch_scheduler_generate(ctx->sched, prompt, &gp, use_cache, streaming_token_callback, &st);
    } else if (mopt.simulations > 0) {
        const char *question = "";
        for (int i = req.num_messages - 1; i >= 0; i--)
            if (req.messages[i].content && strcmp(req.messages[i].role, "user") == 0) {
                question = req.messages[i].content;
                break;
            }
        if (generate_mcts(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp,
                          prompt, question, &gp, &mopt, ctx->prm, streaming_token_callback, &st,
                          best_of_abort_check, &mres) != TN_OK &&
            gstats.completion_tokens == 0)
            gstats.finish_reason = GEN_FINISH_ERROR;
    } else if (best_of > 1) {
        generate_best_of(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp,
                         prompt, &gp, best_of, BEST_OF_SCORE_MEAN_LOGPROB, streaming_token_callback,
                         &st, best_of_abort_check, &bres);
    } else {
        generate_ex(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp, prompt,
                    &gp, streaming_token_callback, &st);
    }
    const char *finish_reason = gen_finish_reason_str(gstats.finish_reason);

    cancel_registry_end(&ctx->cancel, st.public_id);
    metrics_add_tokens(&ctx->metrics, st.tokens_emitted);

    if (gstats.finish_reason == GEN_FINISH_ERROR && !req.stream) {
        /* Nothing was generated because the prompt could not be prepared
         * (template render or tokenization failed) — report it instead of an
         * empty 200 completion. */
        free(st.accum_buf);
        if (free_prompt) free(prompt);
        chat_request_free(&req);
        http_release_generation_slot(ctx, units, !batched);
        http_send_error_ex(fd, HTTP_BAD_REQUEST, "Prompt could not be tokenized for this model",
                           extra_headers);
        return;
    }

    /* Release the engine BEFORE the last write of the response (2026-11-11).
     * Once a client has read the final event / body it may send its next
     * request at once; while the slot was still held at that point, a
     * strictly sequential client got a spurious 429 (no free unit and no
     * --queue-size), and a slow reader held the engine hostage. Nothing
     * below touches engine state: generation has returned (the scheduler
     * marks a job done only after it left its slot), the response is built
     * from request-local buffers (st.accum_buf, gstats, bres, mres), and
     * `finish_reason` is a static string. */
    if (req.stream) {
        http_release_generation_slot(ctx, units, !batched);
        sse_write_done_ex(fd, id, finish_reason, gstats.prompt_tokens, gstats.completion_tokens,
                          gstats.cached_tokens);
    } else {
        /* 2026-07-15: build the JSON body FIRST so Content-Length reflects
         * its real size. The previous code sent "Content-Length: 0" (NULL
         * body) and only then wrote the actual JSON via a second write() —
         * conformant HTTP clients stop reading at 0 bytes and never see it. */
        const char *text = st.accum_buf ? st.accum_buf : "";
        char *json = sse_format_full_response_ex(id, text, finish_reason, gstats.prompt_tokens,
                                                 gstats.completion_tokens, gstats.cached_tokens);
        if (json && mopt.simulations > 0)
            json = append_mcts_json(json, &mres);
        else if (json && best_of > 1)
            json = append_best_of_json(json, &bres);
        http_release_generation_slot(ctx, units, !batched);
        if (json)
            http_send_response_ex(fd, HTTP_OK, "application/json", json, 0, extra_headers);
        else
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        free(json);
    }

    free(st.accum_buf);
    if (free_prompt) free(prompt);
    chat_request_free(&req);
}

/* ── Per-connection handler ──────────────────────────────────────────────── */

static void handle_connection(int fd, ApiContext *ctx) {
    /* Receive the full request (headers and body), framing validated on the way. */
    char *recv_buf       = NULL;
    size_t total         = 0;
    const char *read_err = NULL;
    int read_status      = http_request_read(fd, &recv_buf, &total, &read_err);
    if (read_status != 0) {
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

    /* ── CORS + auth gate (applied ahead of dispatch; /health always open) ── */
    char extra_headers[HTTP_EXTRA_HDRS_BUF];
    extra_headers[0] = '\0';
    if (ctx->server_config.cors.enabled) {
        cors_format_headers(&ctx->server_config.cors, req.has_origin ? req.origin : NULL,
                            extra_headers, sizeof(extra_headers));
    }

    http_request_attach_id(&req, extra_headers, sizeof(extra_headers)); /* echoed on every reply */

    int is_health    = (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0);
    int is_preflight = (strcmp(req.method, "OPTIONS") == 0);

    /* DNS-rebinding guard (everything except /health, which monitors probe). */
    if (!is_health && !api_host_allowed(req.host, &ctx->server_config.cors)) {
        metrics_record_request(&ctx->metrics, HTTP_FORBIDDEN);
        http_send_error_ex(fd, HTTP_FORBIDDEN,
                           "Host not allowed (this server answers to localhost / 127.0.0.1; "
                           "behind a proxy add its address with --cors --cors-origin)",
                           extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    if (is_preflight) {
        http_send_response_ex(fd, HTTP_NO_CONTENT, "text/plain", NULL, 0, extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    if (!is_health && !auth_check_request(&ctx->server_config.auth,
                                          req.has_authorization ? req.authorization : NULL)) {
        metrics_record_request(&ctx->metrics, HTTP_UNAUTHORIZED);
        http_send_error_ex(fd, HTTP_UNAUTHORIZED, "Unauthorized", extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    /* Rate limit the cheap non-generation API routes (generation has its own
     * queue; /health, /metrics and the web UI's static files are not limited). */
    if (strncmp(req.path, "/v1/", 4) == 0 && strncmp(req.path, "/v1/chat/completions", 20) != 0) {
        int retry = 1;
        if (!rate_limit_take(&ctx->rate_limiter, rate_limit_now(), &retry)) {
            char hdrs[HTTP_RETRY_HDRS_BUF];
            snprintf(hdrs, sizeof hdrs, "%sRetry-After: %d\r\n", extra_headers, retry);
            metrics_record_request(&ctx->metrics, HTTP_TOO_MANY_REQUESTS);
            http_send_error_ex(fd, HTTP_TOO_MANY_REQUESTS, "Too many requests; slow down", hdrs);
            free(recv_buf);
            close(fd);
            return;
        }
    }

    /* Cross-site guard for every mutating route (the web UI and API clients
     * pass; a foreign web page cannot, because browsers always attach Origin). */
    if (strcmp(req.method, "POST") == 0 &&
        !api_post_allowed(req.content_type, req.has_origin ? req.origin : NULL, req.host,
                          &ctx->server_config.cors, req.body_len)) {
        metrics_record_request(&ctx->metrics, HTTP_FORBIDDEN);
        http_send_error_ex(fd, HTTP_FORBIDDEN,
                           "Cross-site request refused (the Origin is not this server or "
                           "an allowed --cors-origin)",
                           extra_headers);
        free(recv_buf);
        close(fd);
        return;
    }

    /* ── Route dispatch ──────────────────────────────────────────────────── */
    int status = HTTP_OK;
    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        http_handle_get_models(fd, ctx, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/config") == 0) {
        http_handle_get_config(fd, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/config/defaults") == 0) {
        status = http_handle_config_save(fd, req.body, req.body_len, req.content_type,
                                         req.has_origin ? req.origin : NULL, req.host, ctx,
                                         extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/update") == 0) {
        http_send_json_heap(fd, HTTP_OK, api_update_json(0), extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/update/check") == 0) {
        http_send_json_heap(fd, HTTP_OK, api_update_json(1), extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/prompt_cache") == 0) {
        http_handle_prompt_cache_stats(fd, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/prompt_cache/clear") == 0) {
        status = http_handle_prompt_cache_clear(fd, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/selfcheck") == 0) {
        status = http_handle_selfcheck(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/watermark/detect") == 0) {
        status = http_handle_watermark_detect(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        /* Always available, regardless of --web-ui — distinct from "/",
         * which now serves the web UI (Phase 22.2). */
        http_handle_health(fd, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/metrics") == 0) {
        if (ctx->server_config.metrics.enabled) {
            http_handle_get_metrics(fd, ctx, extra_headers);
        } else {
            status = HTTP_NOT_FOUND;
            http_send_error_ex(fd, HTTP_NOT_FOUND, "Metrics disabled (start with --metrics)",
                               extra_headers);
        }
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/docs") == 0) {
        http_handle_get_docs(fd, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/openapi.json") == 0) {
        http_handle_get_openapi(fd, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/chat/completions") == 0) {
        handle_chat_completions(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions/cancel") == 0) {
        http_handle_cancel(fd, req.body, req.body_len, ctx, extra_headers);
    } else if (strcmp(req.method, "GET") == 0 && ctx->server_config.web_ui != WEBUI_MODE_OFF) {
        /* Web UI (Phase 22.2): "/" and files under "/assets/", placed after
         * every API route so nothing above is ever shadowed. Any other
         * unmatched GET also falls through here for client-side SPA routing. */
        status = static_assets_serve(fd, req.path, &ctx->server_config, extra_headers);
    } else {
        status = HTTP_NOT_FOUND;
        http_send_error_ex(fd, HTTP_NOT_FOUND, "Not found", extra_headers);
    }
    metrics_record_request(&ctx->metrics, status);
    if (g_tn_verbose) fprintf(stderr, "[API] request %s -> %d\n", req.request_id, status);

    free(recv_buf);
    close(fd);
}

/* ── Per-connection thread trampoline (Phase 22 concurrency rearchitecture) ── */

typedef struct {
    int conn_fd;
    ApiContext *ctx;
} ConnArgs;

static void *connection_thread_fn(void *arg) {
    ConnArgs *ca = (ConnArgs *)arg;
    metrics_connection_opened(&ca->ctx->metrics);
    handle_connection(ca->conn_fd, ca->ctx);
    metrics_connection_closed(&ca->ctx->metrics);
    atomic_fetch_sub(&ca->ctx->active_connections, 1);
    free(ca);
    return NULL;
}

/* ── Listener thread ─────────────────────────────────────────────────────── */

static void *listener_thread_fn(void *arg) {
    ApiContext *ctx = (ApiContext *)arg;
    /* api_server_stop() sets ctx->server_fd = -1 after closing it; read it once,
     * here (ordered after api_server_start by pthread_create), so the accept
     * loop never races with that write. */
    const int listen_fd = ctx->server_fd;

    while (atomic_load(&ctx->running)) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int conn_fd        = accept(listen_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (conn_fd < 0) {
            if (!atomic_load(&ctx->running)) break;
            continue;
        }

        if (g_tn_verbose) {
            /* The last octet is masked: the log names the network, not the client. */
            char peer[INET_ADDRSTRLEN + 2];
            net_format_peer_masked(&client_addr.sin_addr, peer, sizeof(peer));
            fprintf(stderr, "[API] Connection from %s\n", peer);
        }

        /* Bound the work a client can hold: a read/write timeout so a stalled
         * (slowloris) peer cannot pin its thread forever, and a connection cap
         * so a flood cannot spawn unbounded threads. 0 disables either. */
        if (ctx->server_config.client_timeout_s > 0) {
            struct timeval tv;
            tv.tv_sec  = ctx->server_config.client_timeout_s;
            tv.tv_usec = 0;
            setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            setsockopt(conn_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }
        int cap = ctx->server_config.max_connections;
        if (cap > 0 && atomic_load(&ctx->active_connections) >= cap) {
            static const char body[] =
                "{\"error\":{\"message\":\"Too many open connections\",\"type\":\"server_error\"}}";
            char busy[HTTP_BUSY_REPLY_BUF];
            int bn = snprintf(busy, sizeof(busy),
                              "HTTP/1.1 503 Service Unavailable\r\n"
                              "Content-Type: application/json\r\n"
                              "Retry-After: 1\r\n"
                              "Connection: close\r\n"
                              "Content-Length: %zu\r\n\r\n%s",
                              sizeof(body) - 1, body);
            if (bn > 0) (void)net_write_all(conn_fd, busy, (size_t)bn);
            close(conn_fd);
            continue;
        }

        ConnArgs *ca = (ConnArgs *)malloc(sizeof(ConnArgs));
        if (!ca) {
            close(conn_fd);
            continue;
        }
        ca->conn_fd = conn_fd;
        ca->ctx     = ctx;

        pthread_t tid;
        atomic_fetch_add(&ctx->active_connections, 1);
        if (pthread_create(&tid, NULL, connection_thread_fn, ca) != 0) {
            atomic_fetch_sub(&ctx->active_connections, 1);
            close(conn_fd);
            free(ca);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void api_context_init(ApiContext *ctx, const Config *cfg, const TransformerWeights *weights,
                      RunState *run_state, const MoEConfig *moe_cfg, Tokenizer *tok,
                      ThreadPool *tp) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg       = cfg;
    ctx->weights   = weights;
    ctx->run_state = run_state;
    ctx->moe_cfg   = moe_cfg;
    ctx->tok       = tok;
    ctx->tp        = tp;
    ctx->server_fd = -1;
    atomic_store(&ctx->running, 0);
    server_config_init(&ctx->server_config);
    metrics_init(&ctx->metrics);
    cancel_state_init(&ctx->cancel);
    pthread_mutex_init(&ctx->generation_mutex, NULL);
    gen_queue_init(&ctx->gen_queue, 0);
}

static void start_parallel(ApiContext *ctx);

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

    /* Phase 21.5 / 26: the live prefix record and snapshot store exist for
     * the server's lifetime (the parallel scheduler's slot 0 shares them). */
    if (!ctx->prefix_cache.tokens && ctx->run_state)
        prefix_cache_init(&ctx->prefix_cache, ctx->run_state->max_seq_len);
    if (!ctx->prompt_store_ready) {
        prompt_store_init(&ctx->prompt_store,
                          (size_t)(ctx->server_config.prompt_cache_mb > 0
                                       ? ctx->server_config.prompt_cache_mb
                                       : 0) *
                              1024u * 1024u,
                          ctx->server_config.prompt_cache_min_tokens);
        ctx->prompt_store_ready = 1;
    }
    start_parallel(ctx);

    rate_limit_init(&ctx->rate_limiter, (double)ctx->server_config.rate_limit,
                    2.0 * (double)ctx->server_config.rate_limit);
    atomic_store(&ctx->running, 1);

    /* Spawn listener thread */
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t));
    if (!tid) {
        close(ctx->server_fd);
        ctx->server_fd = -1;
        atomic_store(&ctx->running, 0);
        return TN_ERR_THREAD_CREATE;
    }
    if (pthread_create(tid, NULL, listener_thread_fn, ctx) != 0) {
        free(tid);
        close(ctx->server_fd);
        ctx->server_fd = -1;
        atomic_store(&ctx->running, 0);
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

/* Phase 23 stage 2b: --parallel N > 1 starts the batch scheduler when the
 * model can run batched; otherwise (or if it cannot be set up) the server
 * stays sequential and /v1/config says why. */
static void start_parallel(ApiContext *ctx) {
    int want            = ctx->server_config.parallel;
    ctx->parallel_slots = 1;
    ctx->parallel_note  = NULL;
    if (want > 1 && !batch_scheduler_supported(ctx->weights, ctx->moe_cfg)) {
        ctx->parallel_note = "this model architecture runs one request at a time "
                             "(batched serving covers dense attention models)";
    } else if (want > 1) {
        int rows = ctx->server_config.parallel_batch;
        if (rows < want) rows = want; /* every slot needs a row per step */
        /* All slots share the KV budget one context was sized from, or
         * the larger pool --kv-ram-fraction of free memory holds. */
        long long budget = kv_strategy_budget_tokens(ctx->cfg, ctx->moe_cfg, tn_get_free_ram());
        if (budget < ctx->run_state->max_seq_len) budget = ctx->run_state->max_seq_len;
        BatchSchedulerOptions o = {want, rows, budget};
        ctx->sched =
            batch_scheduler_create(ctx->cfg, ctx->weights, ctx->run_state, &ctx->prefix_cache,
                                   &ctx->prompt_store, ctx->moe_cfg, ctx->tok, ctx->tp, &o);
        if (ctx->sched) {
            ctx->parallel_slots     = want;
            ctx->parallel_kv_budget = budget;
        } else {
            ctx->parallel_note = "could not allocate the parallel slots (out of memory?)";
        }
    }
    gen_queue_set_capacity(&ctx->gen_queue, ctx->parallel_slots);
    if (want > 1)
        printf("[API] Parallel: %d request%s at once%s%s%s\n", ctx->parallel_slots,
               ctx->parallel_slots == 1 ? "" : "s", ctx->parallel_note ? " — " : "",
               ctx->parallel_note ? ctx->parallel_note : "",
               ctx->parallel_note ? "" : " (batched)");
}

void api_server_stop(ApiContext *ctx) {
    if (!ctx || ctx->server_fd < 0) return;
    atomic_store(&ctx->running, 0);
    /* Wake the blocking accept() in listener_thread_fn. close() alone is
     * *not* reliable here: POSIX explicitly leaves "another thread blocked
     * in accept() on an fd this thread closes" unspecified, and in practice
     * on Linux a bare close() from a different thread frequently does not
     * unblock a concurrent accept() at all — confirmed by this exact hang
     * the first time api_server_stop() became reachable (previously dead
     * code, since main()'s pause() had no signal handler installed and so
     * never returned). shutdown(SHUT_RDWR) first reliably makes the
     * blocked accept() return (ECONNABORTED/EINVAL, both handled by the
     * `if (!atomic_load(&ctx->running)) break;` check below) before the fd is closed. */
    shutdown(ctx->server_fd, SHUT_RDWR);
    close(ctx->server_fd);
    ctx->server_fd = -1;
    if (ctx->listener_thread) {
        pthread_join(*(pthread_t *)ctx->listener_thread, NULL);
        free(ctx->listener_thread);
        ctx->listener_thread = NULL;
    }
    /* Generations still running would read the model main() frees next:
     * cancel them all (also any that start from here on) and wait until the
     * queue is empty, then stop the scheduler thread. 2026-10-22: before,
     * an in-flight request kept generating on freed weights at exit. */
    cancel_registry_shutdown(&ctx->cancel);
    for (int waited_ms = 0;
         gen_queue_in_use(&ctx->gen_queue) > 0 || gen_queue_waiting(&ctx->gen_queue) > 0;
         waited_ms += 10) {
        if (waited_ms == 2000)
            fprintf(stderr, "[API] waiting for in-flight generations to stop...\n");
        struct timespec ts = {0, 10 * 1000000L};
        nanosleep(&ts, NULL);
    }
    batch_scheduler_stop(ctx->sched);

    /* Drain the detached connection threads: wait (bounded) until every one
     * has finished. An idle client is dropped by its read timeout
     * (--client-timeout), so the wait ends within that time; the cap keeps a
     * long timeout from delaying exit more than a few seconds. */
    {
        int cap_ms = ctx->server_config.client_timeout_s > 0
                         ? ctx->server_config.client_timeout_s * 1000 + HTTP_INTERNAL_ERROR
                         : 0;
        if (cap_ms > 5000 || cap_ms == 0) cap_ms = 5000;
        int waited_ms = 0;
        while (atomic_load(&ctx->active_connections) > 0 && waited_ms < cap_ms) {
            struct timespec ts = {0, 10 * 1000000L};
            nanosleep(&ts, NULL);
            waited_ms += 10;
        }
        int left = atomic_load(&ctx->active_connections);
        if (left > 0)
            fprintf(stderr, "[API] %d connection(s) still open after %d ms; exiting anyway\n", left,
                    cap_ms);
    }

    /* server_config/cancel/generation_mutex are still not freed here: a
     * connection that outlived the wait above may reference them, so these
     * small, process-lifetime resources are left for the OS to reclaim at
     * exit rather than risking a use-after-free. */
}
