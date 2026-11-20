/* Chat completions route (POST /v1/chat/completions): parse, wait for engine units, render the
 * prompt, run the generation and write the reply. Split out of http_server.c so each step is a
 * function of its own (acquire units, build prompt, run, send). */

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api/api_info.h"
#include "api/api_server.h"
#include "api/batch_scheduler.h"
#include "api/cancel.h"
#include "api/chat_compile.h"
#include "api/chat_image.h"
#include "api/chat_request.h"
#include "api/chat_stream.h"
#include "api/gen_queue.h"
#include "api/http_request.h"
#include "api/http_response.h"
#include "api/http_routes.h"
#include "api/http_status.h"
#include "api/json_write.h"
#include "api/metrics.h"
#include "api/net_io.h"
#include "api/request_params.h"
#include "api/sse_stream.h"
#include "core/log.h"
#include "reasoning/best_of.h"
#include "reasoning/mcts.h"
#include "transformer/generate.h"

#define HTTP_MAX_PROMPT_BYTES (65536) /* compiled prompt size cap */
#define HTTP_SSE_EVENT_BUF 160        /* one queue-failure SSE event */

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

/* Waits for `units` engine units. Returns 1 once they are held; on any failure the request is
 * freed, the client has its answer, and 0 is returned. */
static int chat_acquire_units(int fd, ApiContext *ctx, ChatRequest *reqp, int units,
                              QueueWaitCtx *qwp, const char *extra_headers) {
    /* With --parallel N, N batched requests run at once; otherwise one
     * generation at a time. With --queue-size N, up to N requests wait their
     * turn in arrival order (Phase 23 stage 1); otherwise (the default) a
     * busy server answers 429 immediately. Unrelated routes
     * (metrics/docs/static/cancel) never touch the queue. */
    *qwp              = (QueueWaitCtx){fd, reqp->output.stream ? 1 : 0, 0, extra_headers};
    QueueWaitCtx *qw  = qwp;
    GenQueueResult qr = gen_queue_acquire_n(&ctx->gen_queue, units,
                                            ctx->server_config.limits.queue_timeout_s * 1000,
                                            queue_wait_notify, qw);
    if (qr == GEN_QUEUE_OK && !atomic_load(&ctx->running)) { /* shutting down: don't start */
        gen_queue_release_n(&ctx->gen_queue, units);
        chat_request_free(reqp);
        if (!qw->headers_sent)
            http_send_error_ex(fd, HTTP_SERVICE_UNAVAILABLE, "Server is shutting down",
                               extra_headers);
        return 0;
    }
    if (qr != GEN_QUEUE_OK) {
        chat_request_free(reqp);
        if (qr == GEN_QUEUE_ABORTED) return 0; /* client already gone */
        const char *why = qr == GEN_QUEUE_TIMEOUT ? "Timed out waiting for the generation queue"
                          : ctx->server_config.limits.queue_size > 0
                              ? "Generation queue is full"
                              : "Another generation is already in flight";
        if (qw->headers_sent) {
            char ev[HTTP_SSE_EVENT_BUF];
            int n = snprintf(ev, sizeof(ev),
                             "data: {\"error\":{\"message\":\"%s\",\"type\":\"server_error\"}}\n\n",
                             why);
            if (n > 0) (void)net_write_all(fd, ev, (size_t)n); /* peer gone: stream already ended */
        } else {
            http_send_error_ex(fd, HTTP_TOO_MANY_REQUESTS, why, extra_headers);
        }
        return 0;
    }
    return 1;
}

/* Resolves the generation settings and renders the prompt: the GGUF chat template handles the
 * whole history itself (gp->roles/contents), a legacy tokenizer gets a compiled prompt that the
 * caller frees when *free_prompt is set. `roles`/`contents` must outlive `gp`. Returns 1 on
 * success; on failure the slot is released, the client has its answer, and 0 is returned. */
static int chat_build_prompt(int fd, ApiContext *ctx, ChatRequest *req, GenerationParams *gp,
                             const char **roles, const char **contents, int units, int batched,
                             const char *extra_headers, char **prompt, int *free_prompt) {
    /* Compile messages into a single prompt.
     * If the tokenizer has an embedded chat template (GGUF), generate()
     * handles template application internally.  For legacy tokenizers without
     * a built-in template, we apply a template here. */
    /* Resolve generation settings: server defaults (CLI flags) overlaid
     * with whatever the client sent, clamped to valid ranges. */
    request_params_resolve(req, &ctx->server_config.gen_defaults, api_max_tokens_limit(ctx), gp);

    /* Per-request templates only when the operator allows them. */
    if (!ctx->server_config.chat.allow_request_template) gp->chat_template = NULL;

    if (ctx->tok->chat_template || gp->chat_template) {
        /* GGUF tokenizer: the model's own Jinja template renders the whole
         * conversation (system/user/assistant turns). 2026-10-18: this used
         * to forward only the last user message, silently dropping all
         * earlier turns and any system message — multi-turn chat in the web
         * UI had no memory. */
        int n = 0;
        for (int i = 0; i < req->num_messages && n < CHAT_MAX_MESSAGES; i++) {
            if (!req->messages[i].content) continue;
            roles[n]      = req->messages[i].role[0] ? req->messages[i].role : "user";
            contents[n++] = req->messages[i].content;
        }
        gp->roles      = roles;
        gp->contents   = contents;
        gp->n_messages = n;
        *prompt        = (char *)(n > 0 ? contents[n - 1] : "");
    } else {
        /* Legacy tokenizer: compile all messages into a formatted prompt */
        char *compiled = (char *)malloc(HTTP_MAX_PROMPT_BYTES);
        if (!compiled) {
            chat_request_free(req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
            return 0;
        }
        ChatTemplateType tmpl = ctx->server_config.chat.chat_format >= 0
                                    ? (ChatTemplateType)ctx->server_config.chat.chat_format
                                    : chat_template_detect_vocab(ctx->tok);
        /* Message text is content, never chat markup (prompt-injection
         * defence, 2026-10-22): neutralise control tokens before the
         * built-in formats wrap it in their own. */
        int neutral_ok = 1;
        for (int i = 0; i < req->num_messages && neutral_ok; i++) {
            if (!req->messages[i].content) continue;
            char *safe = tokenizer_neutralize_control(ctx->tok, req->messages[i].content);
            if (!safe) {
                neutral_ok = 0;
                break;
            } /* fail closed */
            free(req->messages[i].content);
            req->messages[i].content = safe;
        }
        if (!neutral_ok) {
            free(compiled);
            chat_request_free(req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
            return 0;
        }
        if (chat_compile_prompt(req, tmpl, compiled, HTTP_MAX_PROMPT_BYTES) != TN_OK) {
            free(compiled);
            chat_request_free(req);
            http_release_generation_slot(ctx, units, !batched);
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "Prompt compilation failed", extra_headers);
            return 0;
        }
        *prompt      = compiled;
        *free_prompt = 1;
    }
    return 1;
}

/* Runs the generation that matches the request: the parallel scheduler, tree search, best-of-N
 * or the plain loop. Outcome goes to `gstats` and, for the two search modes, `bres` / `mres`. */
static void chat_run_generation(ApiContext *ctx, const ChatRequest *req, GenerationParams *gp,
                                char *prompt, int batched, int use_cache, int best_of,
                                const MctsOptions *mopt, StreamState *st, GenerationStats *gstats,
                                BestOfResult *bres, MctsResult *mres) {
    if (batched) {
        batch_scheduler_generate(ctx->sched, prompt, gp, use_cache, chat_streaming_token_callback,
                                 st);
    } else if (mopt->simulations > 0) {
        const char *question = "";
        for (int i = req->num_messages - 1; i >= 0; i--)
            if (req->messages[i].content && strcmp(req->messages[i].role, "user") == 0) {
                question = req->messages[i].content;
                break;
            }
        if (generate_mcts(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp,
                          prompt, question, gp, mopt, ctx->prm, chat_streaming_token_callback, st,
                          chat_best_of_abort_check, mres) != TN_OK &&
            gstats->completion_tokens == 0)
            gstats->finish_reason = GEN_FINISH_ERROR;
    } else if (best_of > 1) {
        generate_best_of(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp,
                         prompt, gp, best_of, BEST_OF_SCORE_MEAN_LOGPROB,
                         chat_streaming_token_callback, st, chat_best_of_abort_check, bres);
    } else {
        generate_ex(ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok, ctx->tp, prompt,
                    gp, chat_streaming_token_callback, st);
    }
}

/* Releases the engine slot and writes the reply (SSE terminator or one JSON body). The slot is
 * freed before the last write: see the 2026-11-11 note below. */
static void chat_send_result(int fd, ApiContext *ctx, const ChatRequest *req, const StreamState *st,
                             const GenerationStats *gstats, const char *id, int units, int batched,
                             int best_of, const MctsOptions *mopt, const BestOfResult *bres,
                             const MctsResult *mres, const char *extra_headers) {
    const char *finish_reason = gen_finish_reason_str(gstats->finish_reason);
    if (gstats->finish_reason == GEN_FINISH_ERROR && !req->output.stream) {
        /* Nothing was generated because the prompt could not be prepared
         * (template render or tokenization failed) — report it instead of an
         * empty 200 completion. */
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
     * from request-local buffers (st->accum_buf, gstats, bres, mres), and
     * `finish_reason` is a static string. */
    if (req->output.stream) {
        http_release_generation_slot(ctx, units, !batched);
        sse_write_done_ex(fd, id, finish_reason, gstats->prompt_tokens, gstats->completion_tokens,
                          gstats->cached_tokens);
    } else {
        /* 2026-07-15: build the JSON body FIRST so Content-Length reflects
         * its real size. The previous code sent "Content-Length: 0" (NULL
         * body) and only then wrote the actual JSON via a second write() —
         * conformant HTTP clients stop reading at 0 bytes and never see it. */
        const char *text = st->accum_buf ? st->accum_buf : "";
        char *json = sse_format_full_response_ex(id, text, finish_reason, gstats->prompt_tokens,
                                                 gstats->completion_tokens, gstats->cached_tokens);
        if (json && mopt->simulations > 0)
            json = chat_append_mcts_json(json, mres);
        else if (json && best_of > 1)
            json = chat_append_best_of_json(json, bres);
        http_release_generation_slot(ctx, units, !batched);
        if (json)
            http_send_response_ex(fd, HTTP_OK, "application/json", json, 0, extra_headers);
        else
            http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        free(json);
    }
}

void http_handle_chat_completions(int fd, const char *body_start, size_t body_len, ApiContext *ctx,
                                  const char *extra_headers) {
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
    int best_of =
        (req.set_mask & CHAT_SET_BEST_OF) ? req.search.best_of : ctx->server_config.search.best_of;
    if (best_of < 1) best_of = 1;
    if (best_of > ctx->server_config.search.best_of_max)
        best_of = ctx->server_config.search.best_of_max;
    MctsOptions mopt;
    request_mcts_resolve(&req, &ctx->server_config.search.mcts, ctx->server_config.search.mcts_max,
                         &mopt);
    int wants_image = 0;
    if (ctx->server_config.vision.vision_path && ctx->server_config.vision.proj_path)
        for (int i = 0; i < req.num_messages; i++)
            if (strcmp(req.messages[i].role, "user") == 0 && req.messages[i].image_data_url)
                wants_image = 1;
    /* Phase 23 stage 2b: plain chat requests share the engine through the
     * parallel scheduler (one unit each); the rest need all of it. */
    int batched = ctx->sched && best_of <= 1 && mopt.simulations <= 0 && !wants_image;
    int units   = batched ? 1 : http_engine_units(ctx);

    QueueWaitCtx qw;
    if (!chat_acquire_units(fd, ctx, &req, units, &qw, extra_headers)) return;
    int had_image = 0;
    if (batched && !(ctx->server_config.vision.vision_path && ctx->server_config.vision.proj_path))
        for (int i = 0; i < req.num_messages; i++)
            if (strcmp(req.messages[i].role, "user") == 0 && req.messages[i].image_data_url) {
                TN_LOG_INF("vision",
                           "[vision] image_url provided but server has no --vision/--proj "
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

    char *prompt    = NULL;
    int free_prompt = 0;
    GenerationParams gp;
    const char *roles[CHAT_MAX_MESSAGES];
    const char *contents[CHAT_MAX_MESSAGES];
    if (!chat_build_prompt(fd, ctx, &req, &gp, roles, contents, units, batched, extra_headers,
                           &prompt, &free_prompt))
        return;

    /* Generate a request ID */
    char id[32];
    http_new_id(id, sizeof(id));

    /* Set up streaming state */
    StreamState st;
    memset(&st, 0, sizeof(st));
    st.conn_fd      = fd;
    st.is_streaming = req.output.stream;
    st.cancel       = &ctx->cancel;
    snprintf(st.id, sizeof(st.id), "%s", id);
    snprintf(st.public_id, sizeof(st.public_id), "chatcmpl-%s", id);

    if (req.output.stream && !qw.headers_sent) {
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
    int use_cache = (req.set_mask & CHAT_SET_CACHE_PROMPT) ? req.cache_prompt
                                                           : ctx->server_config.cache.prompt_cache;
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
    chat_run_generation(ctx, &req, &gp, prompt, batched, use_cache, best_of, &mopt, &st, &gstats,
                        &bres, &mres);
    cancel_registry_end(&ctx->cancel, st.public_id);
    metrics_add_tokens(&ctx->metrics, st.tokens_emitted);

    chat_send_result(fd, ctx, &req, &st, &gstats, id, units, batched, best_of, &mopt, &bres, &mres,
                     extra_headers);

    free(st.accum_buf);
    if (free_prompt) free(prompt);
    chat_request_free(&req);
}
