/* --server: fills the server settings from the command line, starts the HTTP API and waits for
 * SIGINT / SIGTERM. */

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api/api_server.h"
#include "api/server_config.h"
#include "cli/session.h"
#include "core/log.h"

/* --server mode graceful shutdown: set only by handle_shutdown_signal (the
 * only async-signal-safe thing to do in a handler is set a sig_atomic_t
 * flag), polled by the main thread's wait loop below. Scoped to --server
 * mode only — installed right before the blocking wait, not at process
 * start, so REPL/one-shot Ctrl+C behavior (immediate kill, default
 * disposition) is unchanged. */
static volatile sig_atomic_t g_shutdown_requested = 0;

static void handle_shutdown_signal(int sig) {
    (void)sig;
    g_shutdown_requested = 1;
}

/* strdup() for a server setting that must not silently be missing: an
 * unchecked NULL api_key would start the server with authentication OFF. A
 * failed copy is fatal, like every other required allocation in the engine. */
static char *dup_setting(const char *s) {
    char *c = strdup(s);
    if (!c) {
        TN_LOG_ERR("session_server", "Error: out of memory copying a server setting\n");
        exit(1);
    }
    return c;
}

/* Generation flags become the server-side defaults every request (and the web UI, via
 * GET /v1/config) starts from; the rest are the API hardening and UI knobs. */
static void configure_server(PzSession *S, ApiContext *ctx) {
    const CliServer *server_p = &S->args.server;
    /* Phase 21: Start OpenAI-compatible API server */

    /* Phase 22: API hardening knobs, off by default */
    ctx->server_config.cors.enabled = server_p->cors_enabled;
    for (int ci = 0; ci < server_p->num_cors_origins; ci++) {
        server_config_add_cors_origin(&ctx->server_config, server_p->cors_origins[ci]);
    }
    if (server_p->api_key) ctx->server_config.auth.api_key = dup_setting(server_p->api_key);
    ctx->server_config.metrics.enabled            = server_p->metrics_enabled;
    ctx->server_config.ui.web_ui                  = server_p->web_ui_mode;
    ctx->server_config.numerics.act_quant         = S->args.eval.act_quant;
    ctx->server_config.numerics.quality_tolerance = S->args.eval.quality_tolerance;
    ctx->server_config.numerics.self_check_tokens = S->args.eval.self_check_tokens;
    /* Generation flags become the server-side defaults every request
     * (and the web UI, via GET /v1/config) starts from. */
    args_fill_gen_params(&S->args, &ctx->server_config.gen_defaults);
    /* Phase 30: --mcts* are the request defaults; --prm the judge. */
    ctx->server_config.search.mcts     = S->args.search.mcts;
    ctx->server_config.search.mcts_max = S->args.search.mcts_max;
    if (S->prm.loaded) {
        const char *pb                      = strrchr(S->args.search.prm_path, '/');
        ctx->prm                            = &S->prm;
        ctx->server_config.search.prm_model = pb ? pb + 1 : S->args.search.prm_path;
    }
    {
        const char *base            = strrchr(S->args.input.model_path, '/');
        ctx->server_config.model_id = dup_setting(base ? base + 1 : S->args.input.model_path);
    }
    if (server_p->ui_title) ctx->server_config.ui.ui_title = dup_setting(server_p->ui_title);
    ctx->server_config.ui.ui_logo                  = server_p->ui_logo;
    ctx->server_config.cache.prompt_cache          = S->args.perf.prompt_cache ? 1 : 0;
    ctx->server_config.chat.chat_format            = S->args.sampling.chat_format;
    ctx->server_config.chat.allow_request_template = server_p->no_request_template ? 0 : 1;
    ctx->server_config.chat.allow_config_write     = server_p->no_config_write ? 0 : 1;
    ctx->server_config.chat.chat_template_source   = !S->t.chat_template          ? "none"
                                                     : S->chat_template_from_file ? "file"
                                                     : S->chat_template_builtin   ? "builtin"
                                                                                  : "gguf";
    ctx->server_config.search.best_of              = S->args.sampling.best_of;
    ctx->server_config.limits.queue_size           = server_p->queue_size;
    ctx->server_config.limits.queue_timeout_s      = server_p->queue_timeout_s;
    ctx->server_config.limits.rate_limit           = server_p->rate_limit;
    ctx->server_config.limits.client_timeout_s     = server_p->client_timeout_s;
    ctx->server_config.limits.max_connections      = server_p->max_connections;
    ctx->server_config.limits.parallel             = server_p->parallel;
    ctx->server_config.limits.parallel_batch       = server_p->parallel_batch;
    gen_queue_set_max_waiting(&ctx->gen_queue, server_p->queue_size);
    ctx->server_config.search.best_of_max            = S->args.sampling.best_of_max;
    ctx->server_config.cache.prompt_cache_mb         = S->args.perf.prompt_cache_mb;
    ctx->server_config.cache.prompt_cache_min_tokens = S->args.perf.prompt_cache_min_tokens;
    ctx->server_config.limits.max_tokens_limit =
        (S->args.sampling.max_tokens_limit > 0 && S->args.sampling.max_tokens_limit < S->p.seq_len)
            ? S->args.sampling.max_tokens_limit
            : S->p.seq_len;
    if (server_p->static_dir) ctx->server_config.ui.static_dir = dup_setting(server_p->static_dir);
    if (S->args.input.vision_path && S->args.input.proj_path) {
        ctx->server_config.vision.vision_path = dup_setting(S->args.input.vision_path);
        ctx->server_config.vision.proj_path   = dup_setting(S->args.input.proj_path);
    }
}

void session_run_server(PzSession *S) {
    ApiContext api_ctx;
    api_context_init(&api_ctx, &S->p, &S->w, S->s, &S->mc, &S->t, S->tp);
    configure_server(S, &api_ctx);

    TernaryError api_err = api_server_start(S->args.server.server_port, &api_ctx);
    if (api_err != TN_OK) {
        S->exit_code = 1;
        TN_LOG_ERR("session_server", "Error: Failed to start API server on port %d: %s\n",
                   S->args.server.server_port, tn_error_str(api_err));
    } else {
        printf("Press Ctrl+C to stop.\n");

        /* Graceful shutdown: previously a bare pause() with no signal
         * handler installed meant SIGINT/SIGTERM used the default
         * disposition (terminate immediately) — pause() never actually
         * returns under default disposition, so api_server_stop() and
         * all of main()'s later cleanup (tokenizer_free, gguf_header_free,
         * mapped_file_close, etc.) were unreachable dead code in server
         * mode. Installing a handler makes the signal "caught" instead
         * of fatal, which is what makes pause() return. */
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = handle_shutdown_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);

        /* Block main thread — listener runs in background thread.
         * Loop (not a single pause()) so a signal that arrived between
         * installing the handler and the first pause() call — setting
         * the flag before we ever wait — is still honored immediately
         * rather than requiring a second signal. */
        while (!g_shutdown_requested) {
            pause();
        }
        printf("\nShutting down...\n");
        api_server_stop(&api_ctx);
    }
}
