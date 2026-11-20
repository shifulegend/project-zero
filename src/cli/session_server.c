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
        fprintf(stderr, "Error: out of memory copying a server setting\n");
        exit(1);
    }
    return c;
}

/* Generation flags become the server-side defaults every request (and the web UI, via
 * GET /v1/config) starts from; the rest are the API hardening and UI knobs. */
static void configure_server(PzSession *S, ApiContext *ctx) {
    /* Phase 21: Start OpenAI-compatible API server */

    /* Phase 22: API hardening knobs, off by default */
    ctx->server_config.cors.enabled = S->args.cors_enabled;
    for (int ci = 0; ci < S->args.num_cors_origins; ci++) {
        server_config_add_cors_origin(&ctx->server_config, S->args.cors_origins[ci]);
    }
    if (S->args.api_key) ctx->server_config.auth.api_key = dup_setting(S->args.api_key);
    ctx->server_config.metrics.enabled   = S->args.metrics_enabled;
    ctx->server_config.web_ui            = S->args.web_ui_mode;
    ctx->server_config.act_quant         = S->args.act_quant;
    ctx->server_config.quality_tolerance = S->args.quality_tolerance;
    ctx->server_config.self_check_tokens = S->args.self_check_tokens;
    /* Generation flags become the server-side defaults every request
     * (and the web UI, via GET /v1/config) starts from. */
    args_fill_gen_params(&S->args, &ctx->server_config.gen_defaults);
    /* Phase 30: --mcts* are the request defaults; --prm the judge. */
    ctx->server_config.mcts     = S->args.mcts;
    ctx->server_config.mcts_max = S->args.mcts_max;
    if (S->prm.loaded) {
        const char *pb               = strrchr(S->args.prm_path, '/');
        ctx->prm                     = &S->prm;
        ctx->server_config.prm_model = pb ? pb + 1 : S->args.prm_path;
    }
    {
        const char *base            = strrchr(S->args.model_path, '/');
        ctx->server_config.model_id = dup_setting(base ? base + 1 : S->args.model_path);
    }
    if (S->args.ui_title) ctx->server_config.ui_title = dup_setting(S->args.ui_title);
    ctx->server_config.ui_logo                = S->args.ui_logo;
    ctx->server_config.prompt_cache           = S->args.prompt_cache ? 1 : 0;
    ctx->server_config.chat_format            = S->args.chat_format;
    ctx->server_config.allow_request_template = S->args.no_request_template ? 0 : 1;
    ctx->server_config.allow_config_write     = S->args.no_config_write ? 0 : 1;
    ctx->server_config.chat_template_source   = !S->t.chat_template          ? "none"
                                                : S->chat_template_from_file ? "file"
                                                : S->chat_template_builtin   ? "builtin"
                                                                             : "gguf";
    ctx->server_config.best_of                = S->args.best_of;
    ctx->server_config.queue_size             = S->args.queue_size;
    ctx->server_config.queue_timeout_s        = S->args.queue_timeout_s;
    ctx->server_config.rate_limit             = S->args.rate_limit;
    ctx->server_config.client_timeout_s       = S->args.client_timeout_s;
    ctx->server_config.max_connections        = S->args.max_connections;
    ctx->server_config.parallel               = S->args.parallel;
    ctx->server_config.parallel_batch         = S->args.parallel_batch;
    gen_queue_set_max_waiting(&ctx->gen_queue, S->args.queue_size);
    ctx->server_config.best_of_max             = S->args.best_of_max;
    ctx->server_config.prompt_cache_mb         = S->args.prompt_cache_mb;
    ctx->server_config.prompt_cache_min_tokens = S->args.prompt_cache_min_tokens;
    ctx->server_config.max_tokens_limit =
        (S->args.max_tokens_limit > 0 && S->args.max_tokens_limit < S->p.seq_len)
            ? S->args.max_tokens_limit
            : S->p.seq_len;
    if (S->args.static_dir) ctx->server_config.static_dir = dup_setting(S->args.static_dir);
    if (S->args.vision_path && S->args.proj_path) {
        ctx->server_config.vision_path = dup_setting(S->args.vision_path);
        ctx->server_config.proj_path   = dup_setting(S->args.proj_path);
    }
}

void session_run_server(PzSession *S) {
    ApiContext api_ctx;
    api_context_init(&api_ctx, &S->p, &S->w, S->s, &S->mc, &S->t, S->tp);
    configure_server(S, &api_ctx);

    TernaryError api_err = api_server_start(S->args.server_port, &api_ctx);
    if (api_err != TN_OK) {
        S->exit_code = 1;
        fprintf(stderr, "Error: Failed to start API server on port %d: %s\n", S->args.server_port,
                tn_error_str(api_err));
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
