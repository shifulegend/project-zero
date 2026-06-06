#ifndef TN_API_SERVER_H
#define TN_API_SERVER_H

/*
 * Phase 21 — OpenAI-Compatible API Layer
 *
 * Public interface for the embedded HTTP API server.
 *
 * Usage (from main.c):
 *   ApiContext ctx;
 *   api_context_init(&ctx, cfg, weights, run_state, moe_cfg, tok, tp);
 *   api_server_start(port, &ctx);   // launches listener thread
 *   // ... main thread blocks or runs REPL ...
 *   api_server_stop(&ctx);
 */

#include "core/error.h"
#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

/* ── Inference context passed from main to API handler ──────────────────── */
typedef struct {
    const Config           *cfg;
    const TransformerWeights *weights;
    RunState               *run_state;
    const MoEConfig        *moe_cfg;
    Tokenizer              *tok;
    ThreadPool             *tp;

    /* Server state (filled by api_server_start) */
    int   server_fd;            /* listening socket fd, -1 when stopped */
    void *listener_thread;      /* pthread_t, opaque */
    volatile int running;       /* 1 = running, 0 = stop requested */
} ApiContext;

/* Initialise an ApiContext with all inference state.
 * Does NOT start the server. */
void api_context_init(ApiContext *ctx,
                      const Config *cfg,
                      const TransformerWeights *weights,
                      RunState *run_state,
                      const MoEConfig *moe_cfg,
                      Tokenizer *tok,
                      ThreadPool *tp);

/* Start the HTTP listener on the given port.
 * Spawns a background thread; returns immediately.
 * Returns TN_OK on success or an error code. */
TernaryError api_server_start(int port, ApiContext *ctx);

/* Signal the server to stop accepting new connections and
 * wait for the listener thread to exit. */
void api_server_stop(ApiContext *ctx);

#endif /* TN_API_SERVER_H */
