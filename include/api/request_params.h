#ifndef TN_REQUEST_PARAMS_H
#define TN_REQUEST_PARAMS_H

/*
 * request_params.h — merge a parsed ChatRequest over the server's defaults.
 *
 * The server's defaults are the CLI generation flags (ServerConfig.gen_defaults);
 * the client (API caller or the web UI) overrides any subset per request. All
 * values are clamped to their valid ranges; max_tokens is capped by
 * `max_tokens_limit` (the model context length unless --max-tokens-limit is
 * lower). Stop strings and the watermark key are borrowed from `req`, so the
 * result is valid only while `req` is alive.
 */

#include "api/chat_request.h"
#include "transformer/gen_params.h"

void request_params_resolve(const ChatRequest *req, const GenerationParams *defaults,
                            int max_tokens_limit, GenerationParams *out);

/* Phase 30: the server's --mcts* defaults overlaid with a request's "mcts"
 * fields (non-zero ones), clamped to the shared ranges and to sim_cap
 * (--mcts-max). simulations 0 = no search. */
void request_mcts_resolve(const ChatRequest *req, const MctsOptions *defaults, int sim_cap,
                          MctsOptions *out);

#endif /* TN_REQUEST_PARAMS_H */
