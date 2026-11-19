#ifndef TN_SELFCHECK_ENDPOINT_H
#define TN_SELFCHECK_ENDPOINT_H

/*
 * selfcheck_endpoint.h — POST /v1/selfcheck: the engine's accuracy probe (eval/selfcheck.h)
 * on the loaded model with the server's current settings.
 *
 * Request (optional): {"tolerance": 1.0?, "tokens": 96?}  (defaults: --quality-tolerance,
 *                     --self-check-tokens)
 * Response: {"verdict":"PASS|FAIL|SKIPPED","act_quant":"int8","classifier":"bf16",
 *            "perplexity":{"reference":x,"configured":y},"delta_pct":d,"tolerance_pct":t,
 *            "tokens_scored":n,"seconds":s,"note":"..."}
 * The caller (http_server.c) holds every generation unit and generation_mutex around it.
 */

#include "api/api_server.h"

char *api_selfcheck(ApiContext *ctx, const char *json_body, int *status);

#endif /* TN_SELFCHECK_ENDPOINT_H */
