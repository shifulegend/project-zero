#ifndef TN_API_INFO_H
#define TN_API_INFO_H

/*
 * api_info.h — JSON bodies for the discovery endpoints:
 *   GET /v1/models  — OpenAI model list, plus this engine's model metadata
 *   GET /v1/config  — server-side generation defaults, valid ranges and
 *                     feature flags. The web UI builds every control from
 *                     this response, so nothing in the front end is a
 *                     hardcoded default or limit.
 * Return heap strings (caller frees) or NULL on OOM.
 */

#include "api/api_server.h"

char *api_models_json(const ApiContext *ctx);
char *api_config_json(const ApiContext *ctx);

/* Effective per-request max_tokens cap: server_config.max_tokens_limit when
 * set, else the model context length. */
int api_max_tokens_limit(const ApiContext *ctx);

#endif /* TN_API_INFO_H */
