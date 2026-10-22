#ifndef TN_CONFIG_ENDPOINT_H
#define TN_CONFIG_ENDPOINT_H

/*
 * config_endpoint.h — Phase 40.3 (2026-10-22): where each server default
 * comes from, and saving defaults from the web UI to the `pz config` file.
 *
 * GET /v1/config gains
 *   "config": {"path": str|null, "file_used": bool, "writable": bool,
 *              "keys": {"<json name>": "<config key>", ...},
 *              "sources": {"<json name>": "default"|"file"|"env"|"cli", ...}}
 * for the settings in /v1/config "defaults" (plus "ui_title") that have a
 * config key.
 *
 * POST /v1/config/defaults
 *   {"key": "temperature", "value": 0.7}        number, string or (stop) array
 *   {"key": "temperature", "unset": true}       back to the built-in default
 * The value is validated by the config schema and written to the file (the
 * same one `pz config set` writes). It applies from the next start; the
 * response says so, and names a flag or environment variable that would
 * still override it:
 *   {"saved": true, "key": ..., "config_key": ..., "path": ..., "applies": "restart",
 *    "overridden_by": "cli"|"env"|null, "file_used": bool}
 * Refused (403) with --no-config-write, without a config location, or when
 * the request fails api_config_write_allowed(). Never touches the
 * generation mutex.
 */

#include "api/api_server.h"
#include "api/json_write.h"

/* Writes the "config" member (key and object, no surrounding comma). */
void api_config_sources_json(JsonBuf *jb, const ApiContext *ctx);

/* Cross-site request guard for a state-changing route on a server with no
 * API key: the body must be declared JSON (a browser cannot send that
 * cross-origin without a CORS preflight, which only --cors answers), and an
 * Origin header, when present, must match Host or be a CORS-allowed origin.
 * Returns 1 when allowed. Any argument may be NULL. */
int api_config_write_allowed(const char *content_type, const char *origin,
                             const char *host, const CorsConfig *cors);

/* Returns a heap JSON body (caller frees) and sets *status (200/400/403/500). */
char *api_config_save(const ApiContext *ctx, const char *json_body, int *status);

#endif /* TN_CONFIG_ENDPOINT_H */
