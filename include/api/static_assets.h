#ifndef TN_STATIC_ASSETS_H
#define TN_STATIC_ASSETS_H

/*
 * Phase 22.2 — serves the web chat UI: either the embedded bundle
 * (src/api/webui_bundle_generated.c, the default) or, when
 * ServerConfig.static_dir is set, files read straight off disk (dev mode,
 * so contributors can iterate on webui/dist/ without regenerating the
 * embedded bundle each time).
 *
 * Falls back to serving "/" (index.html) for any unmatched GET path that
 * isn't under /assets/, so client-side SPA routing works.
 */

#include "api/server_config.h"
#include "api/webui_bundle.h"

/* Testable core: looks up `path` within the given manifest (exact match),
 * falling back to the "/" entry for any path NOT under "/assets/" (client-
 * side SPA routing) — returns NULL only for an unmatched path under "/assets/".
 * Takes an explicit manifest/count (rather than reading the global
 * g_webui_assets) so tests can exercise the lookup/fallback logic against a
 * small hand-written manifest instead of the real embedded webui bundle. */
const WebuiAsset *static_assets_find_in(const WebuiAsset *assets, size_t count, const char *path);

/* Writes a full HTTP response (headers + body) for `path` to fd.
 * extra_headers, if non-NULL, must already be "\r\n"-terminated per line
 * (e.g. CORS headers) and is spliced into the header block.
 * Returns the HTTP status code that was written (200 or 404), so the
 * caller can record it in /metrics. */
int static_assets_serve(int fd, const char *path, const ServerConfig *cfg,
                         const char *extra_headers);

#endif /* TN_STATIC_ASSETS_H */
