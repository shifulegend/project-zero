#ifndef TN_WEBUI_BUNDLE_H
#define TN_WEBUI_BUNDLE_H

/*
 * Phase 22.2 — interface between the generated embedded web UI bundle
 * (src/api/webui_bundle_generated.c, produced by tools/gen_webui_bundle.c
 * from webui/dist/, committed to git) and the hand-written server code
 * that serves it (src/api/static_assets.c).
 *
 * Never hand-edit webui_bundle_generated.c — edit webui/src and regenerate
 * via `make webui-bundle`.
 */

#include <stddef.h>

typedef struct {
    const char *path;             /* e.g. "/", "/assets/index-XXXX.js" */
    const unsigned char *data;
    size_t len;
    const char *mime_type;
} WebuiAsset;

extern const WebuiAsset g_webui_assets[];
extern const size_t g_webui_assets_count;

#endif /* TN_WEBUI_BUNDLE_H */
