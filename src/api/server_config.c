/*
 * Phase 22 — src/api/server_config.c
 * See include/api/server_config.h for responsibility.
 */

#include "api/server_config.h"
#include <stdlib.h>
#include <string.h>

void server_config_init(ServerConfig *sc) {
    if (!sc) return;
    memset(sc, 0, sizeof(*sc));
    sc->web_ui = WEBUI_MODE_AUTO;
}

int server_config_add_cors_origin(ServerConfig *sc, const char *origin) {
    if (!sc || !origin) return -1;
    if (sc->cors.num_origins >= SERVER_CORS_MAX_ORIGINS) return -1;
    char *copy = strdup(origin);
    if (!copy) return -1;
    sc->cors.origins[sc->cors.num_origins++] = copy;
    return 0;
}

void server_config_free(ServerConfig *sc) {
    if (!sc) return;
    for (int i = 0; i < sc->cors.num_origins; i++) {
        free(sc->cors.origins[i]);
        sc->cors.origins[i] = NULL;
    }
    sc->cors.num_origins = 0;
    free(sc->auth.api_key);
    sc->auth.api_key = NULL;
    free(sc->static_dir);
    sc->static_dir = NULL;
    free(sc->vision_path);
    sc->vision_path = NULL;
    free(sc->proj_path);
    sc->proj_path = NULL;
}
