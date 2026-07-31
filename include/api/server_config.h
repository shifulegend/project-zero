#ifndef TN_SERVER_CONFIG_H
#define TN_SERVER_CONFIG_H

/*
 * Phase 22 — Server configuration knobs (CORS, auth, metrics, static serving,
 * web UI), populated from CliArgs in main.c and owned by ApiContext. Keeps
 * http_server.c decoupled from CLI argument parsing.
 */

#define SERVER_CORS_MAX_ORIGINS 16

typedef struct {
    int   enabled;                          /* --cors */
    char *origins[SERVER_CORS_MAX_ORIGINS]; /* --cors-origin (repeatable); "*" = any */
    int   num_origins;
} CorsConfig;

typedef struct {
    char *api_key; /* --api-key; NULL = auth disabled (default) */
} AuthConfig;

typedef struct {
    int enabled; /* --metrics */
} MetricsConfig;

typedef enum {
    WEBUI_MODE_AUTO = 0, /* serve the UI whenever server_mode is set */
    WEBUI_MODE_ON,
    WEBUI_MODE_OFF
} WebUiMode;

typedef struct {
    CorsConfig    cors;
    AuthConfig    auth;
    MetricsConfig metrics;
    WebUiMode     web_ui;
    char         *static_dir;  /* --static-dir: serve UI from disk instead of the embedded bundle */
    /* Phase 22.2: image uploads via the web UI's image_url content parts.
     * Mirrors CliArgs.vision_path/proj_path — NULL means vision is disabled
     * for --server mode, matching the CLI's own --image/--vision/--proj
     * requirement (both must be set). */
    char         *vision_path;
    char         *proj_path;
} ServerConfig;

/* Zero-initialise. Must be called before use. */
void server_config_init(ServerConfig *sc);

/* Adds an allowed CORS origin (or "*"). Returns 0 on success, -1 if the
 * origin list is full or origin is NULL. */
int server_config_add_cors_origin(ServerConfig *sc, const char *origin);

/* Frees heap-allocated fields (origins, api_key, static_dir). */
void server_config_free(ServerConfig *sc);

#endif /* TN_SERVER_CONFIG_H */
