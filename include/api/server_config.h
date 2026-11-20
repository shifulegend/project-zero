#ifndef TN_SERVER_CONFIG_H
#define TN_SERVER_CONFIG_H

/*
 * Phase 22 — Server configuration knobs (CORS, auth, metrics, static serving,
 * web UI), populated from CliArgs in main.c and owned by ApiContext. Keeps
 * http_server.c decoupled from CLI argument parsing.
 */

#include "api/cancel.h"
#include "config/pz_config.h"
#include "core/limits.h"
#include "reasoning/mcts_options.h"
#include "transformer/gen_params.h"

#define SERVER_CORS_MAX_ORIGINS 16

/* --parallel / --parallel-batch bounds and defaults: core/limits.h. */

typedef struct {
    int enabled;                            /* --cors */
    char *origins[SERVER_CORS_MAX_ORIGINS]; /* --cors-origin (repeatable); "*" = any */
    int num_origins;
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

/* Web UI: mode, static directory, title and logo. */
typedef struct {
    WebUiMode web_ui;
    char *static_dir; /* --static-dir: serve UI from disk instead of the embedded bundle */
    char *ui_title;   /* --ui-title: web UI brand name (default "project-zero") */
    int ui_logo;      /* --ui-logo: 0 = ascii block wordmark of ui_title (default), 1 = text */
} ServerUi;

/* Image uploads via the web UI's image_url content parts (mirrors the CLI's --vision/--proj; NULL = vision disabled). */
typedef struct {
    char *vision_path;
    char *proj_path;
} ServerVision;

/* Prompt-cache defaults (Phase 26 snapshot store). */
typedef struct {
    int prompt_cache;            /* --prompt-cache on|off: default for KV prefix reuse (1) */
    int prompt_cache_mb;         /* --prompt-cache-mb: Phase 26 snapshot budget (0 = off) */
    int prompt_cache_min_tokens; /* --prompt-cache-min-tokens: shortest stored sequence */
} ServerCache;

/* Request admission and connection limits. */
typedef struct {
    int max_tokens_limit;
    int queue_size;      /* --queue-size: requests that may wait for the slot (0 = 429) */
    int queue_timeout_s; /* --queue-timeout: seconds a queued request waits */
    int rate_limit; /* --rate-limit: requests/second allowed on non-generation routes (0 = off) */
    int client_timeout_s; /* --client-timeout: socket read/write timeout in seconds (0 = none) */
    int max_connections;  /* --max-connections: simultaneous client connections (0 = unlimited) */
    int max_prompt_bytes; /* --max-prompt-bytes: message text a request may carry (0 = unlimited) */
    int parallel;         /* --parallel: chat requests served at once (1 = sequential) */
    int parallel_batch;   /* --parallel-batch: rows per batched forward pass */
} ServerLimits;

/* Chat template handling and config writes. */
typedef struct {
    int chat_format; /* --chat-format: -1 = auto (from vocabulary), else ChatTemplateType;
                                     only used when the model has no embedded chat template */
    int allow_request_template; /* requests may send "chat_template" (1; --no-request-template = 0) */
    int allow_config_write;     /* POST /v1/config/defaults may save (1; --no-config-write = 0) */
    const char
        *chat_template_source; /* "gguf" | "file" | "builtin" | "none" — published by /v1/config */
} ServerChat;

/* Activation numerics and the accuracy probe published by /v1/config. */
typedef struct {
    int act_quant;            /* --act-quant as requested: 0 auto, 1 int8, 2 float */
    double quality_tolerance; /* --quality-tolerance: allowed perplexity increase, percent */
    int self_check_tokens;    /* --self-check-tokens: length of POST /v1/selfcheck's probe */
} ServerNumerics;

/* Candidate search: best-of and tree search (MCTS) defaults and caps. */
typedef struct {
    int best_of;           /* --best-of: default candidates per request (1 = off) */
    int best_of_max;       /* --best-of-max: per-request cap */
    MctsOptions mcts;      /* Phase 30: --mcts* defaults every request starts from */
    int mcts_max;          /* --mcts-max: per-request simulation cap (0 = search disabled) */
    const char *prm_model; /* basename of --prm, NULL when steps are self-scored */
} ServerSearch;

/* Server knobs, grouped by topic (at most 25 fields per struct). */
typedef struct {
    CorsConfig cors;
    AuthConfig auth;
    MetricsConfig metrics;
    ServerUi ui;
    ServerVision vision;
    ServerCache cache;
    ServerLimits limits;
    ServerChat chat;
    ServerNumerics numerics;
    ServerSearch search;
    /* Generation defaults every request starts from (CLI generation flags, see
     * args_fill_gen_params), published to clients via GET /v1/config so the web UI renders its
     * controls from the server's real values. Pointer fields (system_prompt, stop) borrow argv. */
    GenerationParams gen_defaults;
    char *model_id;          /* reported by /v1/models; main.c sets the model file's basename */
    PzProvenance provenance; /* where each setting came from, published by /v1/config */
} ServerConfig;

/* Zero-initialise. Must be called before use. */
void server_config_init(ServerConfig *sc);

/* Adds an allowed CORS origin (or "*"). Returns 0 on success, -1 if the
 * origin list is full or origin is NULL. */
int server_config_add_cors_origin(ServerConfig *sc, const char *origin);

/* Frees heap-allocated fields (origins, api_key, static_dir). */
void server_config_free(ServerConfig *sc);

#endif /* TN_SERVER_CONFIG_H */
