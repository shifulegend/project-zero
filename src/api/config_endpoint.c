/*
 * api/config_endpoint.c — see api/config_endpoint.h.
 */
#include "api/config_endpoint.h"
#include "api/cors.h"
#include "api/json_fields.h"
#include "config/pz_config.h"
#include "transformer/gen_params.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* /v1/config JSON names ↔ config keys (the CLI flag names). */
static const struct { const char *json, *key; } g_map[] = {
    { "max_tokens",         "max-tokens" },
    { "temperature",        "temperature" },
    { "top_p",              "top-p" },
    { "top_k",              "top-k" },
    { "min_p",              "min-p" },
    { "repetition_penalty", "repeat-penalty" },
    { "presence_penalty",   "presence-penalty" },
    { "frequency_penalty",  "frequency-penalty" },
    { "penalty_last_n",     "repeat-last-n" },
    { "seed",               "seed" },
    { "system_prompt",      "system-prompt" },
    { "stop",               "stop" },
    { "ui_title",           "ui-title" },
    { "parallel",           "parallel" },
    { "parallel_batch",     "parallel-batch" },
};
#define N_MAP ((int)(sizeof(g_map) / sizeof(g_map[0])))

static const char *config_key_for(const char *json) {
    for (int i = 0; json && i < N_MAP; i++)
        if (strcmp(g_map[i].json, json) == 0) return g_map[i].key;
    return NULL;
}

static int writable(const ApiContext *ctx) {
    return ctx && ctx->server_config.allow_config_write && pz_config_active_path() != NULL;
}

void api_config_sources_json(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "config"); jb_raw(jb, "{");
    jb_key(jb, "path"); jb_str(jb, pz_config_active_path());
    jb_raw(jb, ","); jb_key(jb, "file_used"); jb_bool(jb, pz_config_active_file_used());
    jb_raw(jb, ","); jb_key(jb, "writable"); jb_bool(jb, writable(ctx));
    jb_raw(jb, ","); jb_key(jb, "keys"); jb_raw(jb, "{");
    for (int i = 0; i < N_MAP; i++) {
        if (i) jb_raw(jb, ",");
        jb_key(jb, g_map[i].json); jb_str(jb, g_map[i].key);
    }
    jb_raw(jb, "},"); jb_key(jb, "sources"); jb_raw(jb, "{");
    for (int i = 0; i < N_MAP; i++) {
        if (i) jb_raw(jb, ",");
        jb_key(jb, g_map[i].json); jb_str(jb, pz_config_active_source(g_map[i].key));
    }
    jb_raw(jb, "}}");
}

/* "http://host[:port]" or "https://..." whose authority equals Host. */
static int origin_matches_host(const char *origin, const char *host) {
    if (!origin || !host || !*host) return 0;
    const char *a = strstr(origin, "://");
    if (!a) return 0;
    a += 3;
    size_t n = strlen(host);
    return strncasecmp(a, host, n) == 0 && (a[n] == '\0' || a[n] == '/');
}

int api_config_write_allowed(const char *content_type, const char *origin,
                             const char *host, const CorsConfig *cors) {
    if (!content_type || strncasecmp(content_type, "application/json", 16) != 0) return 0;
    if (!origin || !*origin) return 1;   /* not a browser cross-site request */
    if (origin_matches_host(origin, host)) return 1;
    return cors && cors->enabled && cors_is_origin_allowed(cors, origin);
}

static char *error_body(const char *msg, int status, int *out_status) {
    *out_status = status;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"error\":{\"message\":");
    jb_str(&jb, msg);
    jb_raw(&jb, ",\"type\":");
    jb_str(&jb, status == 403 ? "permission_error" : status >= 500 ? "server_error" : "invalid_request_error");
    jb_raw(&jb, "}}");
    return jb_finish(&jb);
}

/* Serialises load → change → save between concurrent requests. */
static pthread_mutex_t g_save_lock = PTHREAD_MUTEX_INITIALIZER;

char *api_config_save(const ApiContext *ctx, const char *json_body, int *status) {
    if (!writable(ctx))
        return error_body(ctx && !ctx->server_config.allow_config_write
                              ? "Saving defaults is disabled (--no-config-write)"
                              : "No config file location (set PZ_CONFIG or HOME)", 403, status);
    char *name = json_field_string(json_body, "key");
    const char *key = config_key_for(name);
    if (!key) {
        char m[160];
        snprintf(m, sizeof(m), "\"key\" must name a saveable default, got %s%s%s",
                 name ? "\"" : "", name ? name : "nothing", name ? "\"" : "");
        free(name);
        return error_body(m, 400, status);
    }
    int unset = 0;
    json_field_bool(json_body, "unset", &unset);

    /* The value, as text the schema validates (and parse_args re-reads). */
    char *items[GEN_MAX_STOP];
    int n_items = 0;
    char *scalar = NULL;
    if (!unset) {
        if (strcmp(key, "stop") == 0) {
            n_items = json_field_string_list(json_body, "value", items, GEN_MAX_STOP);
            if (n_items < 0) {
                free(name);
                return error_body("\"value\" must be an array of at most 8 strings", 400, status);
            }
        } else {
            scalar = json_field_string(json_body, "value");
            if (!scalar) scalar = json_field_number_text(json_body, "value");
            if (!scalar) {
                free(name);
                return error_body("\"value\" (number or string) or \"unset\": true is required", 400, status);
            }
        }
    }

    char err[256] = "";
    int rc = 0;
    const char *path = pz_config_active_path();
    PzConfig c;
    pthread_mutex_lock(&g_save_lock);
    if (pz_config_init(&c) != 0 || pz_config_load(&c, path, NULL) < 0) {
        snprintf(err, sizeof(err), "cannot read %s", path);
        rc = 500;
    } else if (unset || (n_items == 0 && !scalar)) {
        pz_config_unset(&c, key);
    } else if (scalar) {
        if (pz_config_set(&c, key, scalar, err, sizeof(err)) != 0) rc = 400;
    } else {
        pz_config_unset(&c, key);
        for (int i = 0; i < n_items && !rc; i++)
            if (pz_config_add(&c, key, items[i], err, sizeof(err)) != 0) rc = 400;
    }
    if (!rc && pz_config_save(&c, path) != 0) {
        snprintf(err, sizeof(err), "cannot write %s", path);
        rc = 500;
    }
    pthread_mutex_unlock(&g_save_lock);
    pz_config_free(&c);
    free(scalar);
    for (int i = 0; i < n_items; i++) free(items[i]);
    if (rc) { free(name); return error_body(err, rc, status); }

    const char *src = pz_config_active_source(key);
    int overridden = strcmp(src, "cli") == 0 || strcmp(src, "env") == 0;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{"); jb_key(&jb, "saved"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "key"); jb_str(&jb, name);
    jb_raw(&jb, ","); jb_key(&jb, "config_key"); jb_str(&jb, key);
    jb_raw(&jb, ","); jb_key(&jb, "unset"); jb_bool(&jb, unset || (n_items == 0 && !scalar));
    jb_raw(&jb, ","); jb_key(&jb, "path"); jb_str(&jb, path);
    jb_raw(&jb, ","); jb_key(&jb, "applies"); jb_str(&jb, "restart");
    jb_raw(&jb, ","); jb_key(&jb, "overridden_by"); jb_str(&jb, overridden ? src : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "file_used"); jb_bool(&jb, pz_config_active_file_used());
    jb_raw(&jb, "}");
    free(name);
    *status = 200;
    return jb_finish(&jb);
}
