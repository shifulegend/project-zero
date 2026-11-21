/*
 * api/config_endpoint.c — see api/config_endpoint.h.
 */
#include "api/config_endpoint.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "api/cors.h"
#include "api/http_status.h"
#include "config/pz_config.h"
#include "core/json_fields.h"
#include "transformer/gen_params.h"

/* /v1/config JSON names ↔ config keys (the CLI flag names). */
static const struct {
    const char *json, *key;
} g_map[] = {
    {"max_tokens", "max-tokens"},
    {"temperature", "temperature"},
    {"top_p", "top-p"},
    {"top_k", "top-k"},
    {"min_p", "min-p"},
    {"repetition_penalty", "repeat-penalty"},
    {"presence_penalty", "presence-penalty"},
    {"frequency_penalty", "frequency-penalty"},
    {"penalty_last_n", "repeat-last-n"},
    {"seed", "seed"},
    {"system_prompt", "system-prompt"},
    {"stop", "stop"},
    {"ui_title", "ui-title"},
    {"parallel", "parallel"},
    {"parallel_batch", "parallel-batch"},
    {"act_quant", "act-quant"},
    {"quality_tolerance", "quality-tolerance"},
    {"self_check_tokens", "self-check-tokens"},
};
#define N_MAP ((int)(sizeof(g_map) / sizeof(g_map[0])))

static const char *config_key_for(const char *json) {
    for (int i = 0; json && i < N_MAP; i++)
        if (strcmp(g_map[i].json, json) == 0) return g_map[i].key;
    return NULL;
}

static int writable(const ApiContext *ctx) {
    return ctx && ctx->server_config.chat.allow_config_write &&
           ctx->server_config.provenance.path != NULL;
}

void api_config_sources_json(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "config");
    jb_raw(jb, "{");
    jb_key(jb, "path");
    jb_str(jb, ctx ? ctx->server_config.provenance.path : NULL);
    jb_raw(jb, ",");
    jb_key(jb, "file_used");
    jb_bool(jb, ctx && ctx->server_config.provenance.file_used);
    jb_raw(jb, ",");
    jb_key(jb, "writable");
    jb_bool(jb, writable(ctx));
    jb_raw(jb, ",");
    jb_key(jb, "keys");
    jb_raw(jb, "{");
    for (int i = 0; i < N_MAP; i++) {
        if (i) jb_raw(jb, ",");
        jb_key(jb, g_map[i].json);
        jb_str(jb, g_map[i].key);
    }
    jb_raw(jb, "},");
    jb_key(jb, "sources");
    jb_raw(jb, "{");
    for (int i = 0; i < N_MAP; i++) {
        if (i) jb_raw(jb, ",");
        jb_key(jb, g_map[i].json);
        jb_str(jb, pz_provenance_source(ctx ? &ctx->server_config.provenance : NULL, g_map[i].key));
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

int api_config_write_allowed(const char *content_type, const char *origin, const char *host,
                             const CorsConfig *cors) {
    if (!content_type || strncasecmp(content_type, "application/json", 16) != 0) return 0;
    if (!origin || !*origin) return 1; /* not a browser cross-site request */
    if (origin_matches_host(origin, host)) return 1;
    return cors && cors->enabled && cors_is_origin_allowed(cors, origin);
}

int api_post_allowed(const char *content_type, const char *origin, const char *host,
                     const CorsConfig *cors, size_t body_len) {
    if (!origin || !*origin) return 1;               /* not a browser request */
    if (origin_matches_host(origin, host)) return 1; /* the server's own origin */
    if (!(cors && cors->enabled && cors_is_origin_allowed(cors, origin))) return 0;
    return body_len == 0 ||
           (content_type && strncasecmp(content_type, "application/json", 16) == 0);
}

/* Host header without its port: "localhost:8080" -> "localhost", "[::1]:80" -> "[::1]". */
static size_t host_name_len(const char *host) {
    if (host[0] == '[') {
        const char *e = strchr(host, ']');
        return e ? (size_t)(e - host) + 1 : strlen(host);
    }
    const char *c = strchr(host, ':');
    return c ? (size_t)(c - host) : strlen(host);
}

int api_host_allowed(const char *host, const CorsConfig *cors) {
    if (!host || !*host) return 1;
    size_t n                        = host_name_len(host);
    static const char *const loop[] = {"localhost", "127.0.0.1", "[::1]"};
    for (size_t i = 0; i < sizeof loop / sizeof loop[0]; i++)
        if (strlen(loop[i]) == n && strncasecmp(host, loop[i], n) == 0) return 1;
    if (cors && cors->enabled) {
        size_t hl = strlen(host);
        for (int i = 0; i < cors->num_origins; i++) {
            const char *o = cors->origins[i];
            if (strcmp(o, "*") == 0) return 1;
            const char *a = strstr(o, "://");
            if (!a) continue;
            a += 3;
            size_t al = strlen(a);
            while (al && a[al - 1] == '/') al--;
            if (al == hl && strncasecmp(a, host, hl) == 0) return 1; /* host:port */
            if (al == n && strncasecmp(a, host, n) == 0 && host[n] == ':')
                return 1; /* origin without port */
        }
    }
    return 0;
}

static char *error_body(const char *msg, int status, int *out_status) {
    *out_status = status;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"error\":{\"message\":");
    jb_str(&jb, msg);
    jb_raw(&jb, ",\"type\":");
    jb_str(&jb, status == HTTP_FORBIDDEN        ? "permission_error"
                : status >= HTTP_INTERNAL_ERROR ? "server_error"
                                                : "invalid_request_error");
    jb_raw(&jb, "}}");
    return jb_finish(&jb);
}

/* A parsed save request: the JSON name and its config key, and the value as text the schema
 * validates (and parse_args re-reads): one scalar, a list of items, or nothing (an unset). */
typedef struct {
    char *name;
    const char *key;
    int unset;
    char *items[GEN_MAX_STOP];
    int n_items;
    char *scalar;
} SaveRequest;

static void save_request_free(SaveRequest *r) {
    free(r->name);
    free(r->scalar);
    for (int i = 0; i < r->n_items; i++) free(r->items[i]);
}

/* The value part of a save request: a list for "stop", a string or number otherwise. Returns the
 * error body for a bad value, NULL when it parsed (or none is needed for an unset). */
static char *parse_save_value(const char *json_body, SaveRequest *r, int *status) {
    if (r->unset) return NULL;
    if (strcmp(r->key, "stop") == 0) {
        r->n_items = json_field_string_list(json_body, "value", r->items, GEN_MAX_STOP);
        if (r->n_items < 0) {
            r->n_items = 0;
            return error_body("\"value\" must be an array of at most 8 strings", HTTP_BAD_REQUEST,
                              status);
        }
        return NULL;
    }
    r->scalar = json_field_string(json_body, "value");
    if (!r->scalar) r->scalar = json_field_number_text(json_body, "value");
    if (!r->scalar)
        return error_body("\"value\" (number or string) or \"unset\": true is required",
                          HTTP_BAD_REQUEST, status);
    return NULL;
}

/* Parses the body. Returns the error body (nothing left allocated) or NULL when *r is filled. */
static char *parse_save_request(const char *json_body, SaveRequest *r, int *status) {
    memset(r, 0, sizeof(*r));
    r->name = json_field_string(json_body, "key");
    r->key  = config_key_for(r->name);
    if (!r->key) {
        char m[160];
        snprintf(m, sizeof(m), "\"key\" must name a saveable default, got %s%s%s",
                 r->name ? "\"" : "", r->name ? r->name : "nothing", r->name ? "\"" : "");
        save_request_free(r);
        return error_body(m, HTTP_BAD_REQUEST, status);
    }
    json_field_bool(json_body, "unset", &r->unset);
    char *err = parse_save_value(json_body, r, status);
    if (err) save_request_free(r);
    return err;
}

/* Applies the requested change to the loaded config. Returns 0, or HTTP_BAD_REQUEST with the
 * schema's message in err. */
static int apply_change(PzConfig *c, const SaveRequest *r, int is_unset, char *err, size_t errn) {
    if (is_unset) {
        pz_config_unset(c, r->key);
        return 0;
    }
    if (r->scalar)
        return pz_config_set(c, r->key, r->scalar, err, errn) != 0 ? HTTP_BAD_REQUEST : 0;
    pz_config_unset(c, r->key);
    for (int i = 0; i < r->n_items; i++)
        if (pz_config_add(c, r->key, r->items[i], err, errn) != 0) return HTTP_BAD_REQUEST;
    return 0;
}

/* Loads the config file, applies the change and writes it back, under the save lock. Returns 0, or
 * the HTTP status of the failure with its text in err. */
static int apply_save(const ApiContext *ctx, const SaveRequest *r, int is_unset, char *err,
                      size_t errn) {
    int rc           = 0;
    const char *path = ctx->server_config.provenance.path;
    PzConfig c;
    pthread_mutex_lock((pthread_mutex_t *)&ctx->config_save_lock); /* a lock, not state */
    if (pz_config_init(&c) != 0 || pz_config_load(&c, path, NULL) < 0) {
        snprintf(err, errn, "cannot read %s", path);
        rc = HTTP_INTERNAL_ERROR;
    } else {
        rc = apply_change(&c, r, is_unset, err, errn);
    }
    if (!rc && pz_config_save(&c, path) != 0) {
        snprintf(err, errn, "cannot write %s", path);
        rc = HTTP_INTERNAL_ERROR;
    }
    pthread_mutex_unlock((pthread_mutex_t *)&ctx->config_save_lock);
    pz_config_free(&c);
    return rc;
}

/* The success body: what was saved, where, and whether a flag or environment variable overrides
 * it until restart. */
static char *save_response(const ApiContext *ctx, const SaveRequest *r, int is_unset) {
    const char *src = pz_provenance_source(&ctx->server_config.provenance, r->key);
    int overridden  = strcmp(src, "cli") == 0 || strcmp(src, "env") == 0;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{");
    jb_key(&jb, "saved");
    jb_bool(&jb, 1);
    jb_raw(&jb, ",");
    jb_key(&jb, "key");
    jb_str(&jb, r->name);
    jb_raw(&jb, ",");
    jb_key(&jb, "config_key");
    jb_str(&jb, r->key);
    jb_raw(&jb, ",");
    jb_key(&jb, "unset");
    jb_bool(&jb, is_unset);
    jb_raw(&jb, ",");
    jb_key(&jb, "path");
    jb_str(&jb, ctx->server_config.provenance.path);
    jb_raw(&jb, ",");
    jb_key(&jb, "applies");
    jb_str(&jb, "restart");
    jb_raw(&jb, ",");
    jb_key(&jb, "overridden_by");
    jb_str(&jb, overridden ? src : NULL);
    jb_raw(&jb, ",");
    jb_key(&jb, "file_used");
    jb_bool(&jb, ctx->server_config.provenance.file_used);
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}

char *api_config_save(const ApiContext *ctx, const char *json_body, int *status) {
    if (!writable(ctx))
        return error_body(ctx && !ctx->server_config.chat.allow_config_write
                              ? "Saving defaults is disabled (--no-config-write)"
                              : "No config file location (set PZ_CONFIG or HOME)",
                          HTTP_FORBIDDEN, status);
    SaveRequest r;
    char *bad = parse_save_request(json_body, &r, status);
    if (bad) return bad;

    int is_unset  = r.unset || (r.n_items == 0 && !r.scalar);
    char err[256] = "";
    int rc        = apply_save(ctx, &r, is_unset, err, sizeof(err));
    char *answer;
    if (rc) {
        answer = error_body(err, rc, status);
    } else {
        answer  = save_response(ctx, &r, is_unset);
        *status = HTTP_OK;
    }
    save_request_free(&r);
    return answer;
}
