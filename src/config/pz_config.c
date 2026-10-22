/*
 * config/pz_config.c — see config/pz_config.h.
 */
#include "config/pz_config.h"
#include "core/platform.h"
#include "reasoning/best_of.h"     /* BEST_OF_N_MAX, shared with parse_args */
#include "api/server_config.h"   /* PZ_PARALLEL_* bounds shared with parse_args */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if TN_POSIX
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ── schema ───────────────────────────────────────────────────────────────
 * Settings worth persisting; each key is the CLI flag without "--", so the
 * value is validated again by parse_args at startup. Per-run inputs
 * (--prompt, --image, --perplexity ...) are deliberately absent. Keep this
 * table in sync with args.c (engineering-rules DoD #6). */
static const PzCfgKey g_schema[] = {
    /* model */
    { "model",           PZ_CFG_STRING, "model", "Default model file (.gguf)", NULL, 0, 0, 0 },
    { "tokenizer",       PZ_CFG_STRING, "model", "Tokenizer file for non-GGUF models", NULL, 0, 0, 0 },
    { "chat-template-file", PZ_CFG_STRING, "model", "Jinja chat template that replaces the model's own", NULL, 0, 0, 0 },
    { "chat-format",     PZ_CFG_STRING, "model", "Prompt format for models without a chat template", NULL, 0, 0, 0 },
    { "lora",            PZ_CFG_STRING, "model", "LoRA adapter to apply", NULL, 0, 0, 0 },
    { "draft-model",     PZ_CFG_STRING, "model", "Draft model for speculative decoding", NULL, 0, 0, 0 },
    { "spec-length",     PZ_CFG_INT,    "model", "Draft tokens per speculative step", NULL, 1, 64, 0 },
    /* hardware */
    { "threads",         PZ_CFG_INT,    "hardware", "Worker threads (0 = auto)", NULL, 0, 1024, 0 },
    { "simd",            PZ_CFG_STRING, "hardware", "SIMD backend", "auto|avx2|avx512f|vnni|scalar", 0, 0, 0 },
    { "classifier",      PZ_CFG_STRING, "hardware", "LM-head precision", "auto|bf16|int8|int4|auto-fast", 0, 0, 0 },
    /* memory / context */
    { "ctx-size",        PZ_CFG_INT,    "memory", "Context length in tokens (0 = sized from RAM)", NULL, 0, 16777216, 0 },
    { "kv-type",         PZ_CFG_STRING, "memory", "KV cache storage", "f32|i8|i4", 0, 0, 0 },
    { "kv-ram-fraction", PZ_CFG_FLOAT,  "memory", "Share of free RAM the KV cache may use (0 = default)", NULL, 0, 1, 0 },
    { "kv-offload",      PZ_CFG_STRING, "memory", "Directory for a disk-backed KV cache", NULL, 0, 0, 0 },
    { "prompt-cache-mb", PZ_CFG_INT,    "memory", "Prompt snapshot cache budget in MiB", NULL, 0, 1048576, 0 },
    /* generation defaults */
    { "max-tokens",      PZ_CFG_INT,    "generation", "Default tokens to generate", NULL, 1, 16777216, 0 },
    { "temperature",     PZ_CFG_FLOAT,  "generation", "Sampling temperature", NULL, 0, 10, 0 },
    { "top-p",           PZ_CFG_FLOAT,  "generation", "Nucleus sampling threshold", NULL, 0, 1, 0 },
    { "top-k",           PZ_CFG_INT,    "generation", "Top-k sampling (0 = off)", NULL, 0, 1000000, 0 },
    { "min-p",           PZ_CFG_FLOAT,  "generation", "Min-p sampling threshold", NULL, 0, 1, 0 },
    { "repeat-penalty",  PZ_CFG_FLOAT,  "generation", "Repetition penalty (1 = off)", NULL, 0, 10, 0 },
    { "presence-penalty", PZ_CFG_FLOAT, "generation", "Presence penalty", NULL, -10, 10, 0 },
    { "frequency-penalty", PZ_CFG_FLOAT,"generation", "Frequency penalty", NULL, -10, 10, 0 },
    { "repeat-last-n",   PZ_CFG_INT,    "generation", "Tokens considered by the penalties (-1 = all)", NULL, -1, 1048576, 0 },
    { "seed",            PZ_CFG_INT,    "generation", "Sampling seed (-1 = random)", NULL, -1, 9.2e18, 0 },
    { "system-prompt",   PZ_CFG_STRING, "generation", "Default system prompt", NULL, 0, 0, 0 },
    { "stop",            PZ_CFG_LIST,   "generation", "Stop sequences", NULL, 0, 0, 0 },
    { "color",           PZ_CFG_STRING, "generation", "Terminal colour", "auto|always|never", 0, 0, 0 },
    /* server */
    { "port",            PZ_CFG_INT,    "server", "HTTP port for --server", NULL, 1, 65535, 0 },
    { "web-ui",          PZ_CFG_STRING, "server", "Web chat UI", "auto|on|off", 0, 0, 0 },
    { "ui-title",        PZ_CFG_STRING, "server", "Web UI title (also renders the wordmark)", NULL, 0, 0, 0 },
    { "cors",            PZ_CFG_BOOL,   "server", "Allow cross-origin requests", NULL, 0, 0, 0 },
    { "cors-origin",     PZ_CFG_LIST,   "server", "Allowed CORS origins", NULL, 0, 0, 0 },
    { "api-key",         PZ_CFG_STRING, "server", "Require this bearer token", NULL, 0, 0, 1 },
    { "metrics",         PZ_CFG_BOOL,   "server", "Expose /metrics", NULL, 0, 0, 0 },
    { "queue-size",      PZ_CFG_INT,    "server", "Requests that may wait for the generation slot", NULL, 0, 1048576, 0 },
    { "queue-timeout",   PZ_CFG_INT,    "server", "Seconds a queued request waits", NULL, 0, 86400, 0 },
    { "parallel",        PZ_CFG_INT,    "server", "Chat requests served at once (1 = one at a time)", NULL, 1, PZ_PARALLEL_MAX, 0 },
    { "parallel-batch",  PZ_CFG_INT,    "server", "Tokens per batched forward pass with --parallel", NULL, 1, PZ_PARALLEL_BATCH_MAX, 0 },
    { "max-tokens-limit", PZ_CFG_INT,   "server", "Per-request max_tokens cap (0 = context)", NULL, 0, 16777216, 0 },
    { "best-of-max",     PZ_CFG_INT,    "server", "Largest best_of a request may ask for", NULL, 1, BEST_OF_N_MAX, 0 },
    { "no-config-write", PZ_CFG_BOOL,   "server", "Web UI may not save server defaults to this file", NULL, 0, 0, 0 },
    { "mcts-max",        PZ_CFG_INT,    "server", "Largest tree-search budget a request may ask for", NULL, 0, 1048576, 0 },
    /* updates */
    { "no-update-check", PZ_CFG_BOOL,   "updates", "Never check for new releases", NULL, 0, 0, 0 },
    { "update-channel",  PZ_CFG_STRING, "updates", "Release channel", "auto|stable|prerelease", 0, 0, 0 },
    { "update-interval", PZ_CFG_INT,    "updates", "Hours between update checks (0 = off)", NULL, 0, 8760, 0 },
};

#define N_SCHEMA ((int)(sizeof(g_schema) / sizeof(g_schema[0])))

const PzCfgKey *pz_config_schema(int *n) { if (n) *n = N_SCHEMA; return g_schema; }

const PzCfgKey *pz_config_find_key(const char *key) {
    if (!key) return NULL;
    for (int i = 0; i < N_SCHEMA; i++)
        if (strcmp(g_schema[i].key, key) == 0) return &g_schema[i];
    return NULL;
}

int pz_config_path(char *buf, size_t n) {
    const char *d;
    int w;
    if ((d = getenv("PZ_CONFIG")) && *d) w = snprintf(buf, n, "%s", d);
    else if ((d = getenv("XDG_CONFIG_HOME")) && *d) w = snprintf(buf, n, "%s/project-zero/config.toml", d);
    else if ((d = getenv("HOME")) && *d) w = snprintf(buf, n, "%s/.config/project-zero/config.toml", d);
    else return -1;
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

/* ── entries ──────────────────────────────────────────────────────────── */

static void entry_clear(PzCfgEntry *e) {
    for (int i = 0; i < e->n; i++) { free(e->v[i]); e->v[i] = NULL; }
    e->n = 0;
    e->set = 0;
    e->src = PZ_SRC_DEFAULT;
}

int pz_config_init(PzConfig *c) {
    if (!c) return -1;
    c->entries = (PzCfgEntry *)calloc((size_t)N_SCHEMA, sizeof(PzCfgEntry));
    if (!c->entries) { c->n = 0; return -1; }
    c->n = N_SCHEMA;
    for (int i = 0; i < N_SCHEMA; i++) c->entries[i].key = &g_schema[i];
    return 0;
}

void pz_config_free(PzConfig *c) {
    if (!c || !c->entries) return;
    for (int i = 0; i < c->n; i++) entry_clear(&c->entries[i]);
    free(c->entries);
    c->entries = NULL;
    c->n = 0;
}

static PzCfgEntry *entry_for(PzConfig *c, const char *key) {
    if (!c || !c->entries) return NULL;
    for (int i = 0; i < c->n; i++)
        if (strcmp(c->entries[i].key->key, key) == 0) return &c->entries[i];
    return NULL;
}

const PzCfgEntry *pz_config_get(const PzConfig *c, const char *key) {
    PzCfgEntry *e = entry_for((PzConfig *)c, key);
    return (e && e->set) ? e : NULL;
}

int pz_config_validate(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    if (!k || !value) { snprintf(err, errn, "missing value"); return -1; }
    if (strlen(value) >= PZ_CFG_MAX_VALUE) { snprintf(err, errn, "%s: value too long", k->key); return -1; }
    if (strchr(value, '\n') && k->type != PZ_CFG_STRING && k->type != PZ_CFG_LIST) {
        snprintf(err, errn, "%s: value may not contain a newline", k->key);
        return -1;
    }
    if (k->type == PZ_CFG_INT || k->type == PZ_CFG_FLOAT) {
        char *end = NULL;
        errno = 0;
        double d = k->type == PZ_CFG_INT ? (double)strtoll(value, &end, 10) : strtod(value, &end);
        if (end == value || *end != '\0' || errno == ERANGE || !isfinite(d)) {
            snprintf(err, errn, "%s: expected %s, got \"%s\"", k->key,
                     k->type == PZ_CFG_INT ? "an integer" : "a number", value);
            return -1;
        }
        if (k->min < k->max && (d < k->min || d > k->max)) {
            snprintf(err, errn, "%s: %s is outside %g..%g", k->key, value, k->min, k->max);
            return -1;
        }
    } else if (k->type == PZ_CFG_BOOL) {
        if (strcmp(value, "true") != 0 && strcmp(value, "false") != 0) {
            snprintf(err, errn, "%s: expected true or false", k->key);
            return -1;
        }
    }
    if (k->choices) {
        size_t vl = strlen(value);
        for (const char *p = k->choices; *p;) {
            const char *bar = strchr(p, '|');
            size_t l = bar ? (size_t)(bar - p) : strlen(p);
            if (l == vl && strncmp(p, value, l) == 0) return 0;
            if (!bar) break;
            p = bar + 1;
        }
        snprintf(err, errn, "%s: \"%s\" is not one of %s", k->key, value, k->choices);
        return -1;
    }
    return 0;
}

static int store(PzConfig *c, const char *key, const char *value, int append, char *err, size_t errn) {
    const PzCfgKey *k = pz_config_find_key(key);
    if (!k) { snprintf(err, errn, "unknown setting \"%s\" (see `pz config list --all`)", key); return -1; }
    if (pz_config_validate(k, value, err, errn) != 0) return -1;
    PzCfgEntry *e = entry_for(c, key);
    if (!e) { snprintf(err, errn, "config not initialised"); return -1; }
    if (!(append && k->type == PZ_CFG_LIST)) entry_clear(e);
    if (e->n >= PZ_CFG_MAX_ITEMS) { snprintf(err, errn, "%s: at most %d items", key, PZ_CFG_MAX_ITEMS); return -1; }
    char *dup = (char *)malloc(strlen(value) + 1);
    if (!dup) { snprintf(err, errn, "out of memory"); return -1; }
    strcpy(dup, value);
    e->v[e->n++] = dup;
    e->set = 1;
    e->src = PZ_SRC_FILE;   /* set/add/parse hold file values; apply_env re-marks */
    return 0;
}

int pz_config_set(PzConfig *c, const char *key, const char *value, char *err, size_t errn) {
    return store(c, key, value, 0, err, errn);
}

int pz_config_add(PzConfig *c, const char *key, const char *value, char *err, size_t errn) {
    return store(c, key, value, 1, err, errn);
}

int pz_config_unset(PzConfig *c, const char *key) {
    PzCfgEntry *e = entry_for(c, key);
    if (!e) return -1;
    entry_clear(e);
    return 0;
}

/* ── parsing ──────────────────────────────────────────────────────────── */

static void warnf(void (*warn)(const char *), int line, const char *what) {
    if (!warn) return;
    char m[320];
    snprintf(m, sizeof(m), "config line %d: %s", line, what);
    warn(m);
}

/* Parses a TOML basic string at *p (after the opening quote) into out. */
static int parse_string(const char **pp, char *out, size_t outn) {
    const char *p = *pp;
    size_t o = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\') {
            char e = *p++;
            switch (e) {
            case 'n': ch = '\n'; break;
            case 't': ch = '\t'; break;
            case '"': ch = '"'; break;
            case '\\': ch = '\\'; break;
            default: return -1;
            }
        }
        if (o + 1 >= outn) return -1;
        out[o++] = ch;
    }
    if (*p != '"') return -1;
    out[o] = '\0';
    *pp = p + 1;
    return 0;
}

/* Parses one scalar (string, number, bool) at *pp into out. */
static int parse_scalar(const char **pp, char *out, size_t outn) {
    const char *p = *pp;
    if (*p == '"') {
        p++;
        if (parse_string(&p, out, outn) != 0) return -1;
    } else {
        size_t o = 0;
        while (*p && *p != ',' && *p != ']' && *p != '#' && !isspace((unsigned char)*p)) {
            if (o + 1 >= outn) return -1;
            out[o++] = *p++;
        }
        if (o == 0) return -1;
        out[o] = '\0';
    }
    *pp = p;
    return 0;
}

static const char *skip_ws(const char *p) { while (*p == ' ' || *p == '\t') p++; return p; }

int pz_config_parse(PzConfig *c, const char *text, void (*warn)(const char *)) {
    if (!c || !text) return 0;
    int loaded = 0, lineno = 0;
    char *val = (char *)malloc(PZ_CFG_MAX_VALUE);
    if (!val) return 0;
    const char *p = text;
    while (*p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        char *line = (char *)malloc(len + 1);
        if (!line) break;
        memcpy(line, p, len);
        line[len] = '\0';
        if (len && line[len - 1] == '\r') line[len - 1] = '\0';
        p = eol ? eol + 1 : p + len;
        lineno++;

        const char *q = skip_ws(line);
        if (*q == '\0' || *q == '#') { free(line); continue; }
        if (*q == '[') { warnf(warn, lineno, "tables are not supported; keys are flat"); free(line); continue; }
        char key[64];
        size_t kl = 0;
        while (*q && (isalnum((unsigned char)*q) || *q == '-' || *q == '_') && kl + 1 < sizeof(key))
            key[kl++] = *q++;
        key[kl] = '\0';
        for (size_t i = 0; i < kl; i++) if (key[i] == '_') key[i] = '-';
        q = skip_ws(q);
        if (kl == 0 || *q != '=') { warnf(warn, lineno, "expected key = value"); free(line); continue; }
        q = skip_ws(q + 1);
        const PzCfgKey *k = pz_config_find_key(key);
        if (!k) {
            char m[128];
            snprintf(m, sizeof(m), "unknown setting \"%s\" ignored", key);
            warnf(warn, lineno, m);
            free(line);
            continue;
        }
        char err[256];
        int ok = 1;
        if (*q == '[') {
            if (k->type != PZ_CFG_LIST) { warnf(warn, lineno, "a list is only valid for list settings"); free(line); continue; }
            pz_config_unset(c, key);
            q = skip_ws(q + 1);
            while (ok && *q && *q != ']') {
                if (parse_scalar(&q, val, PZ_CFG_MAX_VALUE) != 0 || pz_config_add(c, key, val, err, sizeof(err)) != 0) ok = 0;
                q = skip_ws(q);
                if (*q == ',') q = skip_ws(q + 1);
            }
            if (ok && *q != ']') ok = 0;
            if (ok) q = skip_ws(q + 1);
        } else {
            if (parse_scalar(&q, val, PZ_CFG_MAX_VALUE) != 0) ok = 0;
            else {
                q = skip_ws(q);
                if (k->type == PZ_CFG_LIST) ok = pz_config_set(c, key, val, err, sizeof(err)) == 0;
                else ok = pz_config_set(c, key, val, err, sizeof(err)) == 0;
                if (!ok) { warnf(warn, lineno, err); free(line); continue; }
            }
        }
        if (ok && *q != '\0' && *q != '#') ok = 0;
        if (!ok) { warnf(warn, lineno, "malformed value"); pz_config_unset(c, key); }
        else loaded++;
        free(line);
    }
    free(val);
    return loaded;
}

int pz_config_load(PzConfig *c, const char *path, void (*warn)(const char *)) {
    if (!c || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return errno == ENOENT ? 0 : -1;
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { fclose(f); return -1; }
    size_t r;
    while ((r = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += r;
        if (len + 1 >= cap) {
            if (cap >= ((size_t)4 << 20)) { free(buf); fclose(f); return -1; } /* 4 MiB sanity cap */
            char *nb = (char *)realloc(buf, cap * 2);
            if (!nb) { free(buf); fclose(f); return -1; }
            buf = nb;
            cap *= 2;
        }
    }
    fclose(f);
    buf[len] = '\0';
    int n = pz_config_parse(c, buf, warn);
    free(buf);
    return n;
}

/* ── output ───────────────────────────────────────────────────────────── */

static int append(char **buf, size_t *len, size_t *cap, const char *s) {
    size_t l = strlen(s);
    if (*len + l + 1 > *cap) {
        size_t nc = (*cap ? *cap : 1024);
        while (*len + l + 1 > nc) nc *= 2;
        char *nb = (char *)realloc(*buf, nc);
        if (!nb) return -1;
        *buf = nb;
        *cap = nc;
    }
    memcpy(*buf + *len, s, l + 1);
    *len += l;
    return 0;
}

static int append_quoted(char **buf, size_t *len, size_t *cap, const char *v) {
    if (append(buf, len, cap, "\"")) return -1;
    char t[3] = { 0, 0, 0 };
    for (const char *p = v; *p; p++) {
        const char *s = NULL;
        if (*p == '"') s = "\\\"";
        else if (*p == '\\') s = "\\\\";
        else if (*p == '\n') s = "\\n";
        else if (*p == '\t') s = "\\t";
        else { t[0] = *p; t[1] = 0; s = t; }
        if (append(buf, len, cap, s)) return -1;
    }
    return append(buf, len, cap, "\"");
}

char *pz_config_format(const PzConfig *c) {
    char *buf = NULL;
    size_t len = 0, cap = 0;
    if (append(&buf, &len, &cap, "# Project Zero configuration — edit with `pz config` or by hand.\n"
                                  "# Precedence: command-line flag > PZ_* environment > this file > default.\n")) {
        free(buf);
        return NULL;
    }
    const char *group = NULL;
    for (int i = 0; c && i < c->n; i++) {
        const PzCfgEntry *e = &c->entries[i];
        if (!e->set) continue;
        if (!group || strcmp(group, e->key->group) != 0) {
            group = e->key->group;
            if (append(&buf, &len, &cap, "\n# ") || append(&buf, &len, &cap, group) || append(&buf, &len, &cap, "\n")) goto fail;
        }
        if (append(&buf, &len, &cap, e->key->key) || append(&buf, &len, &cap, " = ")) goto fail;
        PzCfgType t = e->key->type;
        if (t == PZ_CFG_LIST) {
            if (append(&buf, &len, &cap, "[")) goto fail;
            for (int j = 0; j < e->n; j++) {
                if (j && append(&buf, &len, &cap, ", ")) goto fail;
                if (append_quoted(&buf, &len, &cap, e->v[j])) goto fail;
            }
            if (append(&buf, &len, &cap, "]")) goto fail;
        } else if (t == PZ_CFG_STRING) {
            if (append_quoted(&buf, &len, &cap, e->v[0])) goto fail;
        } else if (append(&buf, &len, &cap, e->v[0])) goto fail;
        if (append(&buf, &len, &cap, "\n")) goto fail;
    }
    return buf;
fail:
    free(buf);
    return NULL;
}

#if TN_POSIX
static void mkdir_parents(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        mkdir(tmp, 0700);
        *p = '/';
    }
}
#endif

int pz_config_save(const PzConfig *c, const char *path) {
    if (!c || !path) return -1;
    char *text = pz_config_format(c);
    if (!text) return -1;
#if TN_POSIX
    mkdir_parents(path);
#endif
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { free(text); return -1; }
#if TN_POSIX
    chmod(tmp, 0600);   /* may hold an API key */
#endif
    int ok = fputs(text, f) >= 0;
    ok = (fclose(f) == 0) && ok;
    free(text);
    if (!ok || rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

/* ── environment + argv ───────────────────────────────────────────────── */

static void env_name(const char *key, char *out, size_t n) {
    size_t o = snprintf(out, n, "PZ_");
    for (const char *p = key; *p && o + 1 < n; p++)
        out[o++] = (*p == '-') ? '_' : (char)toupper((unsigned char)*p);
    out[o < n ? o : n - 1] = '\0';
}

int pz_config_apply_env(PzConfig *c, void (*warn)(const char *)) {
    int applied = 0;
    for (int i = 0; c && i < c->n; i++) {
        char name[96];
        env_name(c->entries[i].key->key, name, sizeof(name));
        const char *v = getenv(name);
        if (!v || !*v) continue;
        char err[256];
        if (pz_config_set(c, c->entries[i].key->key, v, err, sizeof(err)) == 0) {
            c->entries[i].src = PZ_SRC_ENV;
            applied++;
        }
        else if (warn) {
            char m[400];
            snprintf(m, sizeof(m), "%s ignored: %s", name, err);
            warn(m);
        }
    }
    return applied;
}

int pz_config_merge_argv(const PzConfig *c, int argc, char **argv, char ***out_argv) {
    if (!out_argv || argc < 1 || !argv) return -1;
    /* Flag names live in a static table of "--key" strings built once. */
    static char flags[sizeof(g_schema) / sizeof(g_schema[0])][72];
    int extra = 0;
    for (int i = 0; c && i < c->n; i++) {
        const PzCfgEntry *e = &c->entries[i];
        if (!e->set) continue;
        if (e->key->type == PZ_CFG_BOOL) extra += strcmp(e->v[0], "true") == 0;
        else extra += 2 * e->n;
    }
    char **av = (char **)malloc(((size_t)argc + (size_t)extra + 1) * sizeof(char *));
    if (!av) return -1;
    int k = 0;
    av[k++] = argv[0];
    for (int i = 0; c && i < c->n; i++) {
        const PzCfgEntry *e = &c->entries[i];
        if (!e->set) continue;
        snprintf(flags[i], sizeof(flags[i]), "--%s", e->key->key);
        if (e->key->type == PZ_CFG_BOOL) {
            if (strcmp(e->v[0], "true") == 0) av[k++] = flags[i];
            continue;
        }
        for (int j = 0; j < e->n; j++) { av[k++] = flags[i]; av[k++] = e->v[j]; }
    }
    for (int i = 1; i < argc; i++) av[k++] = argv[i];
    av[k] = NULL;
    *out_argv = av;
    return k;
}

/* ── provenance (Phase 40.3) ──────────────────────────────────────────── */

const char *pz_config_source_name(int src) {
    switch (src) {
    case PZ_SRC_FILE: return "file";
    case PZ_SRC_ENV:  return "env";
    case PZ_SRC_CLI:  return "cli";
    default:          return "default";
    }
}

int pz_config_mark_cli(PzConfig *c, int argc, char **argv) {
    int marked = 0;
    for (int i = 1; c && argv && i < argc; i++) {
        /* parse_args only takes the "--key value" form. */
        if (!argv[i] || strncmp(argv[i], "--", 2) != 0) continue;
        PzCfgEntry *e = entry_for(c, argv[i] + 2);
        if (e && e->src != PZ_SRC_CLI) { e->src = PZ_SRC_CLI; marked++; }
    }
    return marked;
}

static const PzConfig *g_active;
static char g_active_path[1024];
static int g_active_file_used;

void pz_config_publish(const PzConfig *c, const char *path, int file_used) {
    g_active = c;
    g_active_file_used = c ? file_used : 0;
    snprintf(g_active_path, sizeof(g_active_path), "%s", (c && path) ? path : "");
}

const char *pz_config_active_source(const char *key) {
    /* entry_for, not pz_config_get: a key given only as a flag is unset here. */
    const PzCfgEntry *e = (g_active && key) ? entry_for((PzConfig *)g_active, key) : NULL;
    return pz_config_source_name(e ? e->src : PZ_SRC_DEFAULT);
}

const char *pz_config_active_path(void) {
    return g_active_path[0] ? g_active_path : NULL;
}

int pz_config_active_file_used(void) { return g_active_file_used; }
