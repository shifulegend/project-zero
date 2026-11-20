/*
 * config/pz_config.c — see config/pz_config.h.
 */
#include "config/pz_config.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/escape.h"
#include "core/fs_util.h"
#include "core/platform.h"
#if TN_POSIX
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "config/pz_config_schema.h"


const PzCfgKey *pz_config_schema(int *n) {
    if (n) *n = g_pz_cfg_schema_len;
    return g_pz_cfg_schema;
}

const PzCfgKey *pz_config_find_key(const char *key) {
    if (!key) return NULL;
    for (int i = 0; i < g_pz_cfg_schema_len; i++)
        if (strcmp(g_pz_cfg_schema[i].key, key) == 0) return &g_pz_cfg_schema[i];
    return NULL;
}

int pz_config_path(char *buf, size_t n) {
    const char *d;
    int w;
    if ((d = getenv("PZ_CONFIG")) && *d)
        w = snprintf(buf, n, "%s", d);
    else if ((d = getenv("XDG_CONFIG_HOME")) && *d)
        w = snprintf(buf, n, "%s/project-zero/config.toml", d);
    else if ((d = getenv("HOME")) && *d)
        w = snprintf(buf, n, "%s/.config/project-zero/config.toml", d);
    else
        return -1;
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

/* ── entries ──────────────────────────────────────────────────────────── */

static void entry_clear(PzCfgEntry *e) {
    for (int i = 0; i < e->n; i++) {
        free(e->v[i]);
        e->v[i] = NULL;
    }
    e->n   = 0;
    e->set = 0;
    e->src = PZ_SRC_DEFAULT;
}

int pz_config_init(PzConfig *c) {
    if (!c) return -1;
    c->entries = (PzCfgEntry *)calloc((size_t)g_pz_cfg_schema_len, sizeof(PzCfgEntry));
    if (!c->entries) {
        c->n = 0;
        return -1;
    }
    c->n = g_pz_cfg_schema_len;
    for (int i = 0; i < g_pz_cfg_schema_len; i++) c->entries[i].key = &g_pz_cfg_schema[i];
    return 0;
}

void pz_config_free(PzConfig *c) {
    if (!c || !c->entries) return;
    for (int i = 0; i < c->n; i++) entry_clear(&c->entries[i]);
    free(c->entries);
    c->entries = NULL;
    c->n       = 0;
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
    if (!k || !value) {
        snprintf(err, errn, "missing value");
        return -1;
    }
    if (strlen(value) >= PZ_CFG_MAX_VALUE) {
        snprintf(err, errn, "%s: value too long", k->key);
        return -1;
    }
    if (strchr(value, '\n') && k->type != PZ_CFG_STRING && k->type != PZ_CFG_LIST) {
        snprintf(err, errn, "%s: value may not contain a newline", k->key);
        return -1;
    }
    if (k->type == PZ_CFG_INT || k->type == PZ_CFG_FLOAT) {
        char *end = NULL;
        errno     = 0;
        double d  = k->type == PZ_CFG_INT ? (double)strtoll(value, &end, 10) : strtod(value, &end);
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
            size_t l        = bar ? (size_t)(bar - p) : strlen(p);
            if (l == vl && strncmp(p, value, l) == 0) return 0;
            if (!bar) break;
            p = bar + 1;
        }
        snprintf(err, errn, "%s: \"%s\" is not one of %s", k->key, value, k->choices);
        return -1;
    }
    return 0;
}

static int store(PzConfig *c, const char *key, const char *value, int append, char *err,
                 size_t errn) {
    const PzCfgKey *k = pz_config_find_key(key);
    if (!k) {
        snprintf(err, errn, "unknown setting \"%s\" (see `pz config list --all`)", key);
        return -1;
    }
    if (pz_config_validate(k, value, err, errn) != 0) return -1;
    PzCfgEntry *e = entry_for(c, key);
    if (!e) {
        snprintf(err, errn, "config not initialised");
        return -1;
    }
    if (!(append && k->type == PZ_CFG_LIST)) entry_clear(e);
    if (e->n >= PZ_CFG_MAX_ITEMS) {
        snprintf(err, errn, "%s: at most %d items", key, PZ_CFG_MAX_ITEMS);
        return -1;
    }
    char *dup = (char *)malloc(strlen(value) + 1);
    if (!dup) {
        snprintf(err, errn, "out of memory");
        return -1;
    }
    strcpy(dup, value);
    e->v[e->n++] = dup;
    e->set       = 1;
    e->src       = PZ_SRC_FILE; /* set/add/parse hold file values; apply_env re-marks */
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
    size_t o      = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\') {
            int d = tn_escape_simple(*p++);
            if (d < 0) return -1;
            ch = (char)d;
        }
        if (o + 1 >= outn) return -1;
        out[o++] = ch;
    }
    if (*p != '"') return -1;
    out[o] = '\0';
    *pp    = p + 1;
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

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

int pz_config_parse(PzConfig *c, const char *text, void (*warn)(const char *)) {
    if (!c || !text) return 0;
    int loaded = 0, lineno = 0;
    char *val = (char *)malloc(PZ_CFG_MAX_VALUE);
    if (!val) return 0;
    const char *p = text;
    while (*p) {
        const char *eol = strchr(p, '\n');
        size_t len      = eol ? (size_t)(eol - p) : strlen(p);
        char *line      = (char *)malloc(len + 1);
        if (!line) break;
        memcpy(line, p, len);
        line[len] = '\0';
        if (len && line[len - 1] == '\r') line[len - 1] = '\0';
        p = eol ? eol + 1 : p + len;
        lineno++;

        const char *q = skip_ws(line);
        if (*q == '\0' || *q == '#') {
            free(line);
            continue;
        }
        if (*q == '[') {
            warnf(warn, lineno, "tables are not supported; keys are flat");
            free(line);
            continue;
        }
        char key[64];
        size_t kl = 0;
        while (*q && (isalnum((unsigned char)*q) || *q == '-' || *q == '_') && kl + 1 < sizeof(key))
            key[kl++] = *q++;
        key[kl] = '\0';
        for (size_t i = 0; i < kl; i++)
            if (key[i] == '_') key[i] = '-';
        q = skip_ws(q);
        if (kl == 0 || *q != '=') {
            warnf(warn, lineno, "expected key = value");
            free(line);
            continue;
        }
        q                 = skip_ws(q + 1);
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
            if (k->type != PZ_CFG_LIST) {
                warnf(warn, lineno, "a list is only valid for list settings");
                free(line);
                continue;
            }
            pz_config_unset(c, key);
            q = skip_ws(q + 1);
            while (ok && *q && *q != ']') {
                if (parse_scalar(&q, val, PZ_CFG_MAX_VALUE) != 0 ||
                    pz_config_add(c, key, val, err, sizeof(err)) != 0)
                    ok = 0;
                q = skip_ws(q);
                if (*q == ',') q = skip_ws(q + 1);
            }
            if (ok && *q != ']') ok = 0;
            if (ok) q = skip_ws(q + 1);
        } else {
            if (parse_scalar(&q, val, PZ_CFG_MAX_VALUE) != 0)
                ok = 0;
            else {
                q = skip_ws(q);
                if (k->type == PZ_CFG_LIST)
                    ok = pz_config_set(c, key, val, err, sizeof(err)) == 0;
                else
                    ok = pz_config_set(c, key, val, err, sizeof(err)) == 0;
                if (!ok) {
                    warnf(warn, lineno, err);
                    free(line);
                    continue;
                }
            }
        }
        if (ok && *q != '\0' && *q != '#') ok = 0;
        if (!ok) {
            warnf(warn, lineno, "malformed value");
            pz_config_unset(c, key);
        } else
            loaded++;
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
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t r;
    while ((r = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += r;
        if (len + 1 >= cap) {
            if (cap >= ((size_t)4 << 20)) {
                free(buf);
                fclose(f);
                return -1;
            } /* 4 MiB sanity cap */
            char *nb = (char *)realloc(buf, cap * 2);
            if (!nb) {
                free(buf);
                fclose(f);
                return -1;
            }
            buf = nb;
            cap *= 2;
        }
    }
    fclose(f);
    buf[len] = '\0';
    int n    = pz_config_parse(c, buf, warn);
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
    char t[3] = {0, 0, 0};
    for (const char *p = v; *p; p++) {
        const char *s = NULL;
        if (*p == '"')
            s = "\\\"";
        else if (*p == '\\')
            s = "\\\\";
        else if (*p == '\n')
            s = "\\n";
        else if (*p == '\t')
            s = "\\t";
        else {
            t[0] = *p;
            t[1] = 0;
            s    = t;
        }
        if (append(buf, len, cap, s)) return -1;
    }
    return append(buf, len, cap, "\"");
}

char *pz_config_format(const PzConfig *c) {
    char *buf  = NULL;
    size_t len = 0, cap = 0;
    if (append(&buf, &len, &cap,
               "# Project Zero configuration — edit with `pz config` or by hand.\n"
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
            if (append(&buf, &len, &cap, "\n# ") || append(&buf, &len, &cap, group) ||
                append(&buf, &len, &cap, "\n"))
                goto fail;
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
        } else if (append(&buf, &len, &cap, e->v[0]))
            goto fail;
        if (append(&buf, &len, &cap, "\n")) goto fail;
    }
    return buf;
fail:
    free(buf);
    return NULL;
}

#if TN_POSIX
/* fsync the directory holding `path` so the rename itself survives a power loss. */
static void fsync_parent_dir(const char *path) {
    char dir[1100];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (!slash)
        snprintf(dir, sizeof(dir), ".");
    else if (slash == dir)
        dir[1] = '\0';
    else
        *slash = '\0';
    int dfd = open(dir, O_RDONLY);
    if (dfd >= 0) {
        (void)fsync(dfd);
        close(dfd);
    }
}
#endif

int pz_config_save(const PzConfig *c, const char *path) {
    if (!c || !path) return -1;
    char *text = pz_config_format(c);
    if (!text) return -1;
#if TN_POSIX
    tn_mkdir_parents(path);
#endif
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
#if TN_POSIX
    /* The file may hold an API key: create it already 0600 (no window with
     * umask-derived permissions), refuse to follow a pre-planted file or
     * symlink at the predictable temp name (O_EXCL after unlinking a stale one),
     * and fsync before the rename so a crash leaves the old or the new file,
     * never an empty one. */
    (void)unlink(tmp);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        free(text);
        return -1;
    }
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
        remove(tmp);
        free(text);
        return -1;
    }
    int ok = fputs(text, f) >= 0;
    ok     = (fflush(f) == 0) && ok;
    ok     = (fsync(fd) == 0) && ok;
    ok     = (fclose(f) == 0) && ok;
#else
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        free(text);
        return -1;
    }
    int ok = fputs(text, f) >= 0;
    ok     = (fclose(f) == 0) && ok;
#endif
    free(text);
    if (!ok || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
#if TN_POSIX
    fsync_parent_dir(path);
#endif
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
        } else if (warn) {
            char m[400];
            snprintf(m, sizeof(m), "%s ignored: %s", name, err);
            warn(m);
        }
    }
    return applied;
}

/* How many arguments the set entries add, and (in *n_set) how many entries are set. */
static int count_set_args(const PzConfig *c, int *n_set) {
    int extra = 0;
    for (int i = 0; c && i < c->n; i++) {
        const PzCfgEntry *e = &c->entries[i];
        if (!e->set) continue;
        (*n_set)++;
        if (e->key->type == PZ_CFG_BOOL)
            extra += strcmp(e->v[0], "true") == 0;
        else
            extra += 2 * e->n;
    }
    return extra;
}

#define PZ_FLAG_NAME_MAX 72 /* room for "--" and the longest setting name */

int pz_config_merge_argv(const PzConfig *c, int argc, char **argv, char ***out_argv) {
    if (!out_argv || argc < 1 || !argv) return -1;
    int n_set = 0;
    int extra = count_set_args(c, &n_set);
    /* One block: the argument pointers, then the "--key" strings they point at (the caller frees
     * the pointers and the strings go with them). */
    size_t n_ptr = (size_t)argc + (size_t)extra + 1;
    char **av    = (char **)malloc(n_ptr * sizeof(char *) + (size_t)n_set * PZ_FLAG_NAME_MAX);
    if (!av) return -1;
    char *names = (char *)(av + n_ptr);
    int k       = 0;
    av[k++]     = argv[0];
    for (int i = 0; c && i < c->n; i++) {
        const PzCfgEntry *e = &c->entries[i];
        if (!e->set) continue;
        char *flag = names;
        names += PZ_FLAG_NAME_MAX;
        snprintf(flag, PZ_FLAG_NAME_MAX, "--%s", e->key->key);
        if (e->key->type == PZ_CFG_BOOL) {
            if (strcmp(e->v[0], "true") == 0) av[k++] = flag;
            continue;
        }
        for (int j = 0; j < e->n; j++) {
            av[k++] = flag;
            av[k++] = e->v[j];
        }
    }
    for (int i = 1; i < argc; i++) av[k++] = argv[i];
    av[k]     = NULL;
    *out_argv = av;
    return k;
}

/* ── provenance (Phase 40.3) ──────────────────────────────────────────── */

const char *pz_config_source_name(int src) {
    switch (src) {
        case PZ_SRC_FILE:
            return "file";
        case PZ_SRC_ENV:
            return "env";
        case PZ_SRC_CLI:
            return "cli";
        default:
            return "default";
    }
}

int pz_config_mark_cli(PzConfig *c, int argc, char **argv) {
    int marked = 0;
    for (int i = 1; c && argv && i < argc; i++) {
        /* parse_args only takes the "--key value" form. */
        if (!argv[i] || strncmp(argv[i], "--", 2) != 0) continue;
        PzCfgEntry *e = entry_for(c, argv[i] + 2);
        if (e && e->src != PZ_SRC_CLI) {
            e->src = PZ_SRC_CLI;
            marked++;
        }
    }
    return marked;
}

const char *pz_provenance_source(const PzProvenance *p, const char *key) {
    /* entry_for, not pz_config_get: a key given only as a flag is unset here. */
    const PzCfgEntry *e = (p && p->config && key) ? entry_for((PzConfig *)p->config, key) : NULL;
    return pz_config_source_name(e ? e->src : PZ_SRC_DEFAULT);
}
