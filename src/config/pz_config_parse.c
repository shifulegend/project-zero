/*
 * src/config/pz_config_parse.c
 *
 * The TOML-subset reader of the settings file (pz_config_parse): flat `key = value` lines with
 * strings, numbers, booleans and lists. Split out of pz_config.c; behaviour unchanged and pinned by
 * tests/test_pz_config_parse_golden.c.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/pz_config.h"
#include "core/escape.h"

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

/* A bare (unquoted) value ends at a separator, a list close, a comment or whitespace. */
static int ends_bare_value(char ch) {
    return ch == ',' || ch == ']' || ch == '#' || isspace((unsigned char)ch);
}

/* Parses one scalar (string, number, bool) at *pp into out. */
static int parse_scalar(const char **pp, char *out, size_t outn) {
    const char *p = *pp;
    if (*p == '"') {
        p++;
        if (parse_string(&p, out, outn) != 0) return -1;
    } else {
        size_t o = 0;
        while (*p && !ends_bare_value(*p)) {
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

/* Copies the next line of *pp (without its \n or \r\n) and advances *pp past it; NULL when out of
 * memory. */
static char *take_line(const char **pp) {
    const char *p   = *pp;
    const char *eol = strchr(p, '\n');
    size_t len      = eol ? (size_t)(eol - p) : strlen(p);
    char *line      = (char *)malloc(len + 1);
    if (!line) return NULL;
    memcpy(line, p, len);
    line[len] = '\0';
    if (len && line[len - 1] == '\r') line[len - 1] = '\0';
    *pp = eol ? eol + 1 : p + len;
    return line;
}

/* Reads a setting name at *pq ('_' reads as '-'); returns its length (0 when there is none). */
static size_t read_key(const char **pq, char *key, size_t cap) {
    const char *q = *pq;
    size_t kl     = 0;
    while (*q && (isalnum((unsigned char)*q) || *q == '-' || *q == '_') && kl + 1 < cap)
        key[kl++] = *q++;
    key[kl] = '\0';
    for (size_t i = 0; i < kl; i++)
        if (key[i] == '_') key[i] = '-';
    *pq = q;
    return kl;
}

/* One parse in progress: the config, the warning sink and a scratch buffer for values. */
typedef struct {
    PzConfig *c;
    void (*warn)(const char *);
    char *val;
    int lineno;
} ParseCtx;

/* `[a, "b", ...]` after the key's '='; replaces the setting's list. 0 on a malformed list. */
static int parse_list_value(ParseCtx *cx, const char **pq, const char *key) {
    char err[256];
    const char *q = *pq;
    int ok        = 1;
    pz_config_unset(cx->c, key);
    q = skip_ws(q + 1);
    while (ok && *q && *q != ']') {
        if (parse_scalar(&q, cx->val, PZ_CFG_MAX_VALUE) != 0 ||
            pz_config_add(cx->c, key, cx->val, err, sizeof(err)) != 0)
            ok = 0;
        q = skip_ws(q);
        if (*q == ',') q = skip_ws(q + 1);
    }
    if (ok && *q != ']') ok = 0;
    if (ok) q = skip_ws(q + 1);
    *pq = q;
    return ok;
}

/* A single value. Returns 1 when stored, 0 when malformed (caller warns), -1 when the value was
 * well formed but refused by validation (already warned). */
static int parse_scalar_value(ParseCtx *cx, const char **pq, const char *key) {
    char err[256];
    const char *q = *pq;
    if (parse_scalar(&q, cx->val, PZ_CFG_MAX_VALUE) != 0) return 0;
    q = skip_ws(q);
    if (pz_config_set(cx->c, key, cx->val, err, sizeof(err)) != 0) {
        warnf(cx->warn, cx->lineno, err);
        return -1;
    }
    *pq = q;
    return 1;
}

/* The value after `key =`, by the setting's type. Returns 1 when stored, 0 when malformed (the
 * caller warns and unsets), -1 when already handled (a warning was printed). */
static int parse_value(ParseCtx *cx, const char **pq, const PzCfgKey *k, const char *key) {
    if (**pq != '[') return parse_scalar_value(cx, pq, key);
    if (k->type != PZ_CFG_LIST) {
        warnf(cx->warn, cx->lineno, "a list is only valid for list settings");
        return -1;
    }
    return parse_list_value(cx, pq, key);
}

/* Reads `key =` and looks the setting up, leaving *pq at the value. NULL (after a warning) when the
 * line is not `key = value` or the setting is unknown. */
static const PzCfgKey *read_setting(ParseCtx *cx, const char **pq, char *key, size_t cap) {
    size_t kl     = read_key(pq, key, cap);
    const char *q = skip_ws(*pq);
    if (kl == 0 || *q != '=') {
        warnf(cx->warn, cx->lineno, "expected key = value");
        return NULL;
    }
    *pq               = skip_ws(q + 1);
    const PzCfgKey *k = pz_config_find_key(key);
    if (!k) {
        char m[128];
        snprintf(m, sizeof(m), "unknown setting \"%s\" ignored", key);
        warnf(cx->warn, cx->lineno, m);
    }
    return k;
}

/* One line. Returns 1 when it loaded a setting. */
static int parse_line(ParseCtx *cx, const char *line) {
    const char *q = skip_ws(line);
    if (*q == '\0' || *q == '#') return 0;
    if (*q == '[') {
        warnf(cx->warn, cx->lineno, "tables are not supported; keys are flat");
        return 0;
    }
    char key[64];
    const PzCfgKey *k = read_setting(cx, &q, key, sizeof(key));
    if (!k) return 0;
    int ok = parse_value(cx, &q, k, key);
    if (ok < 0) return 0;
    if (ok && *q != '\0' && *q != '#') ok = 0;
    if (ok) return 1;
    warnf(cx->warn, cx->lineno, "malformed value");
    pz_config_unset(cx->c, key);
    return 0;
}

int pz_config_parse(PzConfig *c, const char *text, void (*warn)(const char *)) {
    if (!c || !text) return 0;
    ParseCtx cx = {c, warn, (char *)malloc(PZ_CFG_MAX_VALUE), 0};
    if (!cx.val) return 0;
    int loaded    = 0;
    const char *p = text;
    while (*p) {
        char *line = take_line(&p);
        if (!line) break;
        cx.lineno++;
        loaded += parse_line(&cx, line);
        free(line);
    }
    free(cx.val);
    return loaded;
}
