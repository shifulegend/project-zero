#define _POSIX_C_SOURCE 200809L
#include "core/log.h"

#include <ctype.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static atomic_int g_format = TN_LOG_TEXT;
static atomic_int g_level  = TN_LOG_INFO;

void tn_log_set_format(TnLogFormat fmt) {
    atomic_store(&g_format, (int)fmt);
}

TnLogFormat tn_log_get_format(void) {
    return (TnLogFormat)atomic_load(&g_format);
}

void tn_log_set_level(TnLogLevel lvl) {
    atomic_store(&g_level, (int)lvl);
}

int tn_log_parse_format(const char *s, TnLogFormat *out) {
    if (!s) return -1;
    if (strcmp(s, "text") == 0)
        *out = TN_LOG_TEXT;
    else if (strcmp(s, "json") == 0)
        *out = TN_LOG_JSON;
    else
        return -1;
    return 0;
}

int tn_log_parse_level(const char *s, TnLogLevel *out) {
    if (!s) return -1;
    if (strcmp(s, "debug") == 0)
        *out = TN_LOG_DEBUG;
    else if (strcmp(s, "info") == 0)
        *out = TN_LOG_INFO;
    else if (strcmp(s, "warn") == 0)
        *out = TN_LOG_WARN;
    else if (strcmp(s, "error") == 0)
        *out = TN_LOG_ERROR;
    else
        return -1;
    return 0;
}

const char *tn_log_level_name(TnLogLevel lvl) {
    switch (lvl) {
        case TN_LOG_DEBUG:
            return "debug";
        case TN_LOG_INFO:
            return "info";
        case TN_LOG_WARN:
            return "warn";
        default:
            return "error";
    }
}

/* ── Redaction ───────────────────────────────────────────────────────────── */

#define REDACTED "[redacted]"

static int is_token_char(char c) {
    return isalnum((unsigned char)c) || c == '.' || c == '_' || c == '~' || c == '+' || c == '/' ||
           c == '=' || c == '-';
}

/* Case-insensitive prefix test. */
static int starts_with_ci(const char *s, const char *prefix) {
    for (; *prefix; s++, prefix++)
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix)) return 0;
    return 1;
}

/* Each helper returns the length of the credential starting at `s` and sets *skip to the length
 * of the keyword before it (kept in the output); 0 when `s` does not start that credential. */

/* "Authorization:" header: everything after the colon to the end of the line. */
static size_t span_authorization(const char *s, size_t *skip) {
    if (!starts_with_ci(s, "authorization:")) return 0;
    size_t n = strlen("authorization:");
    while (s[n] == ' ') n++;
    size_t e = n;
    while (s[e] && s[e] != '\n' && s[e] != '\r') e++;
    *skip = n;
    return e - n;
}

/* "Bearer <token>": the token. */
static size_t span_bearer(const char *s, size_t *skip) {
    if (!starts_with_ci(s, "bearer ")) return 0;
    size_t n = strlen("bearer ");
    size_t e = n;
    while (is_token_char(s[e])) e++;
    /* "Bearer" as the scheme is redacted whatever follows; lower case ("the bearer of news") only
     * when the next word looks like a long token. */
    if (strncmp(s, "Bearer ", 7) != 0 && e - n < 12) return 0;
    *skip = n;
    return e - n;
}

/* api_key / api-key / apikey followed by = or : and a value: the value. */
static size_t span_api_key(const char *s, size_t *skip) {
    if (!starts_with_ci(s, "api_key") && !starts_with_ci(s, "api-key") &&
        !starts_with_ci(s, "apikey"))
        return 0;
    size_t n = starts_with_ci(s, "apikey") ? 6 : 7;
    while (s[n] == '"' || s[n] == '\'') n++;
    if (s[n] != '=' && s[n] != ':') return 0;
    n++;
    while (s[n] == ' ' || s[n] == '"' || s[n] == '\'') n++;
    size_t e = n;
    while (is_token_char(s[e])) e++;
    *skip = n;
    return e - n;
}

/* Length of the credential that starts at `s` (the part to replace), 0 when `s` starts none. */
static size_t secret_span(const char *s, size_t *skip) {
    *skip    = 0;
    size_t n = span_authorization(s, skip);
    if (!n) n = span_bearer(s, skip);
    if (!n) n = span_api_key(s, skip);
    return n;
}

/* ── Rendering ───────────────────────────────────────────────────────────── */

typedef struct {
    char *p;
    size_t cap, n;
} Out;

static void put(Out *o, char c) {
    if (o->n + 1 < o->cap) o->p[o->n] = c;
    o->n++;
}

static void put_str(Out *o, const char *s) {
    for (; *s; s++) put(o, *s);
}

/* One byte of the message in `fmt` form. JSON: standard escapes. Text: control characters other
 * than the message's own newline and tab are escaped as \xNN so a value cannot forge a line. */
static void put_escaped(Out *o, unsigned char c, TnLogFormat fmt, int keep_newline) {
    static const char hex[] = "0123456789abcdef";
    if (fmt == TN_LOG_JSON) {
        if (c == '"')
            put_str(o, "\\\"");
        else if (c == '\\')
            put_str(o, "\\\\");
        else if (c == '\n')
            put_str(o, "\\n");
        else if (c == '\r')
            put_str(o, "\\r");
        else if (c == '\t')
            put_str(o, "\\t");
        else if (c < 0x20 || c == 0x7f) {
            put_str(o, "\\u00");
            put(o, hex[c >> 4]);
            put(o, hex[c & 15]);
        } else
            put(o, (char)c);
        return;
    }
    if ((c == '\n' && keep_newline) || c == '\t' || (c >= 0x20 && c != 0x7f)) {
        put(o, (char)c);
    } else {
        put_str(o, "\\x");
        put(o, hex[c >> 4]);
        put(o, hex[c & 15]);
    }
}

size_t tn_log_render(TnLogFormat fmt, const char *msg, char *out, size_t cap) {
    Out o      = {out, cap, 0};
    size_t len = strlen(msg);
    /* A trailing newline belongs to the format: kept in text, dropped in JSON. In text mode every
     * other newline is a line break the call site meant (multi-line messages). */
    int trailing_nl = len > 0 && msg[len - 1] == '\n';
    size_t body_len = trailing_nl && fmt == TN_LOG_JSON ? len - 1 : len;
    for (size_t i = 0; i < body_len; i++) {
        size_t skip, span = secret_span(msg + i, &skip);
        if (span > 0) {
            for (size_t k = 0; k < skip; k++) put_escaped(&o, (unsigned char)msg[i + k], fmt, 0);
            put_str(&o, REDACTED);
            i += skip + span - 1;
            continue;
        }
        /* A text-mode newline is kept only when it ends the message or the call site wrote it
         * as a literal part of the format; values are escaped by their own bytes below. */
        put_escaped(&o, (unsigned char)msg[i], fmt, fmt == TN_LOG_TEXT);
    }
    if (cap > 0) o.p[o.n < cap ? o.n : cap - 1] = '\0';
    return o.n < cap ? o.n : cap - 1;
}

size_t tn_log_sanitize(const char *s, char *out, size_t cap) {
    static const char hex[] = "0123456789abcdef";
    size_t n                = 0;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        int ctl         = c < 0x20 || c == 0x7f;
        if (n + (ctl ? 4 : 1) + 1 > cap) break;
        if (ctl) {
            out[n++] = '\\';
            out[n++] = 'x';
            out[n++] = hex[c >> 4];
            out[n++] = hex[c & 15];
        } else {
            out[n++] = (char)c;
        }
    }
    if (cap > 0) out[n] = '\0';
    return n;
}

/* ── Entry points ────────────────────────────────────────────────────────── */

static void timestamp(char *buf, size_t cap) {
    struct timespec ts;
    struct tm tmv;
    clock_gettime(CLOCK_REALTIME, &ts);
    gmtime_r(&ts.tv_sec, &tmv);
    size_t n = strftime(buf, cap, "%Y-%m-%dT%H:%M:%S", &tmv);
    snprintf(buf + n, cap - n, ".%03dZ", (int)(ts.tv_nsec / 1000000));
}

void tn_logv(TnLogLevel lvl, const char *component, const char *fmt, va_list ap) {
    if ((int)lvl < atomic_load(&g_level)) return;
    char small[1024];
    va_list ap2;
    va_copy(ap2, ap);
    int need  = vsnprintf(small, sizeof small, fmt, ap);
    char *msg = small;
    if (need >= (int)sizeof small) {
        msg = (char *)malloc((size_t)need + 1);
        if (msg)
            vsnprintf(msg, (size_t)need + 1, fmt, ap2);
        else
            msg = small; /* out of memory: the truncated message is still logged */
    }
    va_end(ap2);

    TnLogFormat f = tn_log_get_format();
    size_t cap    = strlen(msg) * 6 + 160 + (component ? strlen(component) * 6 : 0);
    char *line    = (char *)malloc(cap);
    if (line) {
        size_t n = 0;
        if (f == TN_LOG_JSON) {
            char ts[40], comp[128];
            timestamp(ts, sizeof ts);
            tn_log_render(TN_LOG_JSON, component ? component : "", comp, sizeof comp);
            n = (size_t)snprintf(line, cap,
                                 "{\"ts\":\"%s\",\"level\":\"%s\",\"component\":\"%s\",\"msg\":\"",
                                 ts, tn_log_level_name(lvl), comp);
            n += tn_log_render(TN_LOG_JSON, msg, line + n, cap - n - 4);
            n += (size_t)snprintf(line + n, cap - n, "\"}\n");
        } else {
            n = tn_log_render(TN_LOG_TEXT, msg, line, cap);
        }
        fwrite(line, 1, n, stderr); /* one write per line: threads do not interleave */
        free(line);
    }
    if (msg != small) free(msg);
}

void tn_log(TnLogLevel lvl, const char *component, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    tn_logv(lvl, component, fmt, ap);
    va_end(ap);
}

void tn_log_out(TnLogLevel lvl, const char *component, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (tn_log_get_format() == TN_LOG_JSON) {
        tn_logv(lvl, component, fmt, ap);
    } else if ((int)lvl >= atomic_load(&g_level)) {
        vfprintf(stdout, fmt, ap);
    }
    va_end(ap);
}
