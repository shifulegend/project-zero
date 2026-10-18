#include "api/json_write.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void jb_init(JsonBuf *jb) {
    memset(jb, 0, sizeof(*jb));
}

static void jb_append(JsonBuf *jb, const char *s, size_t n) {
    if (jb->err) return;
    if (jb->len + n + 1 > jb->cap) {
        size_t nc = jb->cap ? jb->cap : 256;
        while (nc < jb->len + n + 1) nc *= 2;
        char *nb = (char *)realloc(jb->buf, nc);
        if (!nb) { jb->err = 1; return; }
        jb->buf = nb;
        jb->cap = nc;
    }
    memcpy(jb->buf + jb->len, s, n);
    jb->len += n;
    jb->buf[jb->len] = '\0';
}

void jb_raw(JsonBuf *jb, const char *text) {
    if (text) jb_append(jb, text, strlen(text));
}

void jb_str(JsonBuf *jb, const char *s) {
    if (!s) { jb_raw(jb, "null"); return; }
    jb_append(jb, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char esc[8];
        switch (*p) {
            case '"':  jb_append(jb, "\\\"", 2); break;
            case '\\': jb_append(jb, "\\\\", 2); break;
            case '\n': jb_append(jb, "\\n", 2);  break;
            case '\r': jb_append(jb, "\\r", 2);  break;
            case '\t': jb_append(jb, "\\t", 2);  break;
            default:
                if (*p < 0x20) {
                    snprintf(esc, sizeof(esc), "\\u%04x", *p);
                    jb_append(jb, esc, 6);
                } else {
                    jb_append(jb, (const char *)p, 1);
                }
        }
    }
    jb_append(jb, "\"", 1);
}

void jb_key(JsonBuf *jb, const char *key) {
    jb_str(jb, key);
    jb_append(jb, ":", 1);
}

void jb_int(JsonBuf *jb, long long v) {
    char t[32];
    snprintf(t, sizeof(t), "%lld", v);
    jb_raw(jb, t);
}

void jb_num(JsonBuf *jb, double v) {
    if (isnan(v) || isinf(v)) { jb_raw(jb, "null"); return; }
    char t[48];
    snprintf(t, sizeof(t), "%.6g", v);
    jb_raw(jb, t);
}

void jb_bool(JsonBuf *jb, int v) {
    jb_raw(jb, v ? "true" : "false");
}

char *jb_finish(JsonBuf *jb) {
    if (jb->err || !jb->buf) { jb_free(jb); return NULL; }
    char *out = jb->buf;
    jb->buf = NULL;
    jb->len = jb->cap = 0;
    return out;
}

void jb_free(JsonBuf *jb) {
    free(jb->buf);
    jb->buf = NULL;
    jb->len = jb->cap = 0;
}
