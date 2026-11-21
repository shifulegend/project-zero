/*
 * api/json_scan.c — the JSON scanning primitives shared by the chat-request parser and the generic
 * top-level field access: whitespace, strings (with escapes and \uXXXX), numbers, booleans and
 * skipping a whole value. See api/json_scan.h.
 */

#include "api/json_scan.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/escape.h"

/* ── Internal helpers ────────────────────────────────────────────────────── */

/* Skip whitespace; advance *p. */
void json_skip_ws(const char **p) {
    while (**p && (unsigned char)**p <= ' ') (*p)++;
}

/* Expect and consume a literal character.  Returns 0 on mismatch. */
int json_expect_char(const char **p, char c) {
    json_skip_ws(p);
    if (**p != c) return 0;
    (*p)++;
    return 1;
}

static int hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    for (int k = 0; k < 4; k++) {
        char h = s[k];
        v <<= 4;
        if (h >= '0' && h <= '9')
            v |= (unsigned)(h - '0');
        else if (h >= 'a' && h <= 'f')
            v |= (unsigned)(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F')
            v |= (unsigned)(h - 'A' + 10);
        else
            return 0;
    }
    *out = v;
    return 1;
}

static int utf8_encode(unsigned cp, char *o) {
    if (cp < 0x80) {
        o[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        o[0] = (char)(0xC0 | (cp >> 6));
        o[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12));
        o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18));
    o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Append one decoded byte, dropping it when the buffer is full (one byte stays for the NUL). */
static void put_byte(char *dst, size_t dst_cap, size_t *i, char c) {
    if (dst && *i + 1 < dst_cap) dst[(*i)++] = c;
}

/* A high surrogate at cp followed by "\\uDC00..\\uDFFF" (*p on the last hex digit of the first
 * escape) combines into one code point and consumes the second escape. */
static unsigned join_surrogates(const char **p, unsigned cp) {
    if (cp < 0xD800 || cp > 0xDBFF || (*p)[1] != '\\' || (*p)[2] != 'u') return cp;
    unsigned lo = 0;
    if (!hex4(*p + 3, &lo) || lo < 0xDC00 || lo > 0xDFFF) return cp;
    (*p) += 6;
    return 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
}

/* \uXXXX (+ optional low surrogate) -> UTF-8. *p is on the 'u'. The escape is >= 6 input bytes
 * and UTF-8 output <= 4 per code point, so json_parse_string_heap's length bound holds.
 * (2026-10-18: previously every \u escape became '?'.) Appends all but the last output byte and
 * returns that one, which the caller appends like any other character. */
static char decode_unicode_escape(const char **p, char *dst, size_t dst_cap, size_t *i) {
    unsigned cp = 0;
    if (!hex4(*p + 1, &cp)) return '?';
    (*p) += 4;
    cp = join_surrogates(p, cp);
    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD; /* lone surrogate */
    char u8[4];
    int nb = utf8_encode(cp, u8);
    for (int k = 0; k < nb - 1; k++) put_byte(dst, dst_cap, i, u8[k]);
    return u8[nb - 1];
}

/* The character an escape stands for; *p is on the byte after the backslash. */
static char decode_escape(const char **p, char *dst, size_t dst_cap, size_t *i) {
    int simple = tn_escape_simple(**p);
    if (simple >= 0) return (char)simple;
    if (**p == '/') return '/';
    if (**p == 'u') return decode_unicode_escape(p, dst, dst_cap, i);
    return **p;
}

/*
 * Parse a JSON string starting at *p (which must point to the opening '"').
 * Writes up to (dst_cap - 1) decoded bytes into dst and NUL-terminates.
 * Advances *p past the closing '"'.
 * Returns 0 on error.
 */
int json_parse_string(const char **p, char *dst, size_t dst_cap) {
    json_skip_ws(p);
    if (**p != '"') return 0;
    (*p)++; /* skip opening quote */

    size_t i = 0;
    while (**p && **p != '"') {
        char c = **p;
        if (c == '\\') {
            (*p)++;
            /* 2026-10-18: a trailing backslash used to step past the NUL and
             * keep reading beyond the buffer. */
            if (**p == '\0') return 0;
            c = decode_escape(p, dst, dst_cap, &i);
        }
        put_byte(dst, dst_cap, &i, c);
        (*p)++;
    }
    if (**p != '"') return 0;
    (*p)++; /* skip closing quote */
    if (dst && dst_cap > 0) dst[i] = '\0';
    return 1;
}

/*
 * Heap-allocate a copy of a JSON string (may be longer than fixed buffers).
 * Returns NULL on error.  Caller must free().
 */
char *json_parse_string_heap(const char **p) {
    json_skip_ws(p);
    if (**p != '"') return NULL;

    /* First pass: measure length */
    const char *scan = *p + 1; /* skip opening quote */
    size_t len       = 0;
    while (*scan && *scan != '"') {
        if (*scan == '\\') {
            scan++;
            if (!*scan) return NULL;
        }
        len++;
        scan++;
    }
    if (*scan != '"') return NULL;

    /* Second pass: fill buffer */
    char *buf = (char *)malloc(len + 1);
    if (!buf) return NULL;

    if (!json_parse_string(p, buf, len + 1)) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* Parse a JSON number and return its double value. Advances *p. */
double json_parse_number(const char **p) {
    json_skip_ws(p);
    char buf[64];
    size_t i = 0;
    while (**p && (isdigit((unsigned char)**p) || **p == '.' || **p == '-' || **p == '+' ||
                   **p == 'e' || **p == 'E')) {
        if (i < sizeof(buf) - 1) buf[i++] = **p;
        (*p)++;
    }
    buf[i] = '\0';
    return atof(buf);
}

/* Parse a JSON boolean (true/false). Returns -1 on error, 0/1 on success. */
int json_parse_bool(const char **p) {
    json_skip_ws(p);
    if (strncmp(*p, "true", 4) == 0) {
        *p += 4;
        return 1;
    }
    if (strncmp(*p, "false", 5) == 0) {
        *p += 5;
        return 0;
    }
    return -1;
}

/* Skip an arbitrary JSON value (used to skip unknown keys). */
void json_skip_value(const char **p) {
    json_skip_ws(p);
    char c = **p;
    if (c == '"') {
        char discard[4096];
        json_parse_string(p, discard, sizeof(discard));
    } else if (c == '{') {
        /* Skip object */
        int depth = 0;
        while (**p) {
            if (**p == '{')
                depth++;
            else if (**p == '}') {
                depth--;
                (*p)++;
                if (depth == 0) return;
                continue;
            } else if (**p == '"') {
                char d[4096];
                json_parse_string(p, d, sizeof(d));
                continue;
            }
            (*p)++;
        }
    } else if (c == '[') {
        int depth = 0;
        while (**p) {
            if (**p == '[')
                depth++;
            else if (**p == ']') {
                depth--;
                (*p)++;
                if (depth == 0) return;
                continue;
            } else if (**p == '"') {
                char d[4096];
                json_parse_string(p, d, sizeof(d));
                continue;
            }
            (*p)++;
        }
    } else {
        /* number, bool, null */
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}
