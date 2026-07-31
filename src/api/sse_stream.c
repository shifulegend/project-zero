/*
 * Phase 21 — OpenAI-Compatible API Layer
 * src/api/sse_stream.c
 *
 * Server-Sent Events (SSE) helpers — format and write OpenAI-compatible
 * streaming/non-streaming responses to a file descriptor.
 */

#include "api/sse_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── JSON string escaping ────────────────────────────────────────────────── */

/*
 * Escape src into dst, adding JSON \-escapes for special characters.
 * dst must have at least (3*src_len + 3) bytes available in the worst case.
 * Returns pointer past end of written content in dst (does NOT NUL-terminate).
 */
static char *json_escape(char *dst, const char *src) {
    while (*src) {
        unsigned char c = (unsigned char)*src++;
        switch (c) {
            case '"':  *dst++ = '\\'; *dst++ = '"';  break;
            case '\\': *dst++ = '\\'; *dst++ = '\\'; break;
            case '\n': *dst++ = '\\'; *dst++ = 'n';  break;
            case '\r': *dst++ = '\\'; *dst++ = 'r';  break;
            case '\t': *dst++ = '\\'; *dst++ = 't';  break;
            default:
                if (c < 0x20) {
                    /* Control character: emit \uXXXX */
                    dst += sprintf(dst, "\\u%04x", (unsigned)c);
                } else {
                    *dst++ = (char)c;
                }
                break;
        }
    }
    return dst;
}

/* ── write() wrapper that retries on EINTR ──────────────────────────────── */
static ssize_t write_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, buf + sent, len - sent);
        if (n < 0) return -1;
        sent += (size_t)n;
    }
    return (ssize_t)sent;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int sse_write_token(int fd, const char *id, const char *token_text) {
    if (!token_text || !id) return -1;

    /* Escape the token into a local buffer.  Tokens are at most a few bytes
     * each; 1024 bytes is more than sufficient even after escaping. */
    char escaped[1024];
    char *end = json_escape(escaped, token_text);
    *end = '\0';
    size_t esc_len = (size_t)(end - escaped);

    /* Build the SSE data line.  Fixed overhead ~ 140 bytes + id + token. */
    char buf[2048];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%s\","
             "\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"index\":0,"
                          "\"delta\":{\"content\":\"%.*s\"},"
                          "\"finish_reason\":null}]}\n\n",
        id,
        (int)esc_len, escaped);
    if (n < 0 || (size_t)n >= sizeof(buf)) return -1;

    return (int)write_all(fd, buf, (size_t)n);
}

int sse_write_done(int fd, const char *id) {
    if (!id) return -1;

    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%s\","
             "\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"index\":0,"
                          "\"delta\":{},"
                          "\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n",
        id);
    if (n < 0 || (size_t)n >= sizeof(buf)) return -1;
    return (int)write_all(fd, buf, (size_t)n);
}

char *sse_format_full_response(const char *id, const char *full_text) {
    if (!id || !full_text) return NULL;

    /* Allocate escaped output buffer.  Upper bound: control characters (<0x20)
     * expand to \uXXXX (6 bytes each); all other special chars need at most 2
     * bytes (e.g. \" or \\).  6× src_len covers the worst-case scenario. */
    size_t src_len = strlen(full_text);
    size_t esc_cap = src_len * 6 + 4;
    char  *escaped = (char *)malloc(esc_cap);
    if (!escaped) return NULL;

    char *end = json_escape(escaped, full_text);
    *end = '\0';
    size_t esc_len = (size_t)(end - escaped);

    /* Build response JSON */
    size_t buf_cap = esc_len + 512;
    char  *buf     = (char *)malloc(buf_cap);
    if (!buf) { free(escaped); return NULL; }

    int n = snprintf(buf, buf_cap,
        "{\"id\":\"chatcmpl-%s\","
         "\"object\":\"chat.completion\","
         "\"choices\":[{\"index\":0,"
                      "\"message\":{\"role\":\"assistant\","
                                   "\"content\":\"%.*s\"},"
                      "\"finish_reason\":\"stop\"}],"
         "\"usage\":{\"prompt_tokens\":0,"
                   "\"completion_tokens\":0,"
                   "\"total_tokens\":0}}",
        id,
        (int)esc_len, escaped);

    free(escaped);
    if (n < 0 || (size_t)n >= buf_cap) { free(buf); return NULL; }
    return buf;
}

int sse_write_full_response(int fd, const char *id, const char *full_text) {
    char *json = sse_format_full_response(id, full_text);
    if (!json) return -1;
    int written = (int)write_all(fd, json, strlen(json));
    free(json);
    return written;
}
