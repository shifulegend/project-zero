/*
 * src/api/http_response.c
 *
 * Response writers of the HTTP server: status line + headers + body for plain,
 * streaming (SSE) and error replies. Split out of http_server.c; the code is unchanged.
 */

#include "api/http_response.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/http_status.h"
#include "api/net_io.h"

/* ── HTTP response helpers ───────────────────────────────────────────────── */

/* Fixed text of a header block is under 256 bytes; extra headers longer than the rest of the buffer
 * would be cut off by snprintf (which returns the would-be length, not the written one). They are
 * dropped instead, so every header block that is sent is complete. */
static const char *fit_extra(const char *extra_headers) {
    if (!extra_headers || strlen(extra_headers) + 256 > HTTP_HDR_BUF) return "";
    return extra_headers;
}

/* Length to send for a formatted header block: 0 when formatting failed. */
static size_t hdr_len(int n, size_t cap) {
    if (n <= 0) return 0;
    return (size_t)n < cap ? (size_t)n : cap - 1;
}

/* Copies msg into out as the inside of a JSON string (quotes, backslashes and control characters
 * escaped), truncated to cap-1 bytes without splitting an escape. */
static void json_escape_into(char *out, size_t cap, const char *msg) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)msg; p && *p; p++) {
        char tmp[8];
        size_t k;
        if (*p == '"' || *p == '\\') {
            tmp[0] = '\\';
            tmp[1] = (char)*p;
            k      = 2;
        } else if (*p < 0x20) {
            k = (size_t)snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
        } else {
            tmp[0] = (char)*p;
            k      = 1;
        }
        if (o + k + 1 > cap) break;
        memcpy(out + o, tmp, k);
        o += k;
    }
    out[o] = '\0';
}

/* extra_headers, if non-NULL, must already be "\r\n"-terminated per line
 * (e.g. from cors_format_headers()) and is spliced in ahead of the blank
 * line that ends the header block. */
void http_send_response_ex(int fd, int status, const char *content_type, const char *body,
                           int streaming, const char *extra_headers) {
    char hdr[HTTP_HDR_BUF];
    size_t body_len   = body ? strlen(body) : 0;
    const char *extra = fit_extra(extra_headers);
    int n;
    if (streaming) {
        /* 2026-07-15: pre-existing bug — this claimed "Transfer-Encoding:
         * chunked" but the SSE writers (sse_write_token/done) write raw
         * unframed bytes, not HTTP chunk-size-prefixed framing. A strict
         * client (browser fetch/EventSource; curl --raw) sees a protocol
         * violation. Since the response always closes the connection when
         * done, RFC 7230 §3.3.3 case 7 already lets EOF delimit the body —
         * no Transfer-Encoding header is needed at all. */
        n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d OK\r\n"
                     "Content-Type: %s\r\n"
                     "Cache-Control: no-cache\r\n"
                     "Connection: close\r\n"
                     "%s"
                     "\r\n",
                     status, content_type, extra);
    } else {
        n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d OK\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "%s"
                     "\r\n",
                     status, content_type, body_len, extra);
    }
    if (hdr_len(n, sizeof(hdr)) == 0 || net_write_all(fd, hdr, hdr_len(n, sizeof(hdr))) != 0)
        return;
    if (body && !streaming)
        (void)net_write_all(fd, body, body_len); /* peer gone: nothing left to do */
}

void http_send_error_ex(int fd, int status, const char *msg, const char *extra_headers) {
    char body[HTTP_ERR_BODY_BUF];
    char esc[HTTP_ERR_BODY_BUF - 64]; /* the JSON frame below adds 51 bytes */
    json_escape_into(esc, sizeof(esc), msg);
    snprintf(body, sizeof(body), "{\"error\":{\"message\":\"%s\",\"type\":\"server_error\"}}", esc);
    const char *extra = fit_extra(extra_headers);
    char hdr[HTTP_HDR_BUF];
    int n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d Error\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "%s"
                     "\r\n",
                     status, strlen(body), extra);
    if (hdr_len(n, sizeof(hdr)) > 0) {
        if (net_write_all(fd, hdr, hdr_len(n, sizeof(hdr))) == 0)
            (void)net_write_all(fd, body, strlen(body)); /* peer gone: nothing left to do */
    }
}

void http_send_error(int fd, int status, const char *msg) {
    http_send_error_ex(fd, status, msg, NULL);
}

void http_send_json_heap(int fd, int status, char *body, const char *extra_headers) {
    if (!body) {
        http_send_error_ex(fd, HTTP_INTERNAL_ERROR, "OOM", extra_headers);
        return;
    }
    if (status == HTTP_OK)
        http_send_response_ex(fd, HTTP_OK, "application/json", body, 0, extra_headers);
    else {
        char hdr[HTTP_HDR_BUF];
        int n = snprintf(hdr, sizeof(hdr),
                         "HTTP/1.1 %d Error\r\nContent-Type: application/json\r\n"
                         "Content-Length: %zu\r\nConnection: close\r\n%s\r\n",
                         status, strlen(body), fit_extra(extra_headers));
        /* A peer that is gone must not skip the free(body) below. */
        if (hdr_len(n, sizeof(hdr)) > 0 && net_write_all(fd, hdr, hdr_len(n, sizeof(hdr))) == 0)
            (void)net_write_all(fd, body, strlen(body));
    }
    free(body);
}
