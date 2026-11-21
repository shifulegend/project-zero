/*
 * src/api/http_request.c
 *
 * Request side of the HTTP server: reading a request off a socket with strict framing, and the
 * minimal HTTP/1.1 request parser. Split out of http_server.c; the code is unchanged apart from
 * returning a status instead of answering on the socket.
 */

#include "api/http_request.h"

#include <ctype.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>

#include "api/http_headers.h"
#include "api/http_status.h"

/* Copies a header value starting at *p up to (not including) \r or \n into
 * dst (NUL-terminated, truncated to dst_cap - 1). */
static void copy_header_value(const char *p, char *dst, size_t dst_cap) {
    size_t i = 0;
    while (*p == ' ') p++;
    while (*p && *p != '\r' && *p != '\n' && i < dst_cap - 1) dst[i++] = *p++;
    dst[i] = '\0';
}

int http_request_id_valid(const char *id) {
    size_t n = id ? strlen(id) : 0;
    if (n == 0 || n >= HTTP_REQID_MAX) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)id[i];
        if (!(isalnum(c) || c == '.' || c == '_' || c == ':' || c == '-')) return 0;
    }
    return 1;
}

/* ── The headers http_parse reads: one handler per name, matched case-insensitively ─────── */

static void on_content_length(const char *value, HttpRequest *req) {
    size_t cl;
    if (http_parse_content_length(value, &cl) == 0 && cl <= (size_t)INT_MAX)
        req->content_length =
            (int)cl; /* invalid values were already rejected in handle_connection */
}

static void on_origin(const char *value, HttpRequest *req) {
    copy_header_value(value, req->origin, sizeof(req->origin));
    req->has_origin = 1;
}

static void on_authorization(const char *value, HttpRequest *req) {
    copy_header_value(value, req->authorization, sizeof(req->authorization));
    req->has_authorization = 1;
}

static void on_content_type(const char *value, HttpRequest *req) {
    copy_header_value(value, req->content_type, sizeof(req->content_type));
}

static void on_request_id(const char *value, HttpRequest *req) {
    char id[HTTP_REQID_MAX + 8]; /* longer than allowed, so an over-long ID is rejected */
    copy_header_value(value, id, sizeof(id));
    if (http_request_id_valid(id)) memcpy(req->request_id, id, strlen(id) + 1);
}

static void on_host(const char *value, HttpRequest *req) {
    copy_header_value(value, req->host, sizeof(req->host));
}

static const struct {
    const char *name; /* with its colon */
    size_t len;
    void (*apply)(const char *value, HttpRequest *req);
} k_headers[] = {
    {"content-length:", 15, on_content_length}, {"origin:", 7, on_origin},
    {"authorization:", 14, on_authorization},   {"content-type:", 13, on_content_type},
    {"x-request-id:", 13, on_request_id},       {"host:", 5, on_host},
};

/* Applies the header on the line at p (if it is one we read). */
static void parse_header_line(const char *p, HttpRequest *req) {
    for (size_t k = 0; k < sizeof(k_headers) / sizeof(k_headers[0]); k++) {
        if (strncasecmp(p, k_headers[k].name, k_headers[k].len) != 0) continue;
        k_headers[k].apply(p + k_headers[k].len, req);
        return;
    }
}

/* Copies a token from p up to a space (or, with stop_at_cr, a CR) into dst, truncated to cap - 1;
 * returns where it stopped. */
static const char *copy_token(const char *p, char *dst, size_t cap, int stop_at_cr) {
    size_t i = 0;
    while (*p && *p != ' ' && !(stop_at_cr && *p == '\r') && i < cap - 1) dst[i++] = *p++;
    dst[i] = '\0';
    return p;
}

/* Method and path; the rest of the request line is skipped. Returns the start of the first header
 * line, or NULL when the line is not "METHOD PATH ...". */
static const char *parse_request_line(const char *buf, HttpRequest *req) {
    const char *p = copy_token(buf, req->method, sizeof(req->method), 0);
    if (*p != ' ') return NULL;
    p = copy_token(p + 1, req->path, sizeof(req->path), 1);
    if (*p != ' ') return NULL;

    while (*p && *p != '\n') p++; /* skip to end of request line */
    if (*p) p++;
    return p;
}

/* Reads header lines until the blank line; returns where the body starts. */
static const char *parse_headers(const char *p, HttpRequest *req) {
    while (*p) {
        if (*p == '\r' && *(p + 1) == '\n') return p + 2;
        if (*p == '\n') return p + 1;
        parse_header_line(p, req);
        while (*p && *p != '\n') p++; /* skip to next line */
        if (*p) p++;
    }
    return p;
}

/*
 * Parse a minimal HTTP/1.1 request from a NUL-terminated buffer.
 * Returns 1 on success, 0 on parse failure.
 */
int http_parse(const char *buf, size_t buf_len, HttpRequest *req) {
    memset(req, 0, sizeof(*req));
    req->content_length = -1;

    const char *p = parse_request_line(buf, req);
    if (!p) return 0;
    p = parse_headers(p, req);

    req->body     = (char *)p;
    req->body_len = (buf_len > (size_t)(p - buf)) ? buf_len - (size_t)(p - buf) : 0;
    return 1;
}


/* Reads one request from fd into a malloc'd, NUL-terminated buffer. Returns 0 with *buf and *len
 * set (a short or closed read still returns what arrived; http_parse() decides if it is usable),
 * a positive HTTP status with *err set when the request must be refused before any body is read
 * (bad or duplicate Content-Length 400, any Transfer-Encoding 501, body too large 413), or -1 when
 * out of memory. On a nonzero return nothing is left allocated. */
int http_request_read(int fd, char **buf, size_t *len, const char **err) {
    char *recv_buf = (char *)malloc(HTTP_RECV_BUF);
    if (!recv_buf) return -1;
    size_t total = 0;
    size_t cap   = HTTP_RECV_BUF;

    /* Read until we have headers + full body */
    int headers_done   = 0;
    int content_length = -1;
    size_t header_end  = 0;

    while (total < HTTP_MAX_BODY_BYTES + 4096) {
        ssize_t n = recv(fd, recv_buf + total, cap - total - 1, 0);
        if (n <= 0) break;
        total += (size_t)n;
        recv_buf[total] = '\0';

        if (!headers_done) {
            char *hdrs_end = strstr(recv_buf, "\r\n\r\n");
            if (!hdrs_end) hdrs_end = strstr(recv_buf, "\n\n");
            if (hdrs_end) {
                headers_done = 1;
                header_end = (size_t)(hdrs_end - recv_buf) + (strstr(recv_buf, "\r\n\r\n") ? 4 : 2);

                /* Strict framing: Content-Length is scanned only inside the
                 * header block (never the body), must be plain digits, and
                 * conflicting duplicates are refused. Chunked bodies are not
                 * implemented, so they are refused instead of being read as
                 * an empty or truncated body. */
                HttpFraming fr;
                http_scan_framing(recv_buf, header_end, &fr);
                if (fr.content_length_invalid) {
                    free(recv_buf);
                    *err = "Invalid Content-Length";
                    return HTTP_BAD_REQUEST;
                }
                if (fr.has_transfer_encoding) { /* chunked, gzip, ...: none is implemented */
                    free(recv_buf);
                    *err = "Transfer-Encoding is not supported; send Content-Length";
                    return HTTP_NOT_IMPLEMENTED;
                }
                if (fr.has_content_length) {
                    if (fr.content_length > HTTP_MAX_BODY_BYTES) {
                        free(recv_buf);
                        *err = "Request body too large";
                        return HTTP_PAYLOAD_TOO_LARGE;
                    }
                    content_length = (int)fr.content_length;
                }

                /* 2026-07-15: pre-existing bug — a request with no
                 * Content-Length (the normal case for GET/HEAD/OPTIONS, and
                 * any bodyless POST) has content_length == -1 forever, so
                 * the old "stop once we have the full body" check below
                 * never fired and recv() blocked indefinitely waiting for
                 * bytes that were never coming (client already sent its
                 * full request and is waiting on OUR response). No
                 * Content-Length means no body is expected — stop reading
                 * as soon as headers are complete. */
                if (content_length < 0) break;

                /* If we already have the full body, stop reading */
                if (total >= header_end + (size_t)content_length) break;
            }
        } else {
            if (content_length >= 0 && total >= header_end + (size_t)content_length) break;
        }

        /* Grow buffer if needed */
        if (cap - total < 1024) {
            size_t new_cap = cap * 2;
            if (new_cap > HTTP_MAX_BODY_BYTES + 8192) {
                free(recv_buf);
                *err = "Request body too large";
                return HTTP_PAYLOAD_TOO_LARGE;
            }
            char *new_buf = (char *)realloc(recv_buf, new_cap);
            if (!new_buf) break;
            recv_buf = new_buf;
            cap      = new_cap;
        }
    }
    recv_buf[total] = '\0';
    *buf            = recv_buf;
    *len            = total;
    return 0;
}

/* ── Unique and correlation IDs ──────────────────────────────────────────── */

void http_new_id(atomic_ulong *seq, char *buf, size_t cap) {
    unsigned long n = atomic_fetch_add(seq, 1) + 1;
    snprintf(buf, cap, "%lx-%lx", (unsigned long)time(NULL), n);
}

void http_request_attach_id(atomic_ulong *seq, HttpRequest *req, char *headers, size_t cap) {
    if (!req->request_id[0]) http_new_id(seq, req->request_id, sizeof(req->request_id));
    size_t used = strlen(headers);
    snprintf(headers + used, cap - used, "X-Request-ID: %s\r\n", req->request_id);
}
