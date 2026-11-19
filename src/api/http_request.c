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

/*
 * Parse a minimal HTTP/1.1 request from a NUL-terminated buffer.
 * Returns 1 on success, 0 on parse failure.
 */
int http_request_id_valid(const char *id) {
    size_t n = id ? strlen(id) : 0;
    if (n == 0 || n >= HTTP_REQID_MAX) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)id[i];
        if (!(isalnum(c) || c == '.' || c == '_' || c == ':' || c == '-')) return 0;
    }
    return 1;
}

int http_parse(const char *buf, size_t buf_len, HttpRequest *req) {
    memset(req, 0, sizeof(*req));
    req->content_length = -1;

    /* Method */
    const char *p = buf;
    size_t i      = 0;
    while (*p && *p != ' ' && i < sizeof(req->method) - 1) req->method[i++] = *p++;
    req->method[i] = '\0';
    if (*p != ' ') return 0;
    p++;

    /* Path */
    i = 0;
    while (*p && *p != ' ' && *p != '\r' && i < sizeof(req->path) - 1) req->path[i++] = *p++;
    req->path[i] = '\0';
    if (*p != ' ') return 0;

    /* Skip to end of request line */
    while (*p && *p != '\n') p++;
    if (*p) p++;

    /* Parse headers until blank line */
    while (*p) {
        if (*p == '\r' && *(p + 1) == '\n') {
            p += 2;
            break;
        }
        if (*p == '\n') {
            p++;
            break;
        }

        if (strncasecmp(p, "content-length:", 15) == 0) {
            size_t cl;
            if (http_parse_content_length(p + 15, &cl) == 0 && cl <= (size_t)INT_MAX)
                req->content_length =
                    (int)cl; /* invalid values were already rejected in handle_connection */
        } else if (strncasecmp(p, "origin:", 7) == 0) {
            copy_header_value(p + 7, req->origin, sizeof(req->origin));
            req->has_origin = 1;
        } else if (strncasecmp(p, "authorization:", 14) == 0) {
            copy_header_value(p + 14, req->authorization, sizeof(req->authorization));
            req->has_authorization = 1;
        } else if (strncasecmp(p, "content-type:", 13) == 0) {
            copy_header_value(p + 13, req->content_type, sizeof(req->content_type));
        } else if (strncasecmp(p, "x-request-id:", 13) == 0) {
            char id[HTTP_REQID_MAX + 8]; /* longer than allowed, so an over-long ID is rejected */
            copy_header_value(p + 13, id, sizeof(id));
            if (http_request_id_valid(id)) memcpy(req->request_id, id, strlen(id) + 1);
        } else if (strncasecmp(p, "host:", 5) == 0) {
            copy_header_value(p + 5, req->host, sizeof(req->host));
        }
        /* Skip to next line */
        while (*p && *p != '\n') p++;
        if (*p) p++;
    }

    /* Body starts at p */
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

void http_new_id(char *buf, size_t cap) {
    static pthread_mutex_t id_lock = PTHREAD_MUTEX_INITIALIZER;
    static unsigned long counter   = 0;
    pthread_mutex_lock(&id_lock);
    unsigned long n = ++counter;
    pthread_mutex_unlock(&id_lock);
    snprintf(buf, cap, "%lx-%lx", (unsigned long)time(NULL), n);
}

void http_request_attach_id(HttpRequest *req, char *headers, size_t cap) {
    if (!req->request_id[0]) http_new_id(req->request_id, sizeof(req->request_id));
    size_t used = strlen(headers);
    snprintf(headers + used, cap - used, "X-Request-ID: %s\r\n", req->request_id);
}
