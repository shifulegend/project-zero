#ifndef TN_API_HTTP_REQUEST_H
#define TN_API_HTTP_REQUEST_H

/* Reading and parsing one HTTP/1.1 request (src/api/http_request.c). */

#include <stddef.h>

#define HTTP_RECV_BUF 8192                    /* initial receive buffer per connection */
#define HTTP_MAX_BODY_BYTES (8 * 1024 * 1024) /* 8 MiB: fits base64 image uploads */

#define HTTP_PATH_MAX 256   /* request path */
#define HTTP_ORIGIN_MAX 256 /* Origin header */
#define HTTP_AUTH_MAX 512   /* Authorization header */
#define HTTP_CTYPE_MAX 128  /* Content-Type header */
#define HTTP_HOST_MAX 256   /* Host header */
#define HTTP_REQID_MAX 65   /* X-Request-ID value (64 characters) plus NUL */

typedef struct {
    char method[16];
    char path[HTTP_PATH_MAX];
    char *body; /* points into the receive buffer */
    size_t body_len;
    int content_length; /* from Content-Length header, -1 if not present */
    char origin[HTTP_ORIGIN_MAX];
    char authorization[HTTP_AUTH_MAX];
    char content_type[HTTP_CTYPE_MAX];
    char host[HTTP_HOST_MAX];
    char request_id[HTTP_REQID_MAX]; /* accepted X-Request-ID, "" when absent or invalid */
    int has_origin;
    int has_authorization;
} HttpRequest;

/* 1 when id is a usable correlation ID: 1..64 characters from [A-Za-z0-9._:-]. Anything else (spaces,
 * control characters, quotes) is refused so the value can be echoed in a header and logged safely. */
int http_request_id_valid(const char *id);

/* A process-unique ID ("<hex time>-<hex counter>"), thread-safe. */
void http_new_id(char *buf, size_t cap);

/* Gives req an ID if the client sent none and appends "X-Request-ID: <id>\r\n" to the response
 * header block `headers` (capacity cap), so every reply carries it. */
void http_request_attach_id(HttpRequest *req, char *headers, size_t cap);

/* Parses a NUL-terminated request in buf. Returns 1 on success, 0 on a malformed request. */
int http_parse(const char *buf, size_t buf_len, HttpRequest *req);

/* Reads one request from fd into a malloc'd, NUL-terminated buffer (*buf, *len). Returns 0, a
 * positive HTTP status to answer with *err as the message, or -1 when out of memory. */
int http_request_read(int fd, char **buf, size_t *len, const char **err);

#endif
