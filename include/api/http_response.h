#ifndef TN_API_HTTP_RESPONSE_H
#define TN_API_HTTP_RESPONSE_H

/* Response writers of the HTTP server (src/api/http_response.c). Each writes one complete reply
 * to `fd` and closes nothing; a failed write (peer gone) is ignored. */

#define HTTP_HDR_BUF 1024     /* one response header block */
#define HTTP_ERR_BODY_BUF 256 /* JSON error body */

/* extra_headers, if non-NULL, must already be "\r\n"-terminated per line (e.g. from
 * cors_format_headers()) and is spliced in ahead of the blank line that ends the header block.
 * `streaming` sends only the SSE headers (the body is written by the SSE writers). */
void http_send_response_ex(int fd, int status, const char *content_type, const char *body,
                           int streaming, const char *extra_headers);

/* JSON error body {"error":{"message":msg,...}}. `msg` must not need JSON escaping. */
void http_send_error_ex(int fd, int status, const char *msg, const char *extra_headers);
void http_send_error(int fd, int status, const char *msg);

/* Sends a heap-allocated JSON body with `status` and frees it; NULL body answers 500 "OOM". */
void http_send_json_heap(int fd, int status, char *body, const char *extra_headers);

#endif
