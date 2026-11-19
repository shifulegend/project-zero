#ifndef TN_API_HTTP_HEADERS_H
#define TN_API_HTTP_HEADERS_H

#include <stddef.h>

/* Framing-relevant request headers, parsed strictly (request-smuggling safe):
 * Content-Length must be plain decimal digits, appear at most once, and fit in
 * size_t; Transfer-Encoding is detected so the caller can refuse it. */
typedef struct {
    int    has_content_length;
    size_t content_length;
    int    content_length_invalid; /* non-digits, overflow, empty, or repeated with another value */
    int    chunked;                /* Transfer-Encoding mentions "chunked" */
    int    has_transfer_encoding;
} HttpFraming;

/* Parse a Content-Length header value ("  123  \r\n"). 0 = ok, -1 = invalid. */
int http_parse_content_length(const char *value, size_t *out);

/* Scan the header block (request line + headers, up to but not including the
 * blank line) of `len` bytes. Only text at the start of a line is considered,
 * so a body or a longer header name containing "content-length:" never matches. */
void http_scan_framing(const char *hdrs, size_t len, HttpFraming *out);

#endif
