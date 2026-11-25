#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "api/http_headers.h"

#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>

int http_parse_content_length(const char *v, size_t *out) {
    if (!v || !out) return -1;
    while (*v == ' ' || *v == '\t') v++;
    if (!isdigit((unsigned char)*v)) return -1; /* rejects "", "-1", "+5", "abc" */
    size_t val = 0;
    while (isdigit((unsigned char)*v)) {
        size_t d = (size_t)(*v - '0');
        if (val > (SIZE_MAX - d) / 10) return -1; /* overflow */
        val = val * 10 + d;
        v++;
    }
    while (*v == ' ' || *v == '\t') v++;
    if (*v != '\0' && *v != '\r' && *v != '\n') return -1; /* trailing junk: "12abc", "1 2" */
    *out = val;
    return 0;
}

void http_scan_framing(const char *hdrs, size_t len, HttpFraming *out) {
    memset(out, 0, sizeof(*out));
    const char *p = hdrs, *end = hdrs + len;
    while (p < end && *p != '\n') p++; /* skip the request line */
    if (p < end) p++;
    while (p < end) {
        const char *eol = memchr(p, '\n', (size_t)(end - p));
        size_t ll       = eol ? (size_t)(eol - p) : (size_t)(end - p);
        if (ll >= 15 && strncasecmp(p, "content-length:", 15) == 0) {
            char tmp[64];
            size_t vl = ll - 15;
            if (vl >= sizeof(tmp)) {
                out->content_length_invalid = 1;
            } else {
                memcpy(tmp, p + 15, vl);
                tmp[vl] = '\0';
                size_t v;
                if (http_parse_content_length(tmp, &v) != 0)
                    out->content_length_invalid = 1;
                else if (out->has_content_length && v != out->content_length)
                    out->content_length_invalid = 1;
                else {
                    out->has_content_length = 1;
                    out->content_length     = v;
                }
            }
        } else if (ll >= 18 && strncasecmp(p, "transfer-encoding:", 18) == 0) {
            out->has_transfer_encoding = 1;
            for (size_t i = 18; i + 7 <= ll; i++)
                if (strncasecmp(p + i, "chunked", 7) == 0) {
                    out->chunked = 1;
                    break;
                }
        }
        if (!eol) break;
        p = eol + 1;
    }
}
