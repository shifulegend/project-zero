#ifndef TN_CORS_H
#define TN_CORS_H

/*
 * Phase 22 — CORS (Cross-Origin Resource Sharing) support.
 * Off by default (CorsConfig.enabled == 0) so existing loopback workflows
 * are unaffected; enabled via --cors / --cors-origin.
 */

#include <stddef.h>
#include "api/server_config.h"

/* Returns 1 if origin_hdr is allowed by cfg (or cfg has a "*" entry),
 * 0 otherwise. origin_hdr may be NULL (request had no Origin header). */
int cors_is_origin_allowed(const CorsConfig *cfg, const char *origin_hdr);

/* Formats Access-Control-Allow-* header lines (each "\r\n"-terminated, no
 * trailing blank line) into buf, so the caller can splice them into an
 * existing HTTP header block ahead of the blank line. Writes nothing (and
 * returns 0) when CORS is disabled or origin_hdr isn't allowed. Returns the
 * number of bytes written into buf (excluding the NUL terminator). */
size_t cors_format_headers(const CorsConfig *cfg, const char *origin_hdr,
                            char *buf, size_t cap);

#endif /* TN_CORS_H */
