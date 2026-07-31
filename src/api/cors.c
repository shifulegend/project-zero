/*
 * Phase 22 — src/api/cors.c
 * See include/api/cors.h for responsibility.
 */

#include "api/cors.h"
#include <string.h>
#include <stdio.h>

int cors_is_origin_allowed(const CorsConfig *cfg, const char *origin_hdr) {
    if (!cfg || !cfg->enabled || !origin_hdr) return 0;
    for (int i = 0; i < cfg->num_origins; i++) {
        if (strcmp(cfg->origins[i], "*") == 0) return 1;
        if (strcmp(cfg->origins[i], origin_hdr) == 0) return 1;
    }
    return 0;
}

size_t cors_format_headers(const CorsConfig *cfg, const char *origin_hdr,
                            char *buf, size_t cap) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    if (!cors_is_origin_allowed(cfg, origin_hdr)) return 0;

    int n = snprintf(buf, cap,
        "Access-Control-Allow-Origin: %s\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Vary: Origin\r\n",
        origin_hdr);
    if (n < 0) { buf[0] = '\0'; return 0; }
    if ((size_t)n >= cap) { buf[0] = '\0'; return 0; } /* truncated — drop rather than send malformed headers */
    return (size_t)n;
}
