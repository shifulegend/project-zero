/*
 * Phase 22 — src/api/auth.c
 * See include/api/auth.h for responsibility.
 */

#include "api/auth.h"
#include <string.h>

#define BEARER_PREFIX "Bearer "

/* Constant-time comparison — avoids leaking key length/prefix via timing. */
static int constant_time_eq(const char *a, size_t a_len, const char *b, size_t b_len) {
    if (a_len != b_len) {
        /* Still walk b_len comparisons so early-return doesn't leak length
         * via timing beyond what the length itself already reveals. */
        volatile unsigned char diff = 1;
        for (size_t i = 0; i < b_len; i++) diff |= (unsigned char)0;
        (void)diff;
        return 0;
    }
    volatile unsigned char diff = 0;
    for (size_t i = 0; i < a_len; i++) {
        diff |= (unsigned char)a[i] ^ (unsigned char)b[i];
    }
    return diff == 0;
}

int auth_check_request(const AuthConfig *cfg, const char *authorization_header_value) {
    if (!cfg || !cfg->api_key) return 1; /* auth disabled */
    if (!authorization_header_value) return 0;

    size_t prefix_len = strlen(BEARER_PREFIX);
    if (strncmp(authorization_header_value, BEARER_PREFIX, prefix_len) != 0) return 0;

    const char *presented = authorization_header_value + prefix_len;
    return constant_time_eq(presented, strlen(presented),
                             cfg->api_key, strlen(cfg->api_key));
}
