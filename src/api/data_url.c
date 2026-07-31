/*
 * Phase 22.2 — src/api/data_url.c
 * See include/api/data_url.h for responsibility.
 */

#include "api/data_url.h"
#include <stdlib.h>
#include <string.h>

static int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1; /* padding '=' or invalid — caller stops at these */
}

TernaryError data_url_decode(const char *data_url, unsigned char **out_data, size_t *out_len) {
    if (!data_url || !out_data || !out_len) return TN_ERR_INVALID_ARGS;
    *out_data = NULL;
    *out_len = 0;

    if (strncmp(data_url, "data:", 5) != 0) return TN_ERR_INVALID_ARGS;

    const char *marker = strstr(data_url, ";base64,");
    if (!marker) return TN_ERR_INVALID_ARGS;
    const char *payload = marker + 8; /* skip ";base64," */

    size_t payload_len = strlen(payload);
    /* Strip trailing whitespace/newlines some clients add. */
    while (payload_len > 0 && (payload[payload_len - 1] == '\n' || payload[payload_len - 1] == '\r' ||
                               payload[payload_len - 1] == ' ')) {
        payload_len--;
    }
    if (payload_len == 0) return TN_ERR_INVALID_ARGS;

    unsigned char *buf = (unsigned char *)malloc(payload_len); /* decoded is always <= encoded length */
    if (!buf) return TN_ERR_OOM;

    size_t out_i = 0;
    int group[4];
    int group_n = 0;

    for (size_t i = 0; i < payload_len; i++) {
        unsigned char c = (unsigned char)payload[i];
        if (c == '=') break; /* padding — end of data */
        int v = b64_val(c);
        if (v < 0) continue; /* skip any stray whitespace/invalid char defensively */

        group[group_n++] = v;
        if (group_n == 4) {
            buf[out_i++] = (unsigned char)((group[0] << 2) | (group[1] >> 4));
            buf[out_i++] = (unsigned char)((group[1] << 4) | (group[2] >> 2));
            buf[out_i++] = (unsigned char)((group[2] << 6) | group[3]);
            group_n = 0;
        }
    }
    if (group_n == 2) {
        buf[out_i++] = (unsigned char)((group[0] << 2) | (group[1] >> 4));
    } else if (group_n == 3) {
        buf[out_i++] = (unsigned char)((group[0] << 2) | (group[1] >> 4));
        buf[out_i++] = (unsigned char)((group[1] << 4) | (group[2] >> 2));
    }

    *out_data = buf;
    *out_len = out_i;
    return TN_OK;
}
