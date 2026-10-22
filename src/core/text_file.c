/*
 * core/text_file.c — bounded whole-file reader (see core/text_file.h).
 */
#include "core/text_file.h"

#include <stdio.h>
#include <stdlib.h>

char *tn_read_text_file(const char *path, size_t max_bytes, size_t *len_out) {
    if (len_out) *len_out = 0;
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) return NULL;
    /* Read in chunks rather than trusting ftell: works for pipes too and
     * never allocates more than max_bytes + 1. */
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    while (buf) {
        if (len + 1 >= cap) {
            /* max_bytes + 2: room for one byte past the limit, so a file of
             * exactly max_bytes is accepted and a longer one detected. */
            size_t ncap = cap * 2 > max_bytes + 2 ? max_bytes + 2 : cap * 2;
            if (ncap <= cap) { free(buf); buf = NULL; break; }
            char *nb = (char *)realloc(buf, ncap);
            if (!nb) { free(buf); buf = NULL; break; }
            buf = nb;
            cap = ncap;
        }
        size_t rd = fread(buf + len, 1, cap - 1 - len, f);
        len += rd;
        if (len > max_bytes) { free(buf); buf = NULL; break; }
        if (rd == 0) break;
    }
    int err = ferror(f);
    fclose(f);
    if (!buf || err || len == 0) { free(buf); return NULL; }
    buf[len] = '\0';
    if (len_out) *len_out = len;
    return buf;
}
