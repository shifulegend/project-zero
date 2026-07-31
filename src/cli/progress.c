/*
 * Phase 22.3 — src/cli/progress.c
 * See include/cli/progress.h for responsibility.
 */

#include "cli/progress.h"
#include <stdio.h>

size_t tn_progress_format(char *buf, size_t cap, int stage, int total_stages,
                           const char *label, int is_tty) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    if (!label) label = "";

    int n;
    if (is_tty) {
        n = snprintf(buf, cap, "\r[%d/%d] %s\x1b[K", stage, total_stages, label);
    } else {
        n = snprintf(buf, cap, "[%d/%d] %s\n", stage, total_stages, label);
    }
    if (n < 0) { buf[0] = '\0'; return 0; }
    if ((size_t)n >= cap) return cap - 1;
    return (size_t)n;
}

void tn_progress_stage(int stage, int total_stages, const char *label, int is_tty) {
    char buf[256];
    tn_progress_format(buf, sizeof(buf), stage, total_stages, label, is_tty);
    fputs(buf, stderr);
    fflush(stderr);
}

void tn_progress_done(int is_tty) {
    if (is_tty) fputc('\n', stderr);
}
