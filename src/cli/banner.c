/*
 * Phase 22.5 — src/cli/banner.c
 * See include/cli/banner.h for responsibility.
 *
 * The 5-row block font lives in brand/block_font.c (shared with the web UI
 * logo); this file only animates it.
 */

#include "cli/banner.h"
#include "cli/color.h"
#include "brand/block_font.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define GLYPH_ROWS TN_BLOCK_FONT_ROWS

#define BANNER_TEXT   "PROJECT ZERO"
#define BANNER_BUF_CAP 128

/* Composes GLYPH_ROWS output strings (one per font row) from the shared
 * block font (brand/block_font.c — also the web UI logo), keeping its '.'
 * blank marker. buf must provide GLYPH_ROWS buffers of at least
 * BANNER_BUF_CAP bytes each. */
static void compose_banner(char buf[GLYPH_ROWS][BANNER_BUF_CAP]) {
    char all[GLYPH_ROWS * BANNER_BUF_CAP];
    for (int r = 0; r < GLYPH_ROWS; r++) buf[r][0] = '\0';
    if (!tn_block_font_render(BANNER_TEXT, all, sizeof(all), 1)) return;
    const char *p = all;
    for (int r = 0; r < GLYPH_ROWS && *p; r++) {
        size_t len = strcspn(p, "\n");
        if (len >= BANNER_BUF_CAP) len = BANNER_BUF_CAP - 1;
        memcpy(buf[r], p, len);
        buf[r][len] = '\0';
        p += strcspn(p, "\n");
        if (*p == '\n') p++;
    }
}

/* Renders a composed glyph row as visible block characters (color_enabled)
 * or plain '#'/space (matches the glyph source directly) otherwise; '.' in
 * the glyph data means "blank" and is rendered as a space either way. dim
 * selects DIM instead of BOLD intensity, used for the post-reveal pulse. */
static void print_glyph_row(const char *row, int color_enabled, int dim) {
    if (color_enabled) fputs(tn_color(1, dim ? TN_ANSI_DIM : TN_ANSI_BOLD), stdout);
    if (color_enabled) fputs(tn_color(1, TN_ANSI_GREEN), stdout);
    for (const char *c = row; *c; c++) {
        fputc(*c == '.' ? ' ' : '#', stdout);
    }
    if (color_enabled) fputs(TN_ANSI_RESET, stdout);
}

/* Redraws every row of the reserved block in one intensity, then moves the
 * cursor back to the top — shared by the reveal loop and the post-reveal
 * pulse so both redraw identically. */
static void redraw_block(char rows[GLYPH_ROWS][BANNER_BUF_CAP], int first_visible,
                          int color_enabled, int dim) {
    printf("\x1b[%dA", GLYPH_ROWS); /* cursor up to the top of the reserved block */
    for (int row = 0; row < GLYPH_ROWS; row++) {
        if (row >= first_visible) print_glyph_row(rows[row], color_enabled, dim);
        fputs("\x1b[K\n", stdout); /* clear any leftover chars from a previous frame, then newline */
    }
    fflush(stdout);
}

static void banner_sleep_ms(long ms) {
    struct timespec ts = { .tv_sec = 0, .tv_nsec = ms * 1000000L };
    nanosleep(&ts, NULL);
}

size_t tn_banner_format_plain(char *buf, size_t cap) {
    if (!buf || cap == 0) return 0;

    char rows[GLYPH_ROWS][BANNER_BUF_CAP];
    compose_banner(rows);

    size_t n = 0;
    for (int r = 0; r < GLYPH_ROWS; r++) {
        for (const char *c = rows[r]; *c; c++) {
            if (n + 1 >= cap) { buf[n] = '\0'; return n; }
            buf[n++] = (*c == '.') ? ' ' : '#';
        }
        if (n + 1 >= cap) { buf[n] = '\0'; return n; }
        buf[n++] = '\n';
    }
    buf[n] = '\0';
    return n;
}

void tn_banner_print(int is_tty, int color_enabled) {
    if (!is_tty) {
        char buf[TN_BANNER_PLAIN_BUF_CAP];
        tn_banner_format_plain(buf, sizeof(buf));
        fputs(buf, stdout);
        return;
    }

    char rows[GLYPH_ROWS][BANNER_BUF_CAP];
    compose_banner(rows);

    /* Reserve GLYPH_ROWS blank lines, then redraw the whole reserved block
     * once per frame with an increasing number of rows revealed from the
     * bottom upward — new content appears to rise into place above what's
     * already shown, rather than the (less portable) terminal insert-line
     * sequence. Plain cursor-up + per-line clear+redraw works identically
     * across real terminals and xterm.js alike. */
    for (int i = 0; i < GLYPH_ROWS; i++) putchar('\n');

    for (int frame = 1; frame <= GLYPH_ROWS; frame++) {
        int first_visible = GLYPH_ROWS - frame;
        redraw_block(rows, first_visible, color_enabled, /*dim=*/0);
        /* ~45ms per frame; skip the wait after the last one. nanosleep (not
         * usleep) — usleep was removed from POSIX.1-2008 and isn't declared
         * under this project's _POSIX_C_SOURCE=200809L. */
        if (frame < GLYPH_ROWS) banner_sleep_ms(45);
    }

    /* A brief "shimmer" flourish (dim <-> bold) once fully revealed, rather
     * than freezing the instant the last row lands — a bounded animation
     * (not an indefinite loop: the REPL immediately blocks on stdin for the
     * first prompt afterward, so there's no way to keep animating in the
     * background without a dedicated thread, out of scope for a startup
     * flourish). Only visible when color is enabled — with color off the
     * glyphs render identically either way, so pulsing would be a no-op. */
    if (color_enabled) {
        for (int pulse = 0; pulse < 3; pulse++) {
            banner_sleep_ms(90);
            redraw_block(rows, 0, color_enabled, /*dim=*/1);
            banner_sleep_ms(90);
            redraw_block(rows, 0, color_enabled, /*dim=*/0);
        }
    }
    putchar('\n');
}
