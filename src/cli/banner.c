/*
 * Phase 22.5 — src/cli/banner.c
 * See include/cli/banner.h for responsibility.
 *
 * A tiny built-in 5-row block font (only the letters "PROJECT ZERO" needs)
 * rather than depending on an external figlet/font file — composed
 * programmatically from per-letter glyphs so the concatenation itself
 * can't introduce alignment bugs; only the individual glyphs need to be
 * correct.
 */

#include "cli/banner.h"
#include "cli/color.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define GLYPH_ROWS 5

typedef struct { const char *rows[GLYPH_ROWS]; int width; } Glyph;

static const Glyph GLYPH_P = { { "####.", "#..#.", "####.", "#....", "#...." }, 5 };
static const Glyph GLYPH_R = { { "####.", "#..#.", "####.", "#.#..", "#..#." }, 5 };
static const Glyph GLYPH_O = { { ".###.", "#...#", "#...#", "#...#", ".###." }, 5 };
static const Glyph GLYPH_J = { { "..###", "...#.", "...#.", "#..#.", ".##.." }, 5 };
static const Glyph GLYPH_E = { { "#####", "#....", "###..", "#....", "#####" }, 5 };
static const Glyph GLYPH_C = { { ".####", "#....", "#....", "#....", ".####" }, 5 };
static const Glyph GLYPH_T = { { "#####", "..#..", "..#..", "..#..", "..#.." }, 5 };
static const Glyph GLYPH_Z = { { "#####", "...#.", "..#..", ".#...", "#####" }, 5 };
static const Glyph GLYPH_SPACE = { { "...", "...", "...", "...", "..." }, 3 };

static const Glyph *glyph_for(char c) {
    switch (c) {
        case 'P': return &GLYPH_P;
        case 'R': return &GLYPH_R;
        case 'O': return &GLYPH_O;
        case 'J': return &GLYPH_J;
        case 'E': return &GLYPH_E;
        case 'C': return &GLYPH_C;
        case 'T': return &GLYPH_T;
        case 'Z': return &GLYPH_Z;
        case ' ': return &GLYPH_SPACE;
        default:  return NULL; /* unsupported char: skipped */
    }
}

#define BANNER_TEXT   "PROJECT ZERO"
#define BANNER_BUF_CAP 128

/* Composes GLYPH_ROWS output strings (one per font row) by looking up each
 * character's glyph and concatenating its row with a 1-column gap between
 * letters. buf must provide GLYPH_ROWS buffers of at least BANNER_BUF_CAP
 * bytes each. */
static void compose_banner(char buf[GLYPH_ROWS][BANNER_BUF_CAP]) {
    for (int r = 0; r < GLYPH_ROWS; r++) buf[r][0] = '\0';

    for (const char *p = BANNER_TEXT; *p; p++) {
        const Glyph *g = glyph_for(*p);
        if (!g) continue;
        for (int r = 0; r < GLYPH_ROWS; r++) {
            strncat(buf[r], g->rows[r], BANNER_BUF_CAP - strlen(buf[r]) - 1);
            /* Separator must be '.' (blank), not a literal space — print_glyph_row
             * only treats '.' as blank and renders anything else as '#', so a
             * literal " " here would render as a solid block instead of a gap. */
            if (p[1]) strncat(buf[r], ".", BANNER_BUF_CAP - strlen(buf[r]) - 1);
        }
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

void tn_banner_print(int is_tty, int color_enabled) {
    if (!is_tty) return;

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
