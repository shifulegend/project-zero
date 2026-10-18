#include "brand/block_font.h"
#include <ctype.h>
#include <string.h>

/* '.' = blank, '#' = ink. Moved here from cli/banner.c (2026-10-18) so the
 * web UI logo uses the exact same glyphs as the terminal banner. */
typedef struct { const char *rows[TN_BLOCK_FONT_ROWS]; int width; } Glyph;

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
    switch (toupper((unsigned char)c)) {
        case 'P': return &GLYPH_P;
        case 'R': return &GLYPH_R;
        case 'O': return &GLYPH_O;
        case 'J': return &GLYPH_J;
        case 'E': return &GLYPH_E;
        case 'C': return &GLYPH_C;
        case 'T': return &GLYPH_T;
        case 'Z': return &GLYPH_Z;
        case ' ': case '-': case '_': return &GLYPH_SPACE;
        default:  return NULL;
    }
}

int tn_block_font_supports(const char *text) {
    if (!text || !*text) return 0;
    for (const char *p = text; *p; p++)
        if (!glyph_for(*p)) return 0;
    return 1;
}

size_t tn_block_font_render(const char *text, char *buf, size_t cap, int raw) {
    if (!buf || cap == 0 || !tn_block_font_supports(text)) return 0;
    size_t n = 0;
    for (int r = 0; r < TN_BLOCK_FONT_ROWS; r++) {
        for (const char *p = text; *p; p++) {
            const char *row = glyph_for(*p)->rows[r];
            for (const char *c = row; *c; c++) {
                if (n + 2 >= cap) { buf[0] = '\0'; return 0; }
                buf[n++] = (raw || *c == '#') ? *c : ' ';
            }
            if (p[1]) {                       /* one blank column between letters */
                if (n + 2 >= cap) { buf[0] = '\0'; return 0; }
                buf[n++] = raw ? '.' : ' ';
            }
        }
        buf[n++] = '\n';
    }
    buf[n] = '\0';
    return n;
}
