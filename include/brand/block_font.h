#ifndef TN_BLOCK_FONT_H
#define TN_BLOCK_FONT_H

/*
 * block_font.h — the 5-row block font of the Project Zero wordmark, shared by
 * the CLI startup banner (cli/banner.c) and the web UI logo (served through
 * /v1/config ui.logo_ascii) so both always render the same glyphs.
 *
 * Rows use '#' for ink and ' ' for blank; letters are joined by one blank
 * column. Letters are case-insensitive; '-' and '_' render as a word gap, so
 * the default brand "project-zero" yields the "PROJECT ZERO" wordmark.
 */

#include <stddef.h>

#define TN_BLOCK_FONT_ROWS 5

/* 1 if every character of `text` has a glyph (A-Z subset, space, '-', '_'). */
int tn_block_font_supports(const char *text);

/* Writes the TN_BLOCK_FONT_ROWS rows, each ending in '\n', into buf.
 * `raw` = 1 keeps the internal '.' blank marker instead of ' '.
 * Returns bytes written (excl. NUL), or 0 if text is empty, has an
 * unsupported character, or buf is too small. */
size_t tn_block_font_render(const char *text, char *buf, size_t cap, int raw);

#endif /* TN_BLOCK_FONT_H */
