#ifndef TN_CLI_BANNER_H
#define TN_CLI_BANNER_H

#include <stddef.h> /* size_t */

/*
 * Phase 22.5 — animated ASCII-art startup banner ("PROJECT ZERO"), the kind
 * of splash seen in Claude Code and other modern CLI tools. Shown for every
 * invocation, TTY or not (branding must never depend on the CLI's invocation
 * path) — animated only in a real terminal; piped/redirected output gets the
 * same glyph content as a single plain block with no ANSI escape codes.
 */

/* Pure formatter — no TTY/stream dependency, fully unit-testable. Renders
 * the plain (non-animated, no-ANSI) static banner block: the composed glyph
 * rows for "PROJECT ZERO", each row as '#'/' ' characters (matches
 * print_glyph_row's non-color rendering) with a single trailing '\n' per
 * row and no leading/trailing blank lines. Writes into buf (caller-provided,
 * at least TN_BANNER_PLAIN_BUF_CAP bytes) and returns the number of bytes
 * written (excluding the NUL terminator), or 0 if buf is NULL or cap is too
 * small to hold anything. */
#define TN_BANNER_PLAIN_BUF_CAP 768
size_t tn_banner_format_plain(char *buf, size_t cap);

/* Prints the banner. When is_tty is false, prints the plain static block
 * (via tn_banner_format_plain) once, with no cursor movement and no delay.
 * When is_tty is true, prints the banner with a line-by-line reveal, bottom
 * row first, each new row appearing above the previous (a "sliding up into
 * place" effect), followed by a brief bounded dim/bold "shimmer" pulse once
 * fully revealed (color_enabled only — with color off there is nothing to
 * pulse). color_enabled controls whether the glyphs are tinted with the
 * CLI's accent color (see cli/color.h); ignored in the non-tty plain path,
 * which never emits color codes at all. */
void tn_banner_print(int is_tty, int color_enabled);

#endif /* TN_CLI_BANNER_H */
