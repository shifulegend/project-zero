#ifndef TN_CLI_MD_RENDER_H
#define TN_CLI_MD_RENDER_H

/*
 * Phase 22.3 — Incremental markdown rendering for streamed token pieces.
 *
 * Handles three constructs: fenced code blocks (```...```), inline code
 * (`...`), and bold (**...**). Deliberately not a full CommonMark engine —
 * streamed tokens can split a construct's delimiters across pieces (e.g.
 * "**bo" then "ld**"), so input is buffered until a construct closes.
 * Markdown syntax is always stripped from the rendered output; ANSI color
 * is added only when color_enabled is set, so output is meaningful either
 * way (this is what makes md_render_feed/flush's output independently
 * testable without a real terminal — write to any FILE*, e.g.
 * open_memstream(), and compare the captured text).
 */

#include <stddef.h>
#include <stdio.h>

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
    FILE *out;
    int color_enabled;
} MdRenderState;

/* out: destination stream (e.g. stdout, or open_memstream() in tests).
 * color_enabled: whether to wrap rendered constructs in ANSI codes. */
void md_render_init(MdRenderState *st, FILE *out, int color_enabled);
void md_render_free(MdRenderState *st);

/* Feed one token piece. Renders any newly-complete constructs immediately;
 * buffers trailing text that might be the start of an incomplete construct. */
void md_render_feed(MdRenderState *st, const char *piece);

/* Emits any buffered (incomplete) text as-is (raw, unprocessed) — call once
 * generation finishes so nothing is silently dropped. */
void md_render_flush(MdRenderState *st);

#endif /* TN_CLI_MD_RENDER_H */
