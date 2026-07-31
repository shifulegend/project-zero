/*
 * Phase 22.3 — src/cli/md_render.c
 * See include/cli/md_render.h for responsibility.
 */

#include "cli/md_render.h"
#include "cli/color.h"
#include <stdlib.h>
#include <string.h>

#define MD_INITIAL_CAP 256

/* Safety valve: if an opening marker (code fence/bold/inline-code) hasn't
 * found its closing counterpart within this many buffered bytes, stop
 * waiting and emit everything as plain text instead. Without this, a
 * generation that opens a code fence and never closes it (e.g. max_tokens
 * cuts it off mid-block — common in practice) would buffer the ENTIRE rest
 * of the response and only show it at the final flush, defeating live
 * streaming. 4 KiB comfortably covers real code blocks while bounding the
 * worst case. */
#define MD_MAX_PENDING_UNCLOSED 4096

void md_render_init(MdRenderState *st, FILE *out, int color_enabled) {
    st->buf = (char *)malloc(MD_INITIAL_CAP);
    st->len = 0;
    st->cap = st->buf ? MD_INITIAL_CAP : 0;
    st->out = out;
    st->color_enabled = color_enabled;
}

void md_render_free(MdRenderState *st) {
    free(st->buf);
    st->buf = NULL;
    st->len = st->cap = 0;
}

static int md_ensure_cap(MdRenderState *st, size_t extra) {
    if (st->len + extra <= st->cap) return 1;
    size_t new_cap = st->cap == 0 ? MD_INITIAL_CAP : st->cap * 2;
    while (new_cap < st->len + extra) new_cap *= 2;
    char *nb = (char *)realloc(st->buf, new_cap);
    if (!nb) return 0;
    st->buf = nb;
    st->cap = new_cap;
    return 1;
}

/* Finds needle in buf[start..len), returns index or -1. */
static long md_find(const char *buf, size_t len, const char *needle, size_t nlen, size_t start) {
    if (nlen == 0 || len < nlen) return -1;
    for (size_t i = start; i + nlen <= len; i++) {
        if (memcmp(buf + i, needle, nlen) == 0) return (long)i;
    }
    return -1;
}

/* Finds the earliest standalone single backtick (i.e. not the start of a
 * "```" fence marker) at or after start. */
static long md_find_single_backtick(const char *buf, size_t len, size_t start) {
    for (size_t i = start; i < len; i++) {
        if (buf[i] != '`') continue;
        int is_fence = (i + 2 < len && buf[i+1] == '`' && buf[i+2] == '`');
        if (!is_fence) return (long)i;
        i += 2; /* skip the rest of the fence marker */
    }
    return -1;
}

static void md_emit_raw(MdRenderState *st, const char *s, size_t n) {
    if (n > 0) fwrite(s, 1, n, st->out);
}

static void md_emit_styled(MdRenderState *st, const char *s, size_t n, const char *ansi_code) {
    if (st->color_enabled) fputs(tn_color(1, ansi_code), st->out);
    md_emit_raw(st, s, n);
    if (st->color_enabled) fputs(TN_ANSI_RESET, st->out);
}

/* Removes the first n bytes from the pending buffer (already emitted). */
static void md_consume(MdRenderState *st, size_t n) {
    if (n >= st->len) { st->len = 0; return; }
    memmove(st->buf, st->buf + n, st->len - n);
    st->len -= n;
}

/* Processes as much of the pending buffer as can be resolved right now,
 * leaving anything that might still be an incomplete construct buffered. */
static void md_process(MdRenderState *st) {
    for (;;) {
        long pos_fence = md_find(st->buf, st->len, "```", 3, 0);
        long pos_bold  = md_find(st->buf, st->len, "**", 2, 0);
        long pos_code  = md_find_single_backtick(st->buf, st->len, 0);

        /* Pick the earliest candidate opening marker. */
        long pos = -1;
        size_t mark_len = 0;
        const char *marker = NULL;
        if (pos_fence >= 0 && (pos < 0 || pos_fence < pos)) { pos = pos_fence; mark_len = 3; marker = "```"; }
        if (pos_bold  >= 0 && (pos < 0 || pos_bold  < pos)) { pos = pos_bold;  mark_len = 2; marker = "**"; }
        if (pos_code  >= 0 && (pos < 0 || pos_code  < pos)) { pos = pos_code;  mark_len = 1; marker = "`"; }

        if (pos < 0) {
            /* No opening marker anywhere in the buffer. Emit everything
             * except a small safety tail, in case the next piece continues
             * a multi-char marker split across the boundary (worst case:
             * "``" pending, "`" arrives next to complete "```"). */
            size_t hold_back = (st->len > 2) ? 2 : st->len;
            size_t emit_len = st->len - hold_back;
            md_emit_raw(st, st->buf, emit_len);
            md_consume(st, emit_len);
            return;
        }

        long close = md_find(st->buf, st->len, marker, mark_len, (size_t)pos + mark_len);
        if (close < 0) {
            size_t unclosed_len = st->len - (size_t)pos;
            if (unclosed_len > MD_MAX_PENDING_UNCLOSED) {
                /* This construct has been open too long to keep waiting —
                 * give up and emit everything buffered as plain text so
                 * streaming isn't stalled indefinitely. */
                md_emit_raw(st, st->buf, st->len);
                md_consume(st, st->len);
                return;
            }
            /* Opening marker found but no closing one yet — emit anything
             * before it, keep from the marker onward buffered. */
            md_emit_raw(st, st->buf, (size_t)pos);
            md_consume(st, (size_t)pos);
            return;
        }

        /* Complete construct: [pos, pos+mark_len) .. [close, close+mark_len) */
        md_emit_raw(st, st->buf, (size_t)pos);
        size_t content_start = (size_t)pos + mark_len;
        size_t content_len   = (size_t)close - content_start;
        const char *style = (mark_len == 3) ? TN_ANSI_GREEN
                           : (mark_len == 2) ? TN_ANSI_BOLD
                           : TN_ANSI_CYAN;
        md_emit_styled(st, st->buf + content_start, content_len, style);
        md_consume(st, (size_t)close + mark_len);
        /* Loop again — more constructs may follow in the remaining buffer. */
    }
}

void md_render_feed(MdRenderState *st, const char *piece) {
    if (!piece) return;
    size_t plen = strlen(piece);
    if (plen == 0) return;
    if (!md_ensure_cap(st, plen)) {
        /* OOM: fail safe by flushing raw and dropping buffering for this piece. */
        md_emit_raw(st, st->buf, st->len);
        st->len = 0;
        md_emit_raw(st, piece, plen);
        return;
    }
    memcpy(st->buf + st->len, piece, plen);
    st->len += plen;
    md_process(st);
    fflush(st->out);
}

void md_render_flush(MdRenderState *st) {
    md_emit_raw(st, st->buf, st->len);
    st->len = 0;
    fflush(st->out);
}
