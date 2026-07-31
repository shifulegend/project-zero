/*
 * Phase 22.3 — src/cli/live_stats.c
 * See include/cli/live_stats.h for responsibility.
 */

#include "cli/live_stats.h"
#include "cli/color.h"
#include "cli/timer.h"
#include <stdio.h>

void tn_live_stats_init(TnLiveStats *ls) {
    if (!ls) return;
    ls->start_us = 0;
    ls->count = 0;
}

void tn_live_stats_tick(TnLiveStats *ls) {
    if (!ls) return;
    if (ls->count == 0) ls->start_us = timer_now_us();
    ls->count++;
}

/* Standard 10-frame braille spinner cycle (the same glyph set used by most
 * terminal spinner libraries) — one frame advance per token keeps it moving
 * for as long as generation is streaming, mirroring Claude Code's animated
 * "working" indicator. */
#define TN_SPINNER_FRAME_COUNT 10
static const char *const SPINNER_FRAMES[TN_SPINNER_FRAME_COUNT] = {
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
};

const char *tn_live_stats_spinner_frame(long long tick) {
    long long idx = tick % TN_SPINNER_FRAME_COUNT;
    if (idx < 0) idx += TN_SPINNER_FRAME_COUNT; /* defensive: tick should never be negative */
    return SPINNER_FRAMES[idx];
}

void tn_live_stats_render(const TnLiveStats *ls, int is_tty, int color_enabled) {
    if (!ls || !is_tty) return;
    if (ls->count < 2) return;
    double rate = timer_tokens_per_sec(ls->start_us, timer_now_us(), (int)ls->count);
    const char *spinner = tn_live_stats_spinner_frame(ls->count);
    /* 2026-07-15: a plain "\r...\x1b[K" (as originally written) overwrites
     * whatever the response text's own cursor currently shares that row
     * with — since streamed tokens don't force a newline, the stats update
     * was clobbering the just-printed response text on every single token.
     * Fix: save the cursor (\x1b7), jump to the terminal's last row
     * (\x1b[999;1H — terminals clamp an out-of-range row to their actual
     * height, so this works without querying the real size), print the
     * stats there, then restore the cursor (\x1b8) — a standard "status
     * line" technique that never disturbs the response's own position. */
    /* "\x1b" "7"/"\x1b" "8" split into separate string-literal segments —
     * \x hex escapes greedily consume any following hex digit, so "\x1b7"
     * would parse as a single (invalid, out-of-range) escape, not ESC + '7'. */
    fprintf(stderr, "\x1b" "7\x1b[999;1H%s%s%s%s%s [%lld tok, %.1f tok/s]\x1b[K\x1b[0m\x1b" "8",
            tn_color(color_enabled, TN_ANSI_BOLD), tn_color(color_enabled, TN_ANSI_CYAN),
            spinner, tn_color(color_enabled, TN_ANSI_RESET), tn_color(color_enabled, TN_ANSI_DIM),
            ls->count, rate);
    fflush(stderr);
}
