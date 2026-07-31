#ifndef TN_CLI_LIVE_STATS_H
#define TN_CLI_LIVE_STATS_H

/*
 * Phase 22.3 — Live tok/s indicator for the REPL, updated once per streamed
 * token. Reuses cli/timer.h (timer_now_us/timer_tokens_per_sec) rather than
 * reimplementing rate math. TTY-gated (real call sites in repl.c pass
 * isatty(fileno(stdout))); non-TTY runs keep the existing end-of-generation
 * stderr summary line from generate_with_callback, so this is purely
 * additive.
 */

#include <stdint.h>

typedef struct {
    int64_t start_us; /* set on first tick */
    long long count;
} TnLiveStats;

void tn_live_stats_init(TnLiveStats *ls);

/* Call once per generated token. */
void tn_live_stats_tick(TnLiveStats *ls);

/* Returns the braille spinner glyph for a given tick count — a "moving
 * logo" next to the live tok/s status line, the same idea as Claude Code's
 * animated indicator while it's actively working. Pure function of tick
 * (cycles through a fixed 10-frame set), exposed separately from render()
 * so it's unit-testable without a real terminal. */
const char *tn_live_stats_spinner_frame(long long tick);

/* Renders the current tok/s (via timer_tokens_per_sec) plus the spinner
 * glyph to stderr as an in-place status-line update (see live_stats.c for
 * the cursor save/jump/restore technique). No-op when is_tty is false or
 * fewer than 2 tokens have ticked yet. color_enabled tints the spinner with
 * the CLI's accent color (see cli/color.h); the stats text itself stays dim
 * either way. */
void tn_live_stats_render(const TnLiveStats *ls, int is_tty, int color_enabled);

#endif /* TN_CLI_LIVE_STATS_H */
