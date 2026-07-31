/*
 * tests/test_live_stats.c
 *
 * Phase 22.5 — spinner-frame selection is a pure function of tick count
 * (see include/cli/live_stats.h), tested here without any real terminal.
 * tn_live_stats_render itself writes ANSI-controlled output straight to
 * stderr and is left to manual/screenshot verification, same as the rest
 * of live_stats.c's terminal-side-effect code.
 */

#include <stdio.h>
#include <string.h>
#include "cli/live_stats.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_frame_nonnull_and_nonempty(void) {
    for (long long tick = 0; tick < 20; tick++) {
        const char *f = tn_live_stats_spinner_frame(tick);
        if (!f || f[0] == '\0') { FAIL("frame_nonnull_and_nonempty", "got NULL/empty frame"); return; }
    }
    PASS("frame_nonnull_and_nonempty");
}

static void test_frame_cycles_with_period_10(void) {
    for (long long tick = 0; tick < 30; tick++) {
        const char *a = tn_live_stats_spinner_frame(tick);
        const char *b = tn_live_stats_spinner_frame(tick + 10);
        if (strcmp(a, b) != 0) {
            FAIL("frame_cycles_with_period_10", "frame(tick) != frame(tick+10)");
            return;
        }
    }
    PASS("frame_cycles_with_period_10");
}

static void test_frame_varies_across_cycle(void) {
    const char *first = tn_live_stats_spinner_frame(0);
    int saw_different = 0;
    for (long long tick = 1; tick < 10; tick++) {
        if (strcmp(first, tn_live_stats_spinner_frame(tick)) != 0) { saw_different = 1; break; }
    }
    if (!saw_different) { FAIL("frame_varies_across_cycle", "all frames identical — not animated"); return; }
    PASS("frame_varies_across_cycle");
}

static void test_tick_and_init(void) {
    TnLiveStats ls;
    tn_live_stats_init(&ls);
    if (ls.count != 0) { FAIL("tick_and_init", "expected count 0 after init"); return; }
    tn_live_stats_tick(&ls);
    tn_live_stats_tick(&ls);
    if (ls.count != 2) { FAIL("tick_and_init", "expected count 2 after two ticks"); return; }
    PASS("tick_and_init");
}

int main(void) {
    printf("=== Phase 22.5 Live Stats Tests ===\n");
    test_frame_nonnull_and_nonempty();
    test_frame_cycles_with_period_10();
    test_frame_varies_across_cycle();
    test_tick_and_init();

    printf("\n");
    if (g_failures == 0) { printf("=== All live stats tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
