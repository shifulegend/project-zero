#include "core/step_timing.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int64_t total_ns;
    int64_t count;
} TnStepTimingStat;

static int g_step_timing_enabled = -1;
static TnStepTimingStat g_step_stats[TN_STEP_COUNT];

static const char *g_step_names[TN_STEP_COUNT] = {
    "",
    "Tokenization",
    "BOS injection / chat template",
    "Token embedding",
    "Pre-attention RMSNorm",
    "Q projection",
    "KV-A compression",
    "KV-A latent norm",
    "KV-B expansion",
    "YaRN RoPE",
    "KV cache write",
    "Attention score / softmax",
    "Post-attention output / residual",
    "Pre-FFN RMSNorm",
    "Dense FFN",
    "MoE routing",
    "MoE routed experts",
    "Shared expert",
    "Final RMSNorm",
    "LM head",
    "Greedy sampling",
    "EOS check",
    "DeltaNet conv1d",
    "DeltaNet delta-rule recurrence",
};

int tn_step_timing_enabled(void) {
    if (g_step_timing_enabled < 0) {
        const char *env = getenv("TN_STEP_TIMING");
        g_step_timing_enabled = (env && env[0] == '1') ? 1 : 0;
    }
    return g_step_timing_enabled;
}

int64_t tn_step_timing_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void tn_step_timing_reset(void) {
    if (!tn_step_timing_enabled()) return;
    memset(g_step_stats, 0, sizeof(g_step_stats));
}

void tn_step_timing_add(TnStepTimingId step, int64_t ns) {
    if (!tn_step_timing_enabled()) return;
    if (step <= 0 || step >= TN_STEP_COUNT) return;
    if (ns < 0) return;
    g_step_stats[step].total_ns += ns;
    g_step_stats[step].count++;
}

const char *tn_step_timing_name(TnStepTimingId step) {
    if (step <= 0 || step >= TN_STEP_COUNT) return "";
    return g_step_names[step];
}

void tn_step_timing_report(FILE *out) {
    if (!tn_step_timing_enabled() || !out) return;
    fprintf(out, "\n[STEP_TIMING_BEGIN]\n");
    for (int step = 1; step < TN_STEP_COUNT; step++) {
        double total_ms = (double)g_step_stats[step].total_ns / 1e6;
        double avg_ms = g_step_stats[step].count > 0
                      ? total_ms / (double)g_step_stats[step].count
                      : 0.0;
        fprintf(out,
                "step=%d name=\"%s\" total_ms=%.6f count=%lld avg_ms=%.6f\n",
                step,
                g_step_names[step],
                total_ms,
                (long long)g_step_stats[step].count,
                avg_ms);
    }
    fprintf(out, "[STEP_TIMING_END]\n");
}
