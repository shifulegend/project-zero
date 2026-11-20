/*
 * transformer/gen_session_debug — verbose-mode diagnostics of a generation step
 * (split out of gen_session.c).
 */
#include <stddef.h>

#include "core/log.h"
#include "transformer/gen_session.h"

void gen_session_log_top5(int step, int sampled, const float *logits, int vocab_size) {
    /* Print top-5 logit values to diagnose garbage output */
    float top5v[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    int top5i[5]   = {0, 0, 0, 0, 0};
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > top5v[4]) {
            top5v[4] = logits[i];
            top5i[4] = i;
            for (int j = 3; j >= 0 && top5v[j] < top5v[j + 1]; j--) {
                float tv     = top5v[j];
                top5v[j]     = top5v[j + 1];
                top5v[j + 1] = tv;
                int ti       = top5i[j];
                top5i[j]     = top5i[j + 1];
                top5i[j + 1] = ti;
            }
        }
    }
    tn_log(TN_LOG_DEBUG, "dbg",
           "[DBG] step=%d sampled=%d  top5:[%d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)]\n", step,
           sampled, top5i[0], top5v[0], top5i[1], top5v[1], top5i[2], top5v[2], top5i[3], top5v[3],
           top5i[4], top5v[4]);
}
