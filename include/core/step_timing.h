#ifndef TN_STEP_TIMING_H
#define TN_STEP_TIMING_H

#include <stdint.h>
#include <stdio.h>

typedef enum {
    TN_STEP_1_TOKENIZATION = 1,
    TN_STEP_2_BOS_INJECTION = 2,
    TN_STEP_3_TOKEN_EMBEDDING = 3,
    TN_STEP_4_PRE_ATTN_RMSNORM = 4,
    TN_STEP_5_Q_PROJECTION = 5,
    TN_STEP_6_KV_A_COMPRESSION = 6,
    TN_STEP_7_KV_A_LATENT_NORM = 7,
    TN_STEP_8_KV_B_EXPANSION = 8,
    TN_STEP_9_YARN_ROPE = 9,
    TN_STEP_10_KV_CACHE_WRITE = 10,
    TN_STEP_11_ATTN_SCORE = 11,
    TN_STEP_12_POST_ATTN = 12,
    TN_STEP_13_PRE_FFN_RMSNORM = 13,
    TN_STEP_14_DENSE_FFN = 14,
    TN_STEP_15_MOE_ROUTING = 15,
    TN_STEP_16_MOE_ROUTED = 16,
    TN_STEP_17_SHARED_EXPERT = 17,
    TN_STEP_18_FINAL_RMSNORM = 18,
    TN_STEP_19_LM_HEAD = 19,
    TN_STEP_20_GREEDY_SAMPLING = 20,
    TN_STEP_21_EOS_CHECK = 21,
    /* Qwen35 Gated-DeltaNet linear-attention layers (2026-07-17): these two
     * have no analogue among the MLA-derived steps above, and without them
     * the hybrid path's recurrence time was invisible to the breakdown. */
    TN_STEP_22_DELTANET_CONV = 22,
    TN_STEP_23_DELTANET_RECURRENCE = 23,
    TN_STEP_COUNT = 24
} TnStepTimingId;

int tn_step_timing_enabled(void);
int64_t tn_step_timing_now_ns(void);
void tn_step_timing_reset(void);
void tn_step_timing_add(TnStepTimingId step, int64_t ns);
const char *tn_step_timing_name(TnStepTimingId step);
void tn_step_timing_report(FILE *out);

#endif
