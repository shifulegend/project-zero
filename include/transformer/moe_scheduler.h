#ifndef TN_MOE_SCHEDULER_H
#define TN_MOE_SCHEDULER_H

#include "core/config.h"
#include "core/moe_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifndef MOE_SCORE_BUF_SIZE
#define MOE_SCORE_BUF_SIZE 256
#endif

#ifndef MOE_MAX_PLAN_EXPERTS
#define MOE_MAX_PLAN_EXPERTS 64
#endif

#ifndef MOE_MAX_PLAN_TOKENS
#define MOE_MAX_PLAN_TOKENS 16
#endif

typedef enum {
    MOE_EXEC_EXPERT_CENTRIC = 0,        /* Mode A: Prompt processing & batch inference */
    MOE_EXEC_SINGLE_TOKEN_LOCALITY = 1   /* Mode B: Autoregressive single-token locality */
} MoEExecutionMode;

typedef struct {
    MoEExecutionMode execution_mode;
    int              num_active_experts;
    int              expert_order[MOE_MAX_PLAN_EXPERTS];
    
    /* Inverted Index: Expert ID -> Assigned Token Indices */
    int              expert_token_counts[MOE_MAX_PLAN_EXPERTS];
    int              expert_token_indices[MOE_MAX_PLAN_EXPERTS][MOE_MAX_PLAN_TOKENS];
    float            expert_token_scores[MOE_MAX_PLAN_EXPERTS][MOE_MAX_PLAN_TOKENS];
    
    int              prefetch_strategy; /* 0: disabled, 1: next-expert L1 prefetch */
    int              thread_strategy;   /* 0: caller-participates parallel dispatch */
} MoEExecutionPlan;

/* Usage Analyzer: Tracks historical expert co-occurrence and access frequency */
void moe_analyzer_record_usage(int layer, const int *selected_experts, int k);
void moe_analyzer_reset(int n_layers, int num_experts);

/* Dynamic Expert Scheduler: Formulates optimal execution plan based on input batch size & history */
void moe_scheduler_plan(MoEExecutionPlan *plan,
                        int layer,
                        const int *selected_experts_batch,
                        const float *selected_scores_batch,
                        int num_tokens,
                        int top_k,
                        int num_experts);

#endif /* TN_MOE_SCHEDULER_H */
