#include "transformer/moe_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_TRACK_LAYERS 128
#define MAX_TRACK_EXPERTS 512

static int *g_expert_usage_history = NULL;
static int g_track_n_layers = 0;
static int g_track_n_experts = 0;

void moe_analyzer_reset(int n_layers, int num_experts) {
    if (n_layers > MAX_TRACK_LAYERS) n_layers = MAX_TRACK_LAYERS;
    if (num_experts > MAX_TRACK_EXPERTS) num_experts = MAX_TRACK_EXPERTS;

    size_t total = (size_t)n_layers * num_experts;
    if (!g_expert_usage_history || g_track_n_layers != n_layers || g_track_n_experts != num_experts) {
        free(g_expert_usage_history);
        g_expert_usage_history = (int *)calloc(total, sizeof(int));
        g_track_n_layers = n_layers;
        g_track_n_experts = num_experts;
    } else {
        memset(g_expert_usage_history, 0, total * sizeof(int));
    }
}

void moe_analyzer_record_usage(int layer, const int *selected_experts, int k) {
    if (!g_expert_usage_history || layer < 0 || layer >= g_track_n_layers) return;
    for (int i = 0; i < k; i++) {
        int e = selected_experts[i];
        if (e >= 0 && e < g_track_n_experts) {
            g_expert_usage_history[layer * g_track_n_experts + e]++;
        }
    }
}

static void sort_experts_by_address(int *experts, int count) {
    for (int i = 1; i < count; i++) {
        int key = experts[i];
        int j = i - 1;
        while (j >= 0 && experts[j] > key) {
            experts[j + 1] = experts[j];
            j--;
        }
        experts[j + 1] = key;
    }
}

void moe_scheduler_plan(MoEExecutionPlan *plan,
                        int layer,
                        const int *selected_experts_batch,
                        const float *selected_scores_batch,
                        int num_tokens,
                        int top_k,
                        int num_experts) {
    if (!plan) return;
    memset(plan, 0, sizeof(MoEExecutionPlan));

    if (num_tokens >= 2) {
        /* Mode A: Expert-Centric Loop Inversion (Prompt & Batch Inference) */
        plan->execution_mode = MOE_EXEC_EXPERT_CENTRIC;
        plan->prefetch_strategy = 1;
        plan->thread_strategy = 0;

        /* Build Inverted Index: Expert -> Tokens */
        bool active_mask[MOE_MAX_PLAN_EXPERTS];
        memset(active_mask, 0, sizeof(active_mask));

        for (int t = 0; t < num_tokens; t++) {
            for (int k = 0; k < top_k; k++) {
                int e = selected_experts_batch[t * top_k + k];
                float score = selected_scores_batch[t * top_k + k];
                if (e < 0 || e >= num_experts || e >= MOE_MAX_PLAN_EXPERTS) continue;

                if (!active_mask[e]) {
                    active_mask[e] = true;
                    if (plan->num_active_experts < MOE_MAX_PLAN_EXPERTS) {
                        plan->expert_order[plan->num_active_experts++] = e;
                    }
                }

                int cnt = plan->expert_token_counts[e];
                if (cnt < MOE_MAX_PLAN_TOKENS) {
                    plan->expert_token_indices[e][cnt] = t;
                    plan->expert_token_scores[e][cnt]  = score;
                    plan->expert_token_counts[e]       = cnt + 1;
                }
            }
        }

        /* Sort active experts by ID to maximize address continuity */
        sort_experts_by_address(plan->expert_order, plan->num_active_experts);
    } else {
        /* Mode B: Single-Token Autoregressive Locality Optimization */
        plan->execution_mode = MOE_EXEC_SINGLE_TOKEN_LOCALITY;
        plan->prefetch_strategy = 1;
        plan->thread_strategy = 0;

        for (int k = 0; k < top_k; k++) {
            int e = selected_experts_batch[k];
            float score = selected_scores_batch[k];
            if (e < 0 || e >= num_experts || e >= MOE_MAX_PLAN_EXPERTS) continue;

            if (plan->num_active_experts < MOE_MAX_PLAN_EXPERTS) {
                plan->expert_order[plan->num_active_experts++] = e;
                plan->expert_token_counts[e] = 1;
                plan->expert_token_indices[e][0] = 0;
                plan->expert_token_scores[e][0]  = score;
            }
        }

        /* Sort active experts by ID for address continuity & cache hit rate */
        sort_experts_by_address(plan->expert_order, plan->num_active_experts);
        moe_analyzer_record_usage(layer, plan->expert_order, plan->num_active_experts);
    }
}
