/* Expert hit tracking (Phase 17 diagnostics): which experts each layer used, printed after a run.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transformer/moe_ffn.h"
#include "transformer/moe_ffn_internal.h"

static int g_track_n_layers    = 0;
static int g_track_n_experts   = 0;
static uint32_t *g_expert_hits = NULL; /* [n_layers * num_experts] */

/* Counts one invocation of expert `e` in `layer` (no-op while tracking is off or out of range). */
void moe_track_hit(int layer, int e) {
    if (g_expert_hits && layer < g_track_n_layers && e < g_track_n_experts)
        g_expert_hits[layer * g_track_n_experts + e]++;
}

void moe_expert_tracking_reset(int n_layers, int num_experts) {
    int needed = n_layers * num_experts;
    if (g_track_n_layers != n_layers || g_track_n_experts != num_experts) {
        free(g_expert_hits);
        g_expert_hits     = (uint32_t *)calloc((size_t)needed, sizeof(uint32_t));
        g_track_n_layers  = n_layers;
        g_track_n_experts = num_experts;
    } else if (g_expert_hits) {
        memset(g_expert_hits, 0, (size_t)needed * sizeof(uint32_t));
    }
}

void moe_expert_tracking_print(int n_layers, int num_experts) {
    if (!g_expert_hits || n_layers <= 0 || num_experts <= 0) return;

    printf("\n--- Expert Utilisation Summary (%d MoE layers × %d experts) ---\n", n_layers,
           num_experts);

    int total_unique_global                     = 0;
    uint32_t global_hits[MOE_MAX_EXPERTS_STACK] = {0};

    for (int l = 0; l < n_layers; l++) {
        uint32_t *row     = g_expert_hits + l * num_experts;
        int unique        = 0;
        uint32_t total    = 0;
        int top5[5]       = {-1, -1, -1, -1, -1};
        uint32_t top5v[5] = {0};

        for (int e = 0; e < num_experts; e++) {
            if (row[e] > 0) {
                unique++;
                global_hits[e] += row[e];
            }
            total += row[e];
            /* maintain top-5 */
            for (int k = 0; k < 5; k++) {
                if (row[e] > top5v[k]) {
                    /* shift down */
                    for (int j = 4; j > k; j--) {
                        top5[j]  = top5[j - 1];
                        top5v[j] = top5v[j - 1];
                    }
                    top5[k]  = e;
                    top5v[k] = row[e];
                    break;
                }
            }
        }

        printf("  L%02d: %3d unique / %2d total  (invocations: %4u)  top:", l, unique, num_experts,
               total);
        for (int k = 0; k < 5 && top5[k] >= 0; k++) printf(" E%d(%u)", top5[k], top5v[k]);
        printf("\n");
    }

    /* Global expert utilisation across all layers */
    int global_unique = 0;
    for (int e = 0; e < num_experts; e++)
        if (global_hits[e] > 0) global_unique++;
    printf("  Global: %d / %d experts activated (%.1f%%)\n", global_unique, num_experts,
           100.0f * global_unique / num_experts);
    (void)total_unique_global;
    printf("---\n");
}

void moe_expert_tracking_free(void) {
    free(g_expert_hits);
    g_expert_hits     = NULL;
    g_track_n_layers  = 0;
    g_track_n_experts = 0;
}
