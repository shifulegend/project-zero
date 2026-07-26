#include "transformer/moe_router.h"
#include "math/parallel_matmul.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/**
 * Softmax in-place over a float array of length n.
 * Numerically stable: subtracts max before exp.
 *
 * DeepSeek-V2-Lite uses softmax gating (NOT sigmoid).
 * llama-model.cpp LLM_ARCH_DEEPSEEK2, n_layer==27: expert_gating_func defaults
 * to LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX when not set in GGUF metadata.
 * The SIGMOID case only applies to GLM-4.7 Lite (n_layer==47/48, vocab==154880).
 */
static void softmax_inplace(float *x, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
}

/**
 * Partial selection sort to find the top-k indices by descending value.
 * O(n*k) — acceptable because k is small (e.g. 6) and n ≤ 256.
 * Returns indices in descending score order.
 */
static void top_k_select(const float *scores, int n, int k,
                          int *out_indices) {
    /* Track which indices have been selected already */
    /* Use a tiny stack-allocated visited array; n ≤ 512 */
    /* Safe: num_experts is validated ≤ 512 in moe_config_read */
    int visited[512];
    int n_visit = n < 512 ? n : 512;
    memset(visited, 0, (size_t)n_visit * sizeof(int));

    for (int pick = 0; pick < k; pick++) {
        int   best_idx = -1;
        float best_val = -FLT_MAX;
        for (int i = 0; i < n; i++) {
            if (!visited[i] && scores[i] > best_val) {
                best_val = scores[i];
                best_idx = i;
            }
        }
        out_indices[pick] = best_idx;
        if (best_idx >= 0) visited[best_idx] = 1;
    }
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

void moe_router_forward(float       *expert_scores,
                        int         *selected_experts,
                        float       *selected_scores,
                        const float *x,
                        const float *gate_w,
                        int          dim,
                        int          num_experts,
                        int          top_k,
                        ThreadPool  *tp) {
    /* Step 1: gate_logits[e] = dot(x, gate_w_row[e])
     *
     * gate_w is F32 [num_experts × dim] — the tensor ffn_gate_inp.weight.
     * Gate is tiny: 2048×64 — parallel_matmul handles threading internally.
     */
    parallel_matmul_float32(expert_scores, x, gate_w, dim, num_experts, tp);

    /* Step 2: Softmax — convert raw logits to probabilities.
     * DeepSeek-V2-Lite (n_layer=27) defaults to softmax gating.
     * Ref: llama-model.cpp LLM_ARCH_DEEPSEEK2 → SOFTMAX when GGUF has no gating key.
     */
    softmax_inplace(expert_scores, num_experts);

    /* Step 3: Top-k selection */
    top_k_select(expert_scores, num_experts, top_k, selected_experts);

    /* Step 4: Copy softmax scores for selected experts (no renormalisation —
     * DeepSeek norm_topk_prob=false uses raw softmax probabilities). */
    for (int i = 0; i < top_k; i++) {
        int e = selected_experts[i];
        selected_scores[i] = (e >= 0) ? expert_scores[e] : 0.0f;
    }
}
