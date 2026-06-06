#ifndef TN_MOE_ROUTER_H
#define TN_MOE_ROUTER_H

#include "core/platform.h"
#include "threading/thread_pool.h"

/**
 * moe_router_forward — The MoE "gate": selects top-k experts for a token.
 *
 * Steps:
 *  1. gate_logits[e] = dot(x, gate_w[e])  for each expert e
 *     (a tiny dim × num_experts matmul — 2048×64 F32 gate weights)
 *  2. Softmax over gate_logits (standard competition between experts).
 *     DeepSeek-V2-Lite uses softmax (NOT sigmoid). Sigmoid only applies
 *     to GLM-4.7 Lite. Ref: llama-model.cpp LLM_ARCH_DEEPSEEK2 default.
 *  3. Top-k selection: find the k largest sigmoid scores.
 *
 * @param expert_scores     Output buffer [num_experts] — softmax probs (full).
 * @param selected_experts  Output buffer [top_k]        — chosen expert indices.
 * @param selected_scores   Output buffer [top_k]        — softmax scores for chosen experts.
 * @param x                 Current hidden state [dim].
 * @param gate_w            Gate weight matrix, F32 [num_experts × dim].
 * @param dim               Hidden dimension.
 * @param num_experts       Total number of experts.
 * @param top_k             How many experts to activate.
 * @param tp                Thread pool (may be NULL for sequential).
 */
void moe_router_forward(float       *expert_scores,
                        int         *selected_experts,
                        float       *selected_scores,
                        const float *x,
                        const float *gate_w,
                        int          dim,
                        int          num_experts,
                        int          top_k,
                        ThreadPool  *tp);

#endif /* TN_MOE_ROUTER_H */
