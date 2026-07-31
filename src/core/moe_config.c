#include "core/moe_config.h"
#include <stdint.h>   /* int32_t, uint8_t */
#include <stdio.h>
#include <string.h>

void moe_config_init_dense(MoEConfig *mc) {
    memset(mc, 0, sizeof(*mc));
    /* is_moe = false, all MoE code will be skipped */
}

TernaryError moe_config_read(MoEConfig *mc, const void *data,
                              size_t data_size, size_t *offset) {
    if (!mc || !data || !offset) return TN_ERR_INVALID_WEIGHTS;

    /* MoE header: 7 × int32 = 28 bytes (dense+MoE fields) + 5 × int32 MLA extension = 48 bytes.
     * Old files have only 32 bytes written; bytes 28-47 were padding (all zero) → has_mla=0. */
    const size_t HEADER_SIZE = 48;
    if (*offset + HEADER_SIZE > data_size) {
        /* Accept 32-byte legacy header — MLA fields will be zeroed below */
        if (*offset + 32 > data_size) return TN_ERR_INVALID_WEIGHTS;
    }

    const uint8_t *p = (const uint8_t *)data + *offset;

    int32_t num_experts, num_experts_per_tok, expert_hidden_dim;
    int32_t first_k_dense_replace, is_moe_flag;

    memcpy(&num_experts,           p +  0, 4);
    memcpy(&num_experts_per_tok,   p +  4, 4);
    memcpy(&expert_hidden_dim,     p +  8, 4);
    memcpy(&first_k_dense_replace, p + 12, 4);
    memcpy(&is_moe_flag,           p + 16, 4);

    int32_t n_shared = 0, shared_hdim = 0;
    memcpy(&n_shared,   p + 20, 4);
    memcpy(&shared_hdim, p + 24, 4);

    /* MLA extension fields (bytes 28-47); zero if old 32-byte header */
    int32_t has_mla = 0, kv_lora_rank = 0;
    int32_t qk_nope_head_dim = 0, qk_rope_head_dim = 0, v_head_dim = 0;
    if (*offset + HEADER_SIZE <= data_size) {
        memcpy(&has_mla,           p + 28, 4);
        memcpy(&kv_lora_rank,      p + 32, 4);
        memcpy(&qk_nope_head_dim,  p + 36, 4);
        memcpy(&qk_rope_head_dim,  p + 40, 4);
        memcpy(&v_head_dim,        p + 44, 4);
    }

    if (num_experts <= 0 || num_experts > 512)       return TN_ERR_INVALID_WEIGHTS;
    if (num_experts_per_tok <= 0)                    return TN_ERR_INVALID_WEIGHTS;
    if (num_experts_per_tok > num_experts)           return TN_ERR_INVALID_WEIGHTS;
    if (expert_hidden_dim <= 0)                      return TN_ERR_INVALID_WEIGHTS;
    if (first_k_dense_replace < 0)                   return TN_ERR_INVALID_WEIGHTS;
    if (n_shared < 0 || n_shared > 64)               return TN_ERR_INVALID_WEIGHTS;
    if (n_shared > 0 && shared_hdim <= 0)            return TN_ERR_INVALID_WEIGHTS;
    if (has_mla && (kv_lora_rank <= 0 || qk_nope_head_dim <= 0 ||
                    qk_rope_head_dim <= 0 || v_head_dim <= 0))
        return TN_ERR_INVALID_WEIGHTS;

    mc->num_experts              = num_experts;
    mc->num_experts_per_tok      = num_experts_per_tok;
    mc->expert_hidden_dim        = expert_hidden_dim;
    mc->first_k_dense_replace    = first_k_dense_replace;
    mc->is_moe                   = (is_moe_flag != 0);
    mc->n_shared_experts         = n_shared;
    mc->shared_expert_hidden_dim = (n_shared > 0) ? shared_hdim : 0;
    mc->has_mla                  = (has_mla != 0);
    mc->kv_lora_rank             = has_mla ? kv_lora_rank     : 0;
    mc->qk_nope_head_dim         = has_mla ? qk_nope_head_dim : 0;
    mc->qk_rope_head_dim         = has_mla ? qk_rope_head_dim : 0;
    mc->v_head_dim               = has_mla ? v_head_dim       : 0;

    /* Advance by the full 48 bytes; old 32-byte files advance by 32 */
    *offset += (*offset + HEADER_SIZE <= data_size) ? HEADER_SIZE : 32;
    return TN_OK;
}

void moe_config_print(const MoEConfig *mc) {
    if (!mc) return;
    if (mc->has_linear_attn) {
        printf("[Qwen3.5/3.6] hybrid attention: full_attention_interval=%d "
               "attn_head_dim=%d attn_rope_dim=%d "
               "ssm(conv_kernel=%d group_count=%d inner_size=%d state_size=%d time_step_rank=%d)\n",
               mc->full_attention_interval, mc->attn_head_dim, mc->attn_rope_dim,
               mc->ssm_conv_kernel, mc->ssm_group_count, mc->ssm_inner_size,
               mc->ssm_state_size, mc->ssm_time_step_rank);
    }
    if (!mc->is_moe) {
        printf("[MoE] Dense model (MoE disabled)\n");
        return;
    }
    printf("[MoE] num_experts=%d  top_k=%d  expert_hidden_dim=%d  "
           "first_k_dense=%d  shared_experts=%d (hdim=%d)\n",
           mc->num_experts, mc->num_experts_per_tok,
           mc->expert_hidden_dim, mc->first_k_dense_replace,
           mc->n_shared_experts, mc->shared_expert_hidden_dim);
    if (mc->has_mla) {
        printf("[MLA] kv_lora_rank=%d  qk_nope_head_dim=%d  "
               "qk_rope_head_dim=%d  v_head_dim=%d\n",
               mc->kv_lora_rank, mc->qk_nope_head_dim,
               mc->qk_rope_head_dim, mc->v_head_dim);
    }
}
