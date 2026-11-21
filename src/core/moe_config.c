#include "core/moe_config.h"

#include <stdint.h> /* int32_t, uint8_t */
#include <stdio.h>
#include <string.h>

#include "core/log.h"

void moe_config_init_dense(MoEConfig *mc) {
    memset(mc, 0, sizeof(*mc));
    /* is_moe = false, all MoE code will be skipped */
}

/* The .bin MoE header: 7 x int32 = 28 bytes (dense+MoE fields) + 5 x int32 MLA extension = 48
 * bytes. Old files have only 32 bytes written; bytes 28-47 were padding (all zero) -> has_mla=0. */
#define MOE_BIN_HEADER_SIZE 48
#define MOE_BIN_LEGACY_SIZE 32

typedef struct {
    int32_t num_experts, num_experts_per_tok, expert_hidden_dim;
    int32_t first_k_dense_replace, is_moe_flag;
    int32_t n_shared, shared_hdim;
    int32_t has_mla, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim;
} MoeBinHeader;

/* Reads the fields at p; the MLA extension (bytes 28-47) only when `full` (zero otherwise). */
static void moe_bin_header_read(const uint8_t *p, int full, MoeBinHeader *h) {
    memset(h, 0, sizeof(*h));
    memcpy(&h->num_experts, p + 0, 4);
    memcpy(&h->num_experts_per_tok, p + 4, 4);
    memcpy(&h->expert_hidden_dim, p + 8, 4);
    memcpy(&h->first_k_dense_replace, p + 12, 4);
    memcpy(&h->is_moe_flag, p + 16, 4);
    memcpy(&h->n_shared, p + 20, 4);
    memcpy(&h->shared_hdim, p + 24, 4);
    if (!full) return;
    memcpy(&h->has_mla, p + 28, 4);
    memcpy(&h->kv_lora_rank, p + 32, 4);
    memcpy(&h->qk_nope_head_dim, p + 36, 4);
    memcpy(&h->qk_rope_head_dim, p + 40, 4);
    memcpy(&h->v_head_dim, p + 44, 4);
}

static int moe_bin_experts_valid(const MoeBinHeader *h) {
    if (h->num_experts <= 0 || h->num_experts > 512) return 0;
    if (h->num_experts_per_tok <= 0) return 0;
    if (h->num_experts_per_tok > h->num_experts) return 0;
    if (h->expert_hidden_dim <= 0) return 0;
    return h->first_k_dense_replace >= 0;
}

static int moe_bin_shared_valid(const MoeBinHeader *h) {
    if (h->n_shared < 0 || h->n_shared > 64) return 0;
    return h->n_shared == 0 || h->shared_hdim > 0;
}

static int moe_bin_mla_valid(const MoeBinHeader *h) {
    if (!h->has_mla) return 1;
    return h->kv_lora_rank > 0 && h->qk_nope_head_dim > 0 && h->qk_rope_head_dim > 0 &&
           h->v_head_dim > 0;
}

static void moe_bin_header_apply(MoEConfig *mc, const MoeBinHeader *h) {
    mc->num_experts              = h->num_experts;
    mc->num_experts_per_tok      = h->num_experts_per_tok;
    mc->expert_hidden_dim        = h->expert_hidden_dim;
    mc->first_k_dense_replace    = h->first_k_dense_replace;
    mc->is_moe                   = (h->is_moe_flag != 0);
    mc->n_shared_experts         = h->n_shared;
    mc->shared_expert_hidden_dim = (h->n_shared > 0) ? h->shared_hdim : 0;
    mc->mla.enabled              = (h->has_mla != 0);
    mc->mla.kv_lora_rank         = h->has_mla ? h->kv_lora_rank : 0;
    mc->mla.qk_nope_head_dim     = h->has_mla ? h->qk_nope_head_dim : 0;
    mc->mla.qk_rope_head_dim     = h->has_mla ? h->qk_rope_head_dim : 0;
    mc->mla.v_head_dim           = h->has_mla ? h->v_head_dim : 0;
    /* .bin MoE header carries no router-weighting fields: DeepSeek-V2-Lite
     * semantics (softmax, no renormalization, no scale). */
    mc->expert_gating_func   = MOE_GATING_SOFTMAX;
    mc->expert_weights_norm  = 0;
    mc->expert_weights_scale = 0.0f;
}

TernaryError moe_config_read(MoEConfig *mc, const void *data, size_t data_size, size_t *offset) {
    if (!mc || !data || !offset) return TN_ERR_INVALID_WEIGHTS;

    /* Accept the 32-byte legacy header: the MLA fields are then zero. */
    int full = *offset + MOE_BIN_HEADER_SIZE <= data_size;
    if (!full && *offset + MOE_BIN_LEGACY_SIZE > data_size) return TN_ERR_INVALID_WEIGHTS;

    MoeBinHeader h;
    moe_bin_header_read((const uint8_t *)data + *offset, full, &h);
    if (!moe_bin_experts_valid(&h) || !moe_bin_shared_valid(&h) || !moe_bin_mla_valid(&h))
        return TN_ERR_INVALID_WEIGHTS;

    moe_bin_header_apply(mc, &h);
    /* Advance by the full 48 bytes; old 32-byte files advance by 32 */
    *offset += full ? MOE_BIN_HEADER_SIZE : MOE_BIN_LEGACY_SIZE;
    return TN_OK;
}

void moe_config_print(const MoEConfig *mc) {
    if (!mc) return;
    if (mc->has_linear_attn) {
        tn_log_out(
            TN_LOG_INFO, "qwen3_5",
            "[Qwen3.5/3.6] hybrid attention: full_attention_interval=%d "
            "attn_head_dim=%d attn_rope_dim=%d "
            "ssm(conv_kernel=%d group_count=%d inner_size=%d state_size=%d time_step_rank=%d)\n",
            mc->full_attention_interval, mc->attn_head_dim, mc->attn_rope_dim, mc->ssm_conv_kernel,
            mc->ssm_group_count, mc->ssm_inner_size, mc->ssm_state_size, mc->ssm_time_step_rank);
    }
    if (mc->is_ssm) {
        TN_OUT_INF(
            "mamba", "[Mamba] state-space model: d_inner=%d d_state=%d d_conv=%d dt_rank=%d\n",
            mc->ssm_inner_size, mc->ssm_state_size, mc->ssm_conv_kernel, mc->ssm_time_step_rank);
        return;
    }
    if (!mc->is_moe) {
        TN_OUT_INF("moe", "[MoE] Dense model (MoE disabled)\n");
        return;
    }
    TN_OUT_INF("moe",
               "[MoE] num_experts=%d  top_k=%d  expert_hidden_dim=%d  "
               "first_k_dense=%d  shared_experts=%d (hdim=%d)\n",
               mc->num_experts, mc->num_experts_per_tok, mc->expert_hidden_dim,
               mc->first_k_dense_replace, mc->n_shared_experts, mc->shared_expert_hidden_dim);
    if (mc->mla.enabled) {
        TN_OUT_INF("mla",
                   "[MLA] kv_lora_rank=%d  qk_nope_head_dim=%d  "
                   "qk_rope_head_dim=%d  v_head_dim=%d\n",
                   mc->mla.kv_lora_rank, mc->mla.qk_nope_head_dim, mc->mla.qk_rope_head_dim,
                   mc->mla.v_head_dim);
    }
}
