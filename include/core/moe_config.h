#ifndef TN_MOE_CONFIG_H
#define TN_MOE_CONFIG_H

#include "core/error.h"
#include <stdbool.h>
#include <stddef.h>   /* size_t */
#include <stdint.h>   /* int32_t */

/**
 * MoEConfig — Mixture of Experts routing configuration.
 *
 * For dense models (BitNet, Llama), all fields are zero/false.
 * For MoE models (DeepSeek-V2, Mixtral), these fields describe the
 * routing topology that replaces the standard dense FFN.
 *
 * The MoE FFN structure per layer:
 *   - A gate weight matrix: [dim × num_experts]
 *   - num_experts independent FFNs (each with w1, w2, w3)
 *   - At runtime: only top num_experts_per_tok experts execute per token
 */
typedef struct {
    int  num_experts;             /* total routed experts per layer, e.g. 64 (DeepSeek-V2-Lite) */
    int  num_experts_per_tok;     /* top-k active per token, e.g. 6                               */
    int  expert_hidden_dim;       /* each routed expert's FFN hidden dim                          */
    bool is_moe;                  /* false → dense model, all MoE code is a no-op                */
    int  first_k_dense_replace;   /* DeepSeek: first N layers use dense FFN, rest use MoE         */
                                  /* set to 0 if all layers are MoE (e.g. Mixtral)                */
    int  n_shared_experts;        /* DeepSeek: always-active shared experts per MoE layer (0=none)*/
    int  shared_expert_hidden_dim;/* hidden dim of the shared-expert FFN (n_shared × moe_intermediate_size) */

    /* MLA (Multi-head Latent Attention) fields — Phase 17.5 extension.
     * Bytes 28–47 of the MoE header (were padding, always zero in old files).
     * has_mla == 0 for ALL non-MLA models → MLA code is never reached.           */
    int  has_mla;             /* 1 for DeepSeek-V2-style MLA, 0 otherwise          */
    int  kv_lora_rank;        /* latent KV dim, e.g. 512                           */
    int  qk_nope_head_dim;    /* per-head non-positional Q/K dim, e.g. 128         */
    int  qk_rope_head_dim;    /* per-head RoPE positional dim, e.g. 64             */
    int  v_head_dim;          /* per-head value dim, e.g. 128                      */

    /* Qwen3.5/3.6 hybrid Gated-DeltaNet + Gated-Attention fields.
     * has_linear_attn == 0 for ALL non-hybrid models → this code path is never
     * reached, same convention as has_mla above. Populated by
     * moe_config_from_gguf() for arch=="qwen35" (see gguf_loader.c). */
    int  has_linear_attn;        /* 1 for Qwen3.5/3.6-style hybrid attention, 0 otherwise */
    int  full_attention_interval;/* every Nth layer (1-indexed) is full-attention, rest linear */
    int  attn_head_dim;          /* full-attention per-head dim (key_length/value_length),
                                     NOT dim/n_heads — this arch's Q/K/V shapes don't factor evenly */
    int  attn_rope_dim;          /* partial-rotary dim for full-attention layers (<= attn_head_dim) */
    int  ssm_conv_kernel;        /* depthwise causal conv1d kernel size (e.g. 4) */
    int  ssm_group_count;        /* linear-attention K/Q head count (repeats to ssm_time_step_rank) */
    int  ssm_inner_size;         /* linear-attention total value width (num_v_heads * head_v_dim) */
    int  ssm_state_size;         /* linear-attention per-head K/V dim (e.g. 128) */
    int  ssm_time_step_rank;     /* linear-attention V head count (== num decay/gate heads) */

    /* Qwen3-MoE fields. has_qk_norm == 0 for ALL other models → this code
     * path is never reached, same convention as has_mla/has_linear_attn
     * above. Populated by moe_config_from_gguf() for arch=="qwen3moe" (see
     * gguf_loader.c). Reuses attn_head_dim (already defined above for
     * qwen35) rather than a new field — same "independent head_dim"
     * problem qwen35 already solves, populated from attention.key_length. */
    int  has_qk_norm;            /* 1 for Qwen3-MoE-style per-head QK RMSNorm before RoPE */
} MoEConfig;

/**
 * Initialise MoEConfig for a dense (non-MoE) model.
 * Sets is_moe=false and all other fields to 0.
 * Call this before weights_map() for BitNet/Llama models.
 */
void moe_config_init_dense(MoEConfig *mc);

/**
 * Read MoEConfig from a memory-mapped file at a given byte offset.
 * Advances *offset past the MoE header block.
 * Called from the model loader when the file header signals is_moe=true.
 *
 * MoE header binary layout (48 bytes, little-endian):
 *   int32  num_experts
 *   int32  num_experts_per_tok
 *   int32  expert_hidden_dim
 *   int32  first_k_dense_replace
 *   int32  is_moe  (1 = true)
 *   int32  n_shared_experts       (0 for Mixtral, 2 for DeepSeek-V2-Lite)
 *   int32  shared_expert_hidden_dim (0 if n_shared_experts == 0)
 *   int32  has_mla                (bytes 28-31; 0 in old files)
 *   int32  kv_lora_rank           (bytes 32-35)
 *   int32  qk_nope_head_dim       (bytes 36-39)
 *   int32  qk_rope_head_dim       (bytes 40-43)
 *   int32  v_head_dim             (bytes 44-47)
 *
 * Old model files have bytes 28-47 == 0 → has_mla = 0 → MLA path never taken.
 * No format break for any existing model.
 */
TernaryError moe_config_read(MoEConfig *mc, const void *data,
                              size_t data_size, size_t *offset);

/**
 * Print MoEConfig to stdout for boot diagnostics.
 */
void moe_config_print(const MoEConfig *mc);

/**
 * Returns true if layer `l` should use MoE FFN instead of dense FFN.
 * Dense layers (first_k_dense_replace > 0) always run the dense path.
 */
static inline bool moe_layer_is_moe(const MoEConfig *mc, int layer) {
    if (!mc || !mc->is_moe) return false;
    return layer >= mc->first_k_dense_replace;
}

/**
 * Returns true if layer `l` is a full-attention (GQA + partial-rotary + gate)
 * layer in a Qwen3.5/3.6-style hybrid model; false means it's a linear-attention
 * (Gated DeltaNet) layer. Formula matches llama.cpp's qwen35.cpp default:
 *   is_full_attention(l) == ((l + 1) % full_attention_interval == 0)
 * Always false when has_linear_attn is unset (dense/MLA models never reach this).
 */
static inline bool q35_layer_is_full_attn(const MoEConfig *mc, int layer) {
    if (!mc || !mc->has_linear_attn) return false;
    int interval = mc->full_attention_interval > 0 ? mc->full_attention_interval : 4;
    return ((layer + 1) % interval) == 0;
}

#endif /* TN_MOE_CONFIG_H */
