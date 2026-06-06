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

#endif /* TN_MOE_CONFIG_H */
