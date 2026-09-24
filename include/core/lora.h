#ifndef TN_LORA_H
#define TN_LORA_H

/*
 * lora.h — Phase 19: LoRA (Low-Rank Adaptation) hot-swappable adapters.
 *
 * A LoRA adapter adds a small correction on top of a frozen base model's
 * projections: out = base_matmul(x, W) + scale * (x @ A @ B), where scale =
 * alpha / rank. `A`/`B` are tiny relative to the base weight matrix (rank is
 * typically 8-128), so a whole adapter is usually tens of MB even for a
 * multi-GB base model.
 *
 * Scope (v1, this pass): the generic dense/GQA attention/FFN path only
 * (attention_forward()/ffn_forward() in src/transformer/{attention,ffn}.c) --
 * not MLA, not Qwen3-MoE, not Qwen3.5/3.6 hybrid, not MoE-FFN, not the
 * batched speculative-decoding path. This covers the architecture the
 * overwhelming majority of public HuggingFace LoRA adapters target. See
 * docs/ai/decision-log.md for the full scope rationale.
 *
 * Binary format — .lora.bin (mirrors vision_weights_load.h's own
 * magic+version+fixed-header convention):
 *   [0:4]   magic         0x41524F4C ("LORA" LE)
 *   [4:8]   version       1
 *   [8:12]  n_layers      (int32)
 *   [12:16] rank          (int32)
 *   [16:20] alpha         (float32)
 *   [20:24] dim           (int32) -- base model's attention dim
 *   [24:28] hidden_dim    (int32) -- base model's FFN hidden dim
 *   [28:32] target_mask   (int32) -- bitmask, see LoRaTargetBit below
 *   [32:36] kv_dim        (int32) -- base model's GQA kv_dim (config_kv_dim();
 *                                    == dim for a non-GQA/MHA model)
 *   [36:64] reserved (zeroes, pad to 64-byte alignment)
 *   For each bit set in target_mask, in LoRaTargetBit order, n_layers blocks
 *   of (A then B), float16, row-major:
 *     A [rank x n]   -- n = dim for Q/K/V/O, hidden_dim for DOWN, dim for GATE/UP
 *     B [d x rank]   -- d = dim for Q/O, kv_dim for K/V (GQA -- K/V project
 *                       dim -> kv_dim, NOT dim -> dim; found via a real
 *                       downloaded SmolLM2 LoRA adapter, see
 *                       docs/ai/mistakes.md), dim for GATE/UP, hidden_dim
 *                       for DOWN
 *   (matches HuggingFace PEFT's own lora_A.weight [rank, in_features] /
 *   lora_B.weight [out_features, rank] layout exactly -- the converter does
 *   no reshape/transpose, just an F32->F16 cast.)
 */

#include "core/error.h"
#include <stddef.h>
#include <stdint.h>

#define LORA_BIN_MAGIC 0x41524F4Cu

typedef enum {
    LORA_TARGET_Q     = 0,
    LORA_TARGET_K     = 1,
    LORA_TARGET_V     = 2,
    LORA_TARGET_O     = 3,
    LORA_TARGET_GATE  = 4,
    LORA_TARGET_UP    = 5,
    LORA_TARGET_DOWN  = 6,
    LORA_TARGET_COUNT = 7
} LoRaTargetBit;

/* One decoded (A, B) pair for a single (layer, target module). rank == 0 is
 * the "inactive" sentinel -- lora_apply() treats it as a pure no-op, same
 * NULL-for-inapplicable convention TransformerWeights already uses
 * (q35_attn_q_norm etc., include/core/weights.h). */
typedef struct {
    const float *A;     /* [rank x n], row-major, owned by LoRAWeights */
    const float *B;     /* [d x rank], row-major, owned by LoRAWeights */
    int          rank;
    float        scale; /* alpha / rank, precomputed at load time */
} LoRAModule;

typedef struct {
    int   n_layers;
    int   rank;
    float alpha;
    int   dim;
    int   hidden_dim;

    /* One array per target module, length n_layers each. A module array is
     * NULL entirely when that target wasn't in the adapter's target_mask --
     * checked once here, not via per-layer flags or string comparison in
     * the hot path (see lora.h's format doc for target_mask). */
    LoRAModule *q;
    LoRAModule *k;
    LoRAModule *v;
    LoRAModule *o;
    LoRAModule *gate;
    LoRAModule *up;
    LoRAModule *down;
} LoRAWeights;

/* Safe accessor for a per-layer LoRAModule array that may itself be NULL
 * (the whole target wasn't in this adapter's target_mask, see lora_load()).
 * Callers do `lora_mod(lora->q, layer)` rather than `&lora->q[layer]`
 * directly, which would be undefined behavior when `lora->q == NULL`. */
static inline const LoRAModule *lora_mod(const LoRAModule *arr, int layer) {
    return arr ? &arr[layer] : NULL;
}

/* Loads path (a .lora.bin file) into *lora. dim/kv_dim/hidden_dim/n_layers
 * are the *base model's* values (kv_dim from config_kv_dim(cfg), NOT
 * necessarily == dim -- GQA models project K/V to a narrower width, see
 * lora.h's format doc) -- used only to sanity-check the file's own header
 * matches the model this adapter is being applied to (a LoRA trained for a
 * different-shaped base model would silently corrupt output otherwise).
 * Safe to call lora_free() on a zeroed LoRAWeights (memset convention, same
 * as MappedFile). */
TernaryError lora_load(LoRAWeights *lora, const char *path,
                        int dim, int kv_dim, int hidden_dim, int n_layers);
void lora_free(LoRAWeights *lora);

#endif /* TN_LORA_H */
