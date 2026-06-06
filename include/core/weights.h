#ifndef TN_WEIGHTS_H
#define TN_WEIGHTS_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include <stdbool.h>

/* Layer weight storage type constants for layer_weight_type field.
 * Dispatch is purely data-driven — no model-name checks anywhere. */
#define WEIGHT_TYPE_F32  0   /* F32 heap or mmap — existing fallback path */
#define WEIGHT_TYPE_F16  1   /* F16 mmap zero-copy — 2× bandwidth vs F32  */
#define WEIGHT_TYPE_Q4K  3   /* Q4_K mmap zero-copy — fused matmul kernel  */

typedef struct {
    /* Token embedding table: vocab_size * dim stored as bfloat16 (tn_u16).
     * Used by the classifier (lm_head) and as BF16 lookup fallback.
     * When embd_f32 is non-NULL (GGUF quantized models), embed_token uses
     * the F32 table directly — matching llama.cpp's single-conversion path. */
    tn_u16 *token_embedding_table;

    /* F32 embedding table (optional): populated for GGUF models where
     * token_embd.weight is quantized (Q4_K etc.). Matches llama.cpp's
     * ggml_get_rows() path: Q4_K → F32 directly, no BF16 intermediate.
     * NULL for .bin format models and BF16/F16/F32 GGUF embeddings. */
    float  *embd_f32;

    /* Per-layer attention weights (arrays of n_layers pointers) */
    tn_i8 **wq;          /* Query:  dim * dim */
    tn_i8 **wk;          /* Key:    dim * kv_dim */
    tn_i8 **wv;          /* Value:  dim * kv_dim */
    tn_i8 **wo;          /* Output: dim * dim */
    float  *sq, *sk, *sv, *so;  /* per-layer scales */

    /* Per-layer FFN weights */
    tn_i8 **w1;          /* Gate:   dim * hidden_dim */
    tn_i8 **w2;          /* Down:   hidden_dim * dim */
    tn_i8 **w3;          /* Up:     dim * hidden_dim */
    float  *s1, *s2, *s3;       /* per-layer scales */

    /* Per-layer normalization (kept as float for stability) */
    float **rms_att_weight;      /* dim per layer (input_layernorm) */
    float **rms_ffn_weight;      /* dim per layer (post_attention_layernorm) */
    float **rms_attn_sub_norm;   /* dim per layer (BitNet attn_sub_norm, may be NULL) */
    float **rms_ffn_sub_norm;    /* hidden_dim per layer (BitNet ffn_sub_norm, may be NULL) */
    float  *rms_final_weight;    /* dim */

    /* Output classifier: dim * vocab_size.
     * Weight-tied to token_embedding_table when wcls_is_ternary is false. */
    tn_u16 *wcls;
    float   wcls_scale;
    bool    wcls_is_ternary; /* true = separate packed ternary lm_head */
    bool    layers_are_ternary;

    /* Layer weight storage type for non-ternary GGUF models.
     * Set at load time by gguf_loader based on the on-disk tensor format.
     * Dispatch in attention.c / ffn.c reads this to select the right kernel.
     * The ternary path (layers_are_ternary=true) ignores this field entirely. */
    int     layer_weight_type;  /* WEIGHT_TYPE_F32 / WEIGHT_TYPE_F16 / WEIGHT_TYPE_Q4K */

    /*
     * INT8 quantized classifier (computed at load time from BF16 wcls).
     * Per-row symmetric quantization: each row has a scale factor.
     * Halves LM head bandwidth (656 MB BF16 → 328 MB INT8) for ~2x speedup
     * on the classifier matmul, which is 35% of total inference time.
     * Set to NULL when wcls_is_ternary (ternary path is already fast).
     *
     * Weights are stored as unsigned uint8 (original + 128 bias) to enable
     * VNNI dpbusds (unsigned × signed) for 4x compute throughput over FMA.
     * The bias is corrected at runtime: true_dot = dpbusds_result - 128 * sum_qx.
     */
    tn_u8  *wcls_i8;        /* vocab_size * dim unsigned (biased +128), or NULL */
    float  *wcls_i8_scales;  /* vocab_size per-row scales, or NULL */

    /*
     * INT4 quantized classifier (computed at load time from BF16 wcls).
     * Packs 2 weights per byte: low nibble = w[2k], high nibble = w[2k+1].
     * Unsigned storage: w_signed + 8 → [1, 15] (symmetric around 8).
     * Halves LM head bandwidth again (328 MB INT8 → 164 MB INT4).
     * Runtime: unpack to uint8, use VNNI dpbusds, bias correction = 8 * sum_qx.
     */
    tn_u8  *wcls_i4;        /* vocab_size * ceil(dim/2) packed, or NULL */
    float  *wcls_i4_scales;  /* vocab_size per-row scales, or NULL */

    /*
     * Phase 17: Mixture of Experts (MoE) weights.
     * All pointers are NULL for dense models (BitNet, Llama).
     * Populated by moe_weights_alloc() + moe_weights_map() for MoE models.
     *
     * Dimensions:
     *   moe_gate_w[layer]       → packed ternary [dim × num_experts]
     *   moe_w{1,2,3}[layer][e] → packed ternary expert FFN weights
     *   moe_s{1,2,3}[layer][e] → per-expert ternary scales
     *
     * Shared experts (DeepSeek-V2 style — always active per MoE layer):
     *   moe_shared_w{1,2,3}[layer] → packed ternary shared FFN weights
     *   moe_shared_s{1,2,3}[layer] → shared expert scales
     *   NULL when MoEConfig.n_shared_experts == 0
     */
    tn_i8  **moe_gate_w;         /* [n_layers] gate matrix pointers           */
    float   *moe_gate_s;         /* [n_layers] gate scale factors             */
    tn_i8 ***moe_w1;             /* [n_layers][num_experts] gate/up proj      */
    tn_i8 ***moe_w2;             /* [n_layers][num_experts] down proj         */
    tn_i8 ***moe_w3;             /* [n_layers][num_experts] up proj           */
    float  **moe_s1;             /* [n_layers][num_experts] w1 scales         */
    float  **moe_s2;             /* [n_layers][num_experts] w2 scales         */
    float  **moe_s3;             /* [n_layers][num_experts] w3 scales         */
    /* Shared experts (DeepSeek-V2): always-active FFN per MoE layer */
    tn_i8  **moe_shared_w1;      /* [n_layers] shared gate_proj              */
    tn_i8  **moe_shared_w2;      /* [n_layers] shared down_proj              */
    tn_i8  **moe_shared_w3;      /* [n_layers] shared up_proj                */
    float   *moe_shared_s1;      /* [n_layers] shared w1 scale               */
    float   *moe_shared_s2;      /* [n_layers] shared w2 scale               */
    float   *moe_shared_s3;      /* [n_layers] shared w3 scale               */

    /* Phase 17.6: MLA (Multi-head Latent Attention) weight pointers.
     * All NULL for non-MLA models — zero memory overhead.
     * Populated by moe_weights_alloc() + moe_weights_map() when has_mla=1.
     *
     * Dimensions (DeepSeek-V2-Lite example):
     *   mla_wq[layer]    → packed ternary [n_heads*(qk_nope+qk_rope) × dim]
     *                       e.g. [3072 × 2048]
     *   mla_wkv_a[layer] → packed ternary [(kv_lora_rank+qk_rope) × dim]
     *                       e.g. [576 × 2048]
     *   mla_wkv_b[layer] → packed ternary [n_kv_heads*(qk_nope+v_head) × kv_lora_rank]
     *                       e.g. [4096 × 512]
     */
    tn_i8  **mla_wq;             /* [n_layers] full Q projection              */
    tn_i8  **mla_wkv_a;          /* [n_layers] KV compress projection         */
    tn_i8  **mla_wkv_b;          /* [n_layers] KV expand projection           */
    float   *mla_sq;             /* [n_layers] Q projection scale             */
    float   *mla_skv_a;          /* [n_layers] KV compress scale              */
    float   *mla_skv_b;          /* [n_layers] KV expand scale                */

    /* GGUF DeepSeek: routed expert weights kept in quantized form to avoid
     * dequantizing ~50 GB of expert data upfront. When has_expert_quant=true:
     *   moe_w{1,3}[l][e] → raw quantized bytes (mmap pointer, NOT heap)
     *   moe_w2[l][e]     → raw quantized bytes (mmap pointer, NOT heap)
     *   moe_s{1,2,3}[l][e] unused (set to 0.0f)
     * Dequantization happens per-expert at inference time in moe_ffn.c. */
    bool    has_expert_quant;           /* true = routed expert weights are quantized */
    int     expert_w13_quant_type;      /* GGUFType for w1/w3 — fallback when per_layer is NULL */
    int     expert_w2_quant_type;       /* GGUFType for w2 (down experts) — layer 0 fallback   */
    int    *expert_w13_quant_per_layer; /* per-layer GGUFType for w1/w3; NULL=use above */
    int    *expert_w2_quant_per_layer;  /* per-layer GGUFType for w2;    NULL=use above */
    /* true = MLA projection weights (wq/wkv_a/wkv_b/wo) are stored as raw
     * quantized bytes (heap copy) rather than dequantized F32.
     * When set, mla_wq/mla_wkv_a/mla_wkv_b/wo store raw Q4_K bytes. */
    bool    has_mla_quant;
    /* Per-layer quant type for shared expert projections.
     * GGUF type code stored per layer: 12=Q4_K (use parallel_matmul_q4k),
     * 13=Q5_K→F32 (use parallel_matmul_float32), 0=F32.
     * NULL when all layers use the same type (use scalar has_shared_w*_quant).
     * Follows the expert_w2_quant_per_layer pattern for consistency. */
    int    *shared_w1_type_per_layer;  /* per-layer GGUF type for gate proj */
    int    *shared_w2_type_per_layer;  /* per-layer GGUF type for down proj */
    int    *shared_w3_type_per_layer;  /* per-layer GGUF type for up   proj */

    int      moe_alloc_layers;         /* number of layers allocated for MoE arrays */
} TransformerWeights;

/**
 * Allocate the pointer arrays (not the weight data itself — that's mmap'd).
 */
TernaryError weights_alloc_pointers(TransformerWeights *w, const Config *cfg);

/**
 * Free the pointer arrays.
 */
void weights_free_pointers(TransformerWeights *w);

/**
 * Map weight pointers from mmap'd binary data.
 *
 * @param mc  MoE config — pass NULL or a zero-init struct for dense models.
 *            When mc->is_moe, per-layer dense FFN is skipped for MoE layers
 *            and moe_weights_map() is called automatically at the end.
 *
 * File layout for scale_mode=2 (ternary MoE):
 *   [0..127]   Header + MoE config (weight_data pointer starts at byte 128)
 *   [128+]     Per-layer weights (attention only for MoE layers)
 *   [after]    MoE block (gate + per-expert FFNs + optional shared expert)
 */
TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                         const MoEConfig *mc,
                         tn_i8 *data, size_t data_size);

/**
 * Build INT8 and INT4 quantized classifier variants from the BF16 wcls table.
 * Called from weights_map() (native format) and weights_from_gguf() (GGUF).
 * No-op when wcls_is_ternary is true or hardware does not benefit.
 */
void weights_build_classifier_quant(TransformerWeights *w, const Config *cfg);

#endif /* TN_WEIGHTS_H */
