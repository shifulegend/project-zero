#ifndef TN_ROPE_H
#define TN_ROPE_H

#include "core/config.h"

/**
 * Precompute the RoPE frequency table: freq[i] = 1.0 / pow(theta, 2*i/head_dim).
 *
 * @param out       Output array (must hold head_dim/2 floats)
 * @param head_dim  Dimension per head (must be even)
 * @param theta     RoPE base frequency (e.g. 10000.0)
 */
void rope_precompute_freqs(float *out, int head_dim, float theta);

/**
 * Apply Rotary Positional Embedding (RoPE) with full YaRN frequency interpolation.
 *
 * Uses NORMAL layout: interleaved pairs (v[2i], v[2i+1]) rotated by freq[i].
 * Supports YaRN per-dimension blending:
 *   - freq_scale = 1/factor (e.g. 0.025 for factor=40)
 *   - ext_factor = 1.0 for yarn, 0.0 for no extension
 *   - attn_factor = 1/(1 + log_mul * ln(1/freq_scale))
 *   - corr[0..1] = dimension blend boundary [low, high]
 *
 * @param q             Query vector (dim floats, modified in-place)
 * @param k             Key vector (kv_dim floats, modified in-place)
 * @param freq          Precomputed frequency table (head_dim/2 floats)
 * @param head_dim      Dimension per head
 * @param pos           Token position in the sequence
 * @param n_heads       Number of query heads
 * @param n_kv_heads    Number of KV heads
 * @param freq_scale    YaRN frequency scale (1/rope_factor). 1.0 = no scaling.
 * @param ext_factor    YaRN extension factor (0=linear, 1=full YaRN)
 * @param attn_factor   YaRN amplitude correction factor
 * @param corr          YaRN correction dimension bounds [low, high]
 */
void apply_rope(float *q, float *k, const float *freq, int head_dim, int pos, int n_heads,
                int n_kv_heads, float freq_scale, float ext_factor, float attn_factor,
                const float corr[2]);

/* Pair layout of the rotary embedding — which two dimensions rotate together.
 * NORM: (v[2i], v[2i+1]) — llama-family GGUFs (the converter permutes Q/K).
 * NEOX: (v[i], v[i+d/2]) — "rotate half": Qwen2/3, Phi, Gemma, StableLM,
 * BitNet b1.58 GGUFs, ... (llama.cpp's LLAMA_ROPE_TYPE_NEOX group). */
typedef enum { ROPE_LAYOUT_NORM = 0, ROPE_LAYOUT_NEOX = 1 } RopeLayout;

/* apply_rope with an explicit layout (apply_rope == ROPE_LAYOUT_NORM). */
void apply_rope_ex(float *q, float *k, const float *freq, int head_dim, int pos, int n_heads,
                   int n_kv_heads, float freq_scale, float ext_factor, float attn_factor,
                   const float corr[2], int layout);

/* Layout a GGUF general.architecture string uses (mirrors llama.cpp's
 * llama_model_rope_type for the architectures this engine can load). */
int rope_layout_for_arch(const char *arch);
const char *rope_layout_name(int layout);
int rope_layout_parse(const char *s); /* "norm" | "neox" -> layout, -1 otherwise */

/* YaRN correction dimension bounds of the blend zone: corr[0] = low, corr[1] = high. Dimensions
 * below low are fully extrapolated (no scaling), above high fully interpolated (scaled by
 * freq_scale). Matches ggml_rope_yarn_corr_dims() from llama.cpp exactly. */
void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast,
                         float beta_slow, float corr[2]);

/* The corr[] a model's attention passes to apply_rope_ex: the YaRN bounds when the config enables
 * the extension (rope_yarn_ext_factor != 0), {0, 0} otherwise. */
void rope_corr_from_config(const Config *cfg, int head_dim, float corr[2]);

#endif /* TN_ROPE_H */
