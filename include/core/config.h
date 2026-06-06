#ifndef TN_CONFIG_H
#define TN_CONFIG_H

#include "core/error.h"
#include <stdio.h>

typedef struct {
    int dim;           /* transformer dimension (e.g., 4096) */
    int hidden_dim;    /* FFN intermediate dimension (e.g., 14336) */
    int n_layers;      /* number of transformer layers (e.g., 32) */
    int n_heads;       /* number of attention heads (e.g., 32) */
    int n_kv_heads;    /* number of KV heads for GQA (e.g., 8) */
    int vocab_size;    /* vocabulary size (e.g., 32000) */
    int seq_len;       /* maximum sequence length (e.g., 2048) */
    int act_type;      /* activation: 0 = SiLU, 1 = ReLU2 */
    float rope_theta;  /* RoPE base frequency */
    int scale_mode;    /* 0 = Matrix-scaled Ternary, 1 = Raw Float32 */
    int bos_token_id;  /* BOS token ID to prepend (-1 = none, auto-set from model) */
    /* YaRN RoPE scaling parameters (read from GGUF rope.scaling.* keys) */
    float rope_freq_scale;      /* 1/factor, e.g. 0.025 for factor=40 */
    float rope_yarn_ext_factor; /* 0=none, 1=full YaRN interpolation */
    float rope_yarn_attn_factor;/* RoPE amplitude scale = 1/(1+0.1*ln(factor)), so after
                                   GGML correction (* (1+0.1*ln(factor))) mscale → 1.0 */
    float rope_yarn_beta_fast;  /* beta_fast for corr_dims (default 32) */
    float rope_yarn_beta_slow;  /* beta_slow for corr_dims (default 1) */
    int   rope_orig_ctx_len;    /* original training context length */
    float rope_yarn_log_mul;    /* raw yarn_log_multiplier from GGUF (e.g. 0.0707).
                                   Used for kq_scale: iscale = (1+log_mul*ln(factor))^2
                                   / sqrt(head_dim), matching llama.cpp deepseek2.cpp */
    float rms_norm_eps;         /* RMSNorm epsilon (default 1e-5; DeepSeek needs 1e-6) */
    int   eos_token_id;         /* EOS token ID (-1 = none) */
} Config;

/**
 * Read config from the first bytes of a mapped file.
 * Validates all fields are positive and within sane bounds.
 */
TernaryError config_read(Config *cfg, const void *mapped_ptr, size_t file_size);

/**
 * Print config to stdout for boot diagnostics.
 */
void config_print(const Config *cfg);

/**
 * Returns the head dimension (dim / n_heads).
 */
static inline int config_head_dim(const Config *cfg) {
    return cfg->dim / cfg->n_heads;
}

/**
 * Returns the KV dimension per layer.
 */
static inline int config_kv_dim(const Config *cfg) {
    return (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
}

#endif /* TN_CONFIG_H */
