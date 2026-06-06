#include "core/config.h"
#include "core/platform.h"
#include <string.h>
#include <stdio.h>

TernaryError config_read(Config *cfg, const void *mapped_ptr, size_t file_size) {
    /* Need at least magic + version + Config */
    size_t header_size = sizeof(tn_u32) * 2 + sizeof(Config);
    TN_CHECK(file_size >= header_size, TN_ERR_INVALID_CONFIG);

    const tn_u8 *ptr = (const tn_u8 *)mapped_ptr;

    /* Check magic number */
    tn_u32 magic;
    memcpy(&magic, ptr, sizeof(magic));
    TN_CHECK(magic == TN_MAGIC, TN_ERR_INVALID_MAGIC);
    ptr += sizeof(magic);

    /* Check version */
    tn_u32 version;
    memcpy(&version, ptr, sizeof(version));
    TN_CHECK(version == TN_VERSION, TN_ERR_VERSION_MISMATCH);
    ptr += sizeof(version);

    /* Read config struct */
    /* Read fixed-size config fields (40 bytes total, matches on-disk .bin layout) */
    memcpy(cfg, ptr, 40);
    ptr += 40;
    cfg->bos_token_id = -1;  /* not stored in .bin format; set by GGUF loader */
    cfg->eos_token_id = -1;  /* not stored in .bin format; set by GGUF loader */
    cfg->rms_norm_eps = 1e-5f; /* default; overridden by GGUF loader */

    /* RoPE scaling: .bin format predates YaRN — set safe defaults so apply_rope
     * behaves as standard (no-scaling) RoPE.  attn_factor=1.0 is critical: if left
     * at 0 every rotated Q/K value is multiplied by 0 → complete garbage output. */
    cfg->rope_freq_scale      = 1.0f;  /* no linear scaling */
    cfg->rope_yarn_ext_factor = 0.0f;  /* no YaRN interpolation */
    cfg->rope_yarn_attn_factor= 1.0f;  /* amplitude scale = 1 (pass-through) */
    cfg->rope_yarn_beta_fast  = 32.0f;
    cfg->rope_yarn_beta_slow  = 1.0f;
    cfg->rope_orig_ctx_len    = cfg->seq_len;
    cfg->rope_yarn_log_mul    = 0.0f;

    /* Perform basic validation */
    TN_CHECK(cfg->dim > 0 && cfg->n_layers > 0, TN_ERR_INVALID_CONFIG);
    TN_CHECK(cfg->dim % cfg->n_heads == 0, TN_ERR_INVALID_CONFIG);
    TN_CHECK(cfg->n_heads % cfg->n_kv_heads == 0, TN_ERR_INVALID_CONFIG);

    return TN_OK;
}

void config_print(const Config *cfg) {
    printf("Model Configuration:\n");
    printf("  dim:        %d\n", cfg->dim);
    printf("  hidden_dim: %d\n", cfg->hidden_dim);
    printf("  n_layers:   %d\n", cfg->n_layers);
    printf("  n_heads:    %d\n", cfg->n_heads);
    printf("  n_kv_heads: %d\n", cfg->n_kv_heads);
    printf("  vocab_size: %d\n", cfg->vocab_size);
    printf("  seq_len:    %d\n", cfg->seq_len);
    printf("  act_type:   %d (%s)\n", cfg->act_type, cfg->act_type == 1 ? "ReLU2" : "SiLU");
    printf("  rope_theta: %.1f\n", cfg->rope_theta);
    printf("  head_dim:   %d\n", config_head_dim(cfg));
    printf("  kv_dim:     %d\n", config_kv_dim(cfg));
}
