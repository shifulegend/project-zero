#ifndef TN_KV_STRATEGY_H
#define TN_KV_STRATEGY_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/platform.h"

/**
 * KV cache strategy tiers, ordered by memory usage (low → high quality).
 */
typedef enum {
  KV_SLIDING_I4, /* Lowest memory: sliding window + int4 quantization */
  KV_SLIDING_I8, /* Low memory: sliding window + int8 quantization */
  KV_QUANT_I4,   /* Medium: full context + int4 quantization */
  KV_QUANT_I8,   /* Good: full context + int8 quantization */
  KV_FULL_F32    /* Best quality: full context + float32 (no compression) */
} KVStrategy;

/**
 * Result from strategy selection — includes both the chosen
 * strategy and the computed maximum sequence length.
 */
typedef struct {
  KVStrategy strategy;
  int max_seq_len; /* Effective max sequence length for this strategy */
} KVStrategyResult;

/**
 * Select the optimal KV cache strategy based on model config and available RAM.
 *
 * Decision thresholds (see kv_strategy.c for the exact GB_* constants):
 *   free_ram > 32 GB  → KV_FULL_F32
 *   free_ram >  8 GB  → KV_QUANT_I8
 *   free_ram >  5 GB  → KV_SLIDING_I8 (window = min(seq_len, 1024))
 *   free_ram >  2 GB  → KV_SLIDING_I4 (window = min(seq_len, 1024))
 *   free_ram <= 2 GB  → KV_SLIDING_I4 (window = min(seq_len, 512))
 * plus a safety cap (see kv_strategy.c) that further reduces max_seq_len if
 * the resulting F32 KV cache would exceed 60% of free_ram.
 *
 * @param cfg       Model configuration
 * @param free_ram  Available system RAM in bytes
 * @param mc        Optional MoE/hybrid config (NULL for dense models). When
 *                  mc->attn_head_dim > 0 (qwen35 hybrid or qwen3moe — see
 *                  moe_config.h), the safety cap uses that instead of
 *                  config_head_dim(cfg) (== dim/n_heads), since both those
 *                  architectures' real per-head KV dimension is independent
 *                  of dim/n_heads and the naive formula silently
 *                  underestimates true KV cache cost (confirmed: 2x too
 *                  small for Qwen3-30B-A3B, 64 vs the real 128 — GitHub
 *                  issue #32, still segfaulting after the qwen3moe loader
 *                  fix because this safety cap couldn't see the real
 *                  head_dim). MLA (has_mla) is a different, smaller-than-
 *                  naive cache shape (a low-rank latent, not
 *                  n_kv_heads*head_dim) and is intentionally NOT covered by
 *                  this parameter — the naive formula over-estimates its
 *                  cost, which is conservative (smaller max_seq_len than
 *                  necessary) rather than unsafe, and fixing that precisely
 *                  needs MLA-specific accounting, flagged as a separate
 *                  follow-up rather than solved here.
 * @return          Strategy result with chosen strategy and max_seq_len
 */
KVStrategyResult select_kv_strategy(const Config *cfg, tn_i64 free_ram, const MoEConfig *mc);

/**
 * Get a human-readable name for a KV strategy.
 */
const char *kv_strategy_name(KVStrategy s);

/**
 * Probe available system RAM.
 * Returns bytes of free RAM, or -1 on failure.
 */
tn_i64 tn_get_free_ram(void);

#endif /* TN_KV_STRATEGY_H */
