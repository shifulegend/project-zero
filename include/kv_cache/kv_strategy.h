#ifndef TN_KV_STRATEGY_H
#define TN_KV_STRATEGY_H

#include "core/config.h"
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
 * Decision thresholds:
 *   free_ram > 16 GB  → KV_FULL_F32
 *   free_ram >  4 GB  → KV_QUANT_I8
 *   free_ram >  1 GB  → KV_SLIDING_I8 (window = min(seq_len, 2048))
 *   free_ram <= 1 GB  → KV_SLIDING_I4 (window = min(seq_len, 1024))
 *
 * @param cfg       Model configuration
 * @param free_ram  Available system RAM in bytes
 * @return          Strategy result with chosen strategy and max_seq_len
 */
KVStrategyResult select_kv_strategy(const Config *cfg, tn_i64 free_ram);

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
