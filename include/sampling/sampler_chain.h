#ifndef TN_SAMPLER_CHAIN_H
#define TN_SAMPLER_CHAIN_H

/*
 * sampler_chain.h — configurable logit-processing + sampling pipeline.
 *
 * One place that turns raw logits into a token, driven entirely by a
 * SamplerSettings value (CLI flags / API request / web UI — never literals):
 *
 *   penalties (repetition, presence, frequency over the last N tokens)
 *   -> watermark bias (Phase 33, optional)
 *   -> greedy argmax when temperature <= 0
 *   -> top-k mask -> min-p mask -> temperature -> top-p -> sample
 *
 * With default settings (top_k=0, min_p=0, neutral penalties, no watermark)
 * the result is bit-identical to the original generate() cascade
 * (argmax | temperature+top-p | temperature+plain CDF), so golden outputs do
 * not move.
 */

#include "sampling/watermark.h"

typedef struct {
    float temperature;        /* 0 = greedy */
    float top_p;              /* 1 = disabled */
    int   top_k;              /* 0 = disabled */
    float min_p;              /* 0 = disabled; keep tokens with p >= min_p * p_max */
    float repetition_penalty; /* 1 = disabled (CTRL-style divide/multiply) */
    float presence_penalty;   /* 0 = disabled (OpenAI semantics) */
    float frequency_penalty;  /* 0 = disabled (OpenAI semantics) */
    int   penalty_last_n;     /* window for all penalties; 0 = disabled, -1 = whole history */
} SamplerSettings;

/* Neutral defaults: temperature 1, top_p 1, everything else disabled,
 * penalty_last_n 64. */
void sampler_settings_default(SamplerSettings *ss);

/* Clamp every field into its valid range (documented in the struct). */
void sampler_settings_sanitize(SamplerSettings *ss);

/* 1 when any penalty is active (lets callers skip history bookkeeping). */
int sampler_penalties_active(const SamplerSettings *ss);

/* Applies repetition/presence/frequency penalties in place using the last
 * `penalty_last_n` entries of history[0..n_history). */
void sampler_apply_penalties(float *logits, int vocab_size, const SamplerSettings *ss,
                             const int *history, int n_history);

/* Masks logits outside the top-k and below the min-p cut to -INFINITY.
 * `temperature` is the one that will be applied later (min-p is defined on
 * the tempered distribution). No-op for disabled settings. */
void sampler_apply_truncation(float *logits, int vocab_size, int top_k, float min_p,
                              float temperature);

/* Second half of the pipeline only (greedy | truncation -> temperature ->
 * top-p -> sample) — for callers that insert their own mask (e.g. the JSON
 * grammar) between the penalty/watermark stage and sampling. */
int sampler_chain_finish(float *logits, int vocab_size, const SamplerSettings *ss,
                         unsigned long long *rng_state);

/* Full pipeline. `logits` is modified in place. `wm` may be NULL. */
int sampler_chain_sample(float *logits, int vocab_size, const SamplerSettings *ss,
                         const int *history, int n_history,
                         const WatermarkConfig *wm, int prev_token,
                         unsigned long long *rng_state);

#endif /* TN_SAMPLER_CHAIN_H */
