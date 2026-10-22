#ifndef TN_WATERMARK_H
#define TN_WATERMARK_H

/*
 * watermark.h — Phase 33: cryptographic, toggleable output watermarking.
 *
 * Green-list scheme (Kirchenbauer et al., 2023): for every generation step the
 * vocabulary is pseudo-randomly split into a "green" fraction `gamma` and a
 * "red" remainder, keyed on (secret key, previous token). Green logits get
 * `+delta` before sampling, so watermarked text over-represents green tokens.
 * Detection re-derives the same split from the key alone (no model needed) and
 * reports a one-proportion z-score: unwatermarked text scores ~0, watermarked
 * text scores high (z > ~4 is a strong signal).
 *
 * Self-contained module: no dependency on the transformer, tokenizer or
 * sampler — callers hand it logits (bias) or token IDs (detect). Off by default
 * everywhere (CLI --watermark, API "watermark" object, web UI toggle).
 */

#include <stdbool.h>
#include <stdint.h>

/* Documented defaults (paper's recommended operating point). */
#define WATERMARK_DEFAULT_GAMMA 0.25f
#define WATERMARK_DEFAULT_DELTA 2.0f

typedef struct {
    bool     enabled;
    uint64_t key;    /* secret key; derive from a passphrase via watermark_key_from_string() */
    float    gamma;  /* green-list fraction, (0, 1) */
    float    delta;  /* logit bias added to green tokens, >= 0 */
} WatermarkConfig;

typedef struct {
    int    n_scored; /* distinct (prev, token) bigrams scored (repeats count once) */
    int    n_green;  /* how many of those were green */
    double z_score;  /* (n_green - gamma*n) / sqrt(n*gamma*(1-gamma)); 0 when n_scored == 0 */
    double green_fraction;
} WatermarkResult;

/* Disabled config with the documented gamma/delta defaults and key 0. */
void watermark_config_default(WatermarkConfig *wm);

/* FNV-1a 64-bit hash of a passphrase -> key. NULL/"" -> 0. */
uint64_t watermark_key_from_string(const char *passphrase);

/* Clamp gamma into [0.01, 0.99] and delta into [0, 20]; returns wm for chaining. */
WatermarkConfig *watermark_config_sanitize(WatermarkConfig *wm);

/* 1 if `token` is on the green list given `prev_token` (prev < 0 allowed). */
int watermark_is_green(const WatermarkConfig *wm, int prev_token, int token);

/* Adds wm->delta to every green logit in place. No-op when !wm->enabled. */
void watermark_bias_logits(const WatermarkConfig *wm, float *logits,
                           int vocab_size, int prev_token);

/* Scores a token sequence. Works regardless of wm->enabled (only key/gamma are used). */
void watermark_detect(const WatermarkConfig *wm, const int *tokens, int n_tokens,
                      WatermarkResult *out);

#endif /* TN_WATERMARK_H */
