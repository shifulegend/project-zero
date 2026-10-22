#include "sampling/watermark.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

void watermark_config_default(WatermarkConfig *wm) {
    if (!wm) return;
    wm->enabled = false;
    wm->key     = 0;
    wm->gamma   = WATERMARK_DEFAULT_GAMMA;
    wm->delta   = WATERMARK_DEFAULT_DELTA;
}

uint64_t watermark_key_from_string(const char *passphrase) {
    if (!passphrase || !*passphrase) return 0;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const unsigned char *p = (const unsigned char *)passphrase; *p; p++) {
        h ^= *p;
        h *= 0x100000001b3ULL;
    }
    return h;
}

WatermarkConfig *watermark_config_sanitize(WatermarkConfig *wm) {
    if (!wm) return wm;
    if (!(wm->gamma >= 0.01f)) wm->gamma = 0.01f;   /* also catches NaN */
    if (wm->gamma > 0.99f)     wm->gamma = 0.99f;
    if (!(wm->delta >= 0.0f))  wm->delta = 0.0f;
    if (wm->delta > 20.0f)     wm->delta = 20.0f;
    return wm;
}

/* splitmix64 finalizer — strong avalanche, cheap enough to run per vocab entry. */
static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t context_seed(const WatermarkConfig *wm, int prev_token) {
    return mix64(wm->key ^ mix64((uint64_t)(int64_t)prev_token));
}

static int is_green_seeded(uint64_t seed, int token, uint64_t threshold) {
    return mix64(seed ^ (uint64_t)(uint32_t)token) < threshold;
}

static uint64_t gamma_threshold(float gamma) {
    /* gamma in (0,1) -> fraction of the 64-bit range. 2^64 as a double is exact. */
    double t = (double)gamma * 18446744073709551616.0;
    if (t >= 18446744073709551615.0) return UINT64_MAX;
    if (t <= 0.0) return 0;
    return (uint64_t)t;
}

int watermark_is_green(const WatermarkConfig *wm, int prev_token, int token) {
    if (!wm) return 0;
    return is_green_seeded(context_seed(wm, prev_token), token, gamma_threshold(wm->gamma));
}

void watermark_bias_logits(const WatermarkConfig *wm, float *logits,
                           int vocab_size, int prev_token) {
    if (!wm || !wm->enabled || !logits || vocab_size <= 0 || wm->delta == 0.0f) return;
    uint64_t seed = context_seed(wm, prev_token);
    uint64_t thr  = gamma_threshold(wm->gamma);
    for (int i = 0; i < vocab_size; i++) {
        if (is_green_seeded(seed, i, thr)) logits[i] += wm->delta;
    }
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

void watermark_detect(const WatermarkConfig *wm, const int *tokens, int n_tokens,
                      WatermarkResult *out) {
    if (!out) return;
    out->n_scored = 0;
    out->n_green = 0;
    out->z_score = 0.0;
    out->green_fraction = 0.0;
    if (!wm || !tokens || n_tokens < 2) return;

    /* Score each distinct (prev, token) bigram once. Repeated phrases are
     * common in model output and would otherwise count the same green/red
     * draw many times, breaking the independence the z-test assumes
     * (2026-10-18: un-watermarked repetitive text scored z≈4.6 before this). */
    int n_pairs = n_tokens - 1;
    uint64_t *pairs = (uint64_t *)malloc((size_t)n_pairs * sizeof(uint64_t));
    if (!pairs) return;
    for (int i = 0; i < n_pairs; i++)
        pairs[i] = ((uint64_t)(uint32_t)tokens[i] << 32) | (uint32_t)tokens[i + 1];
    qsort(pairs, (size_t)n_pairs, sizeof(uint64_t), cmp_u64);

    uint64_t thr = gamma_threshold(wm->gamma);
    for (int i = 0; i < n_pairs; i++) {
        if (i > 0 && pairs[i] == pairs[i - 1]) continue;
        int prev = (int)(int32_t)(uint32_t)(pairs[i] >> 32);
        int tok  = (int)(int32_t)(uint32_t)(pairs[i] & 0xffffffffu);
        out->n_scored++;
        if (is_green_seeded(context_seed(wm, prev), tok, thr)) out->n_green++;
    }
    free(pairs);

    double n = (double)out->n_scored;
    double g = (double)wm->gamma;
    out->green_fraction = (double)out->n_green / n;
    double denom = sqrt(n * g * (1.0 - g));
    out->z_score = denom > 0.0 ? ((double)out->n_green - g * n) / denom : 0.0;
}
