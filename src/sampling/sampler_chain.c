#include "sampling/sampler_chain.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "math/simd_dispatch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void sampler_settings_default(SamplerSettings *ss) {
    if (!ss) return;
    ss->temperature        = 1.0f;
    ss->top_p              = 1.0f;
    ss->top_k              = 0;
    ss->min_p              = 0.0f;
    ss->repetition_penalty = 1.0f;
    ss->presence_penalty   = 0.0f;
    ss->frequency_penalty  = 0.0f;
    ss->penalty_last_n     = 64;
}

void sampler_settings_sanitize(SamplerSettings *ss) {
    if (!ss) return;
    if (!(ss->temperature >= 0.0f)) ss->temperature = 0.0f;
    if (ss->temperature > 5.0f)     ss->temperature = 5.0f;
    if (!(ss->top_p > 0.0f) || ss->top_p > 1.0f) ss->top_p = 1.0f;
    if (ss->top_k < 0)              ss->top_k = 0;
    if (!(ss->min_p >= 0.0f) || ss->min_p >= 1.0f) ss->min_p = 0.0f;
    if (!(ss->repetition_penalty > 0.0f)) ss->repetition_penalty = 1.0f;
    if (ss->repetition_penalty > 10.0f)   ss->repetition_penalty = 10.0f;
    if (!(ss->presence_penalty >= -2.0f) || ss->presence_penalty > 2.0f)
        ss->presence_penalty = 0.0f;
    if (!(ss->frequency_penalty >= -2.0f) || ss->frequency_penalty > 2.0f)
        ss->frequency_penalty = 0.0f;
    if (ss->penalty_last_n < -1) ss->penalty_last_n = -1;
}

int sampler_penalties_active(const SamplerSettings *ss) {
    if (!ss || ss->penalty_last_n == 0) return 0;
    return ss->repetition_penalty != 1.0f || ss->presence_penalty != 0.0f ||
           ss->frequency_penalty != 0.0f;
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

void sampler_apply_penalties(float *logits, int vocab_size, const SamplerSettings *ss,
                             const int *history, int n_history) {
    if (!logits || !history || n_history <= 0 || !sampler_penalties_active(ss)) return;
    int n = n_history;
    if (ss->penalty_last_n > 0 && ss->penalty_last_n < n) n = ss->penalty_last_n;
    const int *window = history + (n_history - n);

    /* Sort a copy so each distinct token is visited once with its count. */
    int *sorted = (int *)malloc((size_t)n * sizeof(int));
    if (!sorted) return;
    memcpy(sorted, window, (size_t)n * sizeof(int));
    qsort(sorted, (size_t)n, sizeof(int), cmp_int);

    for (int i = 0; i < n;) {
        int tok = sorted[i];
        int count = 0;
        while (i < n && sorted[i] == tok) { count++; i++; }
        if (tok < 0 || tok >= vocab_size) continue;
        float l = logits[tok];
        if (ss->repetition_penalty != 1.0f)
            l = (l > 0.0f) ? l / ss->repetition_penalty : l * ss->repetition_penalty;
        l -= ss->frequency_penalty * (float)count + ss->presence_penalty;
        logits[tok] = l;
    }
    free(sorted);
}

/* k-th largest value (1-based) via a bounded insertion buffer, k small. */
static float kth_largest(const float *logits, int vocab_size, int k) {
    float *best = (float *)malloc((size_t)k * sizeof(float));
    if (!best) return -INFINITY;
    for (int i = 0; i < k; i++) best[i] = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (v <= best[k - 1]) continue;
        int j = k - 1;
        while (j > 0 && v > best[j - 1]) { best[j] = best[j - 1]; j--; }
        best[j] = v;
    }
    float r = best[k - 1];
    free(best);
    return r;
}

void sampler_apply_truncation(float *logits, int vocab_size, int top_k, float min_p,
                              float temperature) {
    if (!logits || vocab_size <= 0) return;
    if (top_k > 0 && top_k < vocab_size) {
        float cut = kth_largest(logits, vocab_size, top_k);
        /* Ties at the cut are kept — at most a handful of extra candidates. */
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < cut) logits[i] = -INFINITY;
    }
    if (min_p > 0.0f && min_p < 1.0f) {
        float mx = -INFINITY;
        for (int i = 0; i < vocab_size; i++) if (logits[i] > mx) mx = logits[i];
        /* p_i >= min_p * p_max  <=>  l_i/T >= l_max/T + ln(min_p) */
        float t = temperature > 0.0f ? temperature : 1.0f;
        float cut = mx + t * logf(min_p);
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < cut) logits[i] = -INFINITY;
    }
}

int sampler_chain_finish(float *logits, int vocab_size, const SamplerSettings *ss,
                         unsigned long long *rng_state) {
    SamplerSettings def;
    if (!ss) { sampler_settings_default(&def); ss = &def; }

    float temperature = ss->temperature;
    if (temperature <= 0.0f || temperature < 1e-6f)
        return sample_argmax(logits, vocab_size);

    sampler_apply_truncation(logits, vocab_size, ss->top_k, ss->min_p, temperature);
    apply_temperature(logits, vocab_size, temperature);
    if (ss->top_p < 1.0f && ss->top_p > 0.0f)
        return sample_top_p(logits, vocab_size, ss->top_p, rng_state);

    tn_softmax(logits, vocab_size);
    float r = rng_float(rng_state);
    float cdf = 0.0f;
    int next = vocab_size - 1; /* fallback */
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) { next = i; break; }
    }
    return next;
}

int sampler_chain_sample(float *logits, int vocab_size, const SamplerSettings *ss,
                         const int *history, int n_history,
                         const WatermarkConfig *wm, int prev_token,
                         unsigned long long *rng_state) {
    sampler_apply_penalties(logits, vocab_size, ss, history, n_history);
    watermark_bias_logits(wm, logits, vocab_size, prev_token);
    return sampler_chain_finish(logits, vocab_size, ss, rng_state);
}
