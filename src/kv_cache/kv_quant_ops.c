#include "kv_cache/kv_quant_ops.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

int kv_type_parse(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "f32") == 0) return KV_TYPE_F32;
    if (strcmp(s, "i8") == 0) return KV_TYPE_I8;
    if (strcmp(s, "i4") == 0) return KV_TYPE_I4;
    return -1;
}

const char *kv_type_name(KVCacheType t) {
    return t == KV_TYPE_I8 ? "I8" : t == KV_TYPE_I4 ? "I4" : "F32";
}

void kvq_store_row(tn_i8 *dst, float *scale_out, const float *src, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < n; i++) {
        float q = src[i] * inv;
        dst[i] = (tn_i8)(q >= 0.0f ? (int)(q + 0.5f) : (int)(q - 0.5f));
    }
    *scale_out = scale;
}

float kvq_dot(const float *q, const tn_i8 *k, int n) {
    /* 16 independent lanes so the loop vectorises to one 512-bit (or two
     * 256-bit) FMA per step without -ffast-math; 4 scalar accumulators only
     * gave 4-wide code and made i8 KV slower than F32 (2026-10-19). */
    float acc[16] = {0};
    int i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; j++) acc[j] += q[i + j] * (float)k[i + j];
    float t = 0.0f;
    for (; i < n; i++) t += q[i] * (float)k[i];
    for (int j = 0; j < 16; j++) t += acc[j];
    return t;
}

void kvq_axpy(float *out, float a, const tn_i8 *v, int n) {
    for (int i = 0; i < n; i++) out[i] += a * (float)v[i];
}

/* ── int4 (groups of TN_KV_I4_GROUP) ─────────────────────────────────────
 * Asymmetric (Q4_1-style): per group a scale d and minimum m, code
 * q = round((x - m) / d) in 0..15, value = q*d + m. Scales array holds
 * (d, m) pairs. 2026-10-19: a symmetric -7..7 code cost +15-19 % perplexity
 * on SmolLM2 (keys have skewed channels); see docs/ai/decision-log.md. */

static void kvq4_store(uint8_t *dst, float *scales, const float *src, int n) {
    enum { H = TN_KV_I4_GROUP / 2 };
    for (int g = 0; g < n / TN_KV_I4_GROUP; g++) {
        const float *x = src + g * TN_KV_I4_GROUP;
        float lo = x[0], hi = x[0];
        for (int i = 1; i < TN_KV_I4_GROUP; i++) {
            if (x[i] < lo) lo = x[i];
            if (x[i] > hi) hi = x[i];
        }
        float d = (hi - lo) / 15.0f;
        float inv = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t *o = dst + g * H;
        for (int i = 0; i < H; i++) {
            int q0 = (int)lrintf((x[i] - lo) * inv), q1 = (int)lrintf((x[i + H] - lo) * inv);
            q0 = q0 < 0 ? 0 : q0 > 15 ? 15 : q0;
            q1 = q1 < 0 ? 0 : q1 > 15 ? 15 : q1;
            o[i] = (uint8_t)(q0 | (q1 << 4));
        }
        scales[2 * g] = d;
        scales[2 * g + 1] = lo;
    }
}

static float kvq4_dot(const float *q, const uint8_t *k, const float *scales, int n) {
    /* sum_i q_i * (d_g * c_i + m_g), one 16-lane accumulator for the whole
     * row and a single horizontal reduction at the end (a per-group
     * reduction cost more than the dot itself). */
    enum { H = TN_KV_I4_GROUP / 2 };
    float acc[16] = {0};
    for (int g = 0; g < n / TN_KV_I4_GROUP; g++) {
        const float *qq = q + g * TN_KV_I4_GROUP;
        const uint8_t *b = k + g * H;
        float d = scales[2 * g], m = scales[2 * g + 1];
        for (int j = 0; j < H; j++) acc[j] += qq[j] * ((float)(b[j] & 15) * d + m);
        for (int j = 0; j < H; j++) acc[j] += qq[j + H] * ((float)(b[j] >> 4) * d + m);
    }
    float t = 0.0f;
    for (int j = 0; j < 16; j++) t += acc[j];
    return t;
}

static void kvq4_axpy(float *out, float a, const uint8_t *v, const float *scales, int n) {
    enum { H = TN_KV_I4_GROUP / 2 };
    for (int g = 0; g < n / TN_KV_I4_GROUP; g++) {
        float ad = a * scales[2 * g], am = a * scales[2 * g + 1];
        const uint8_t *b = v + g * H;
        float *oo = out + g * TN_KV_I4_GROUP;
        for (int j = 0; j < H; j++) oo[j] += ad * (float)(b[j] & 15) + am;
        for (int j = 0; j < H; j++) oo[j + H] += ad * (float)(b[j] >> 4) + am;
    }
}

/* ── type dispatch ───────────────────────────────────────────────────────── */

int kvq_row_layout(KVCacheType t, int n, int *bytes, int *scales) {
    if (n <= 0) return -1;
    if (t == KV_TYPE_I8) { *bytes = n; *scales = 1; return 0; }
    if (t == KV_TYPE_I4 && n % TN_KV_I4_GROUP == 0) {
        *bytes = n / 2;
        *scales = 2 * (n / TN_KV_I4_GROUP);     /* (scale, min) per group */
        return 0;
    }
    return -1;
}

void kvq_store(KVCacheType t, tn_i8 *dst, float *scales, const float *src, int n) {
    if (t == KV_TYPE_I4) kvq4_store((uint8_t *)dst, scales, src, n);
    else kvq_store_row(dst, scales, src, n);
}

float kvq_dot_row(KVCacheType t, const float *q, const tn_i8 *k, const float *scales, int n) {
    if (t == KV_TYPE_I4) return kvq4_dot(q, (const uint8_t *)k, scales, n);
    return kvq_dot(q, k, n) * scales[0];
}

void kvq_axpy_row(KVCacheType t, float *out, float a, const tn_i8 *v, const float *scales, int n) {
    if (t == KV_TYPE_I4) kvq4_axpy(out, a, (const uint8_t *)v, scales, n);
    else kvq_axpy(out, a * scales[0], v, n);
}
