#include "kv_cache/kv_quant_ops.h"
#include <math.h>
#include <string.h>

int kv_type_parse(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "f32") == 0) return KV_TYPE_F32;
    if (strcmp(s, "i8") == 0) return KV_TYPE_I8;
    return -1;
}

const char *kv_type_name(KVCacheType t) {
    return t == KV_TYPE_I8 ? "I8" : "F32";
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
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += q[i] * (float)k[i];
        a1 += q[i + 1] * (float)k[i + 1];
        a2 += q[i + 2] * (float)k[i + 2];
        a3 += q[i + 3] * (float)k[i + 3];
    }
    for (; i < n; i++) a0 += q[i] * (float)k[i];
    return (a0 + a1) + (a2 + a3);
}

void kvq_axpy(float *out, float a, const tn_i8 *v, int n) {
    for (int i = 0; i < n; i++) out[i] += a * (float)v[i];
}
