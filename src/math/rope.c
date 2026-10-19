#include "math/rope.h"
#include <math.h>
#include <string.h>

void rope_precompute_freqs(float *out, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        out[i] = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    }
}

/* YaRN ramp: 1 = extrapolate (use unscaled theta), 0 = interpolate (use scaled theta).
 * i0 is the raw dimension index; we use i0/2 as the half-index for comparison. */
static float rope_yarn_ramp_r(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / (high - low > 0.001f ? high - low : 0.001f);
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;
}

/* Apply NORMAL-type RoPE with full YaRN blending to a single vector `v` of size d.
 * freq must have d/2 entries. pos is the token position. */
static void rope_apply_head_yarn(float *v, const float *freq, int d, int pos,
                                  float freq_scale, float ext_factor, float attn_factor,
                                  const float corr[2]) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float theta_extrap = (float)pos * freq[i];
        float theta_interp = freq_scale * theta_extrap;

        float theta;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp = rope_yarn_ramp_r(corr[0], corr[1], 2 * i) * ext_factor;
            theta  = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        } else {
            theta = theta_interp;
        }

        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float v0 = v[2 * i];
        float v1 = v[2 * i + 1];
        v[2 * i]     = v0 * c - v1 * s;
        v[2 * i + 1] = v0 * s + v1 * c;
    }
}

/* NEOX ("rotate half") variant of rope_apply_head_yarn: dimension i pairs with
 * i + d/2. Same frequencies and YaRN blending; only the pairing differs. */
static void rope_apply_head_yarn_neox(float *v, const float *freq, int d, int pos,
                                       float freq_scale, float ext_factor, float attn_factor,
                                       const float corr[2]) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float theta_extrap = (float)pos * freq[i];
        float theta_interp = freq_scale * theta_extrap;
        float theta;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp = rope_yarn_ramp_r(corr[0], corr[1], 2 * i) * ext_factor;
            theta  = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        } else {
            theta = theta_interp;
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float v0 = v[i];
        float v1 = v[i + half];
        v[i]        = v0 * c - v1 * s;
        v[i + half] = v0 * s + v1 * c;
    }
}

void apply_rope_ex(float *q, float *k, const float *freq,
                   int head_dim, int pos, int n_heads, int n_kv_heads,
                   float freq_scale, float ext_factor, float attn_factor,
                   const float corr[2], int layout) {
    if (layout != ROPE_LAYOUT_NEOX) {
        apply_rope(q, k, freq, head_dim, pos, n_heads, n_kv_heads,
                   freq_scale, ext_factor, attn_factor, corr);
        return;
    }
    for (int h = 0; h < n_heads; h++)
        rope_apply_head_yarn_neox(q + h * head_dim, freq, head_dim, pos,
                                  freq_scale, ext_factor, attn_factor, corr);
    for (int h = 0; h < n_kv_heads; h++)
        rope_apply_head_yarn_neox(k + h * head_dim, freq, head_dim, pos,
                                  freq_scale, ext_factor, attn_factor, corr);
}

int rope_layout_for_arch(const char *arch) {
    /* llama.cpp src/llama-model.cpp, llama_model_rope_type(): NEOX group. */
    static const char *const neox[] = {
        "bitnet", "bitnet-b1.58", "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
        "stablelm", "phi2", "phi3", "phimoe", "gemma", "gemma2", "gemma3", "gemma3n",
        "starcoder2", "gptneox", "falcon", "olmo2", "olmoe", "openelm", "codeshell",
        "orion", "nemotron", "exaone", "exaone4", "minicpm3", "dbrx", "grok",
        "plamo", "plamo2", "dots1", "hunyuan-moe", "hunyuan-dense", "smallthinker",
        "seed_oss", "lfm2", NULL
    };
    if (!arch) return ROPE_LAYOUT_NORM;
    for (int i = 0; neox[i]; i++)
        if (strcmp(arch, neox[i]) == 0) return ROPE_LAYOUT_NEOX;
    return ROPE_LAYOUT_NORM;
}

const char *rope_layout_name(int layout) { return layout == ROPE_LAYOUT_NEOX ? "neox" : "norm"; }

int rope_layout_parse(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "norm") == 0) return ROPE_LAYOUT_NORM;
    if (strcmp(s, "neox") == 0) return ROPE_LAYOUT_NEOX;
    return -1;
}

void apply_rope(float *q, float *k, const float *freq,
                int head_dim, int pos, int n_heads, int n_kv_heads,
                float freq_scale, float ext_factor, float attn_factor,
                const float corr[2]) {
    for (int h = 0; h < n_heads; h++)
        rope_apply_head_yarn(q + h * head_dim, freq, head_dim, pos,
                             freq_scale, ext_factor, attn_factor, corr);
    for (int h = 0; h < n_kv_heads; h++)
        rope_apply_head_yarn(k + h * head_dim, freq, head_dim, pos,
                             freq_scale, ext_factor, attn_factor, corr);
}
