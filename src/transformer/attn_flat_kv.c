/*
 * src/transformer/attn_flat_kv.c
 *
 * Head scan and KV store over a flat per-layer cache, shared by the Qwen3-MoE and Qwen3.5/3.6
 * full-attention layers (include/transformer/attn_flat_kv.h).
 */

#include "transformer/attn_flat_kv.h"

#include <math.h>
#include <string.h>

#include "math/simd_core_ops.h"
#include "math/simd_vec_ops.h"

void attn_flat_store(float *key_layer, float *value_layer, int n_kv_h, int max_seq, int head_dim,
                     int mapped_pos, const float *k, const float *v) {
    for (int kh = 0; kh < n_kv_h; kh++) {
        size_t off = ((size_t)kh * max_seq + (size_t)mapped_pos) * head_dim;
        memcpy(&key_layer[off], &k[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
        memcpy(&value_layer[off], &v[(size_t)kh * head_dim], (size_t)head_dim * sizeof(float));
    }
}

void attn_flat_head(const SlidingWindow *sw, const float *key_layer, const float *value_layer,
                    int kh, int max_seq, int head_dim, int pos, const float *q_h, float *att,
                    float *out_h) {
    int valid_ctx     = sw_valid_count(sw, pos);
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t < valid_ctx; t++) {
        int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t = sw_map_position(sw, hist);
        size_t off   = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
        att[t]       = tn_vec_dot(q_h, &key_layer[off], head_dim) * inv_sqrt_hd;
    }
    tn_softmax(att, valid_ctx);

    memset(out_h, 0, (size_t)head_dim * sizeof(float));
    for (int t = 0; t < valid_ctx; t++) {
        int hist     = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t = sw_map_position(sw, hist);
        size_t off   = ((size_t)kh * max_seq + (size_t)mapped_t) * head_dim;
        tn_vec_saxpy(out_h, att[t], &value_layer[off], head_dim);
    }
}
