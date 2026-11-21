#ifndef TN_ATTN_FLAT_KV_H
#define TN_ATTN_FLAT_KV_H

/* Attention over a flat per-layer KV cache laid out [kv_head][pos][head_dim], as the Qwen3-MoE and
 * Qwen3.5/3.6 full-attention layers keep it (the generic path's cache is the interleaved
 * KV_CACHE_IDX layout, with optional quantisation: see attention.c). */

#include "kv_cache/sliding_window.h"

/* Stores one row's K and V (n_kv_h heads of head_dim floats each) at the physical slot
 * mapped_pos. */
void attn_flat_store(float *key_layer, float *value_layer, int n_kv_h, int max_seq, int head_dim,
                     int mapped_pos, const float *k, const float *v);

/* One query head q_h (head_dim floats) against kv head kh's cache over the valid window at `pos`:
 * scores q . K[t] / sqrt(head_dim) into att, softmax, then the weighted sum of V into out_h. */
void attn_flat_head(const SlidingWindow *sw, const float *key_layer, const float *value_layer,
                    int kh, int max_seq, int head_dim, int pos, const float *q_h, float *att,
                    float *out_h);

#endif /* TN_ATTN_FLAT_KV_H */
