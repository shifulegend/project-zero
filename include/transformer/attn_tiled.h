#ifndef TN_ATTN_TILED_H
#define TN_ATTN_TILED_H

/*
 * attn_tiled.h — Phase 27.3: tiled single-query attention with an online
 * softmax ("FlashAttention for CPU", decode form).
 *
 * The standard path writes a full score row (valid_ctx floats) per head,
 * runs a softmax pass over it, then a third pass over V. This kernel walks
 * the context in L1-sized tiles: scores for one tile live in a small stack
 * buffer, a running max/sum rescales the output accumulator when the max
 * grows, and V for the tile is consumed while its K rows are still hot.
 * No O(seq_len) score buffer, one pass over the context.
 *
 * Selection is per RunState (CLI --attn-kernel standard|tiled, --attn-tile N; see
 * core/attn_config.h, reported by GET /v1/config). Applies to the generic dense/GQA
 * single-token attention path.
 */

#include "core/attn_config.h"
#include "kv_cache/sliding_window.h"

#define ATTN_TILE_MAX 1024

/* tile <= 0 = derive from the L1d size (cache_probe.h); clamped to [8, ATTN_TILE_MAX]. */
void attn_kernel_configure(AttnConfig *c, AttnKernel kernel, int tile, int head_dim);
const char *attn_kernel_name(AttnKernel k);
/* Heads go to the thread pool from `n` valid positions (negative = never). */
void attn_parallel_min_ctx_set(AttnConfig *c, int n);

/* "standard" | "tiled" -> kernel; returns -1 for anything else. */
int attn_kernel_parse(const char *s);

/*
 * out[head_dim] = softmax(q·K^T * scale) · V over the valid context of one
 * (layer, kv_head) slab. k_head/v_head point at position 0 of that slab
 * (rows of head_dim floats, physical positions from sw_map_position()).
 */
void attn_head_tiled(float *out, const float *q, const float *k_head, const float *v_head,
                     int head_dim, const SlidingWindow *sw, int pos, int valid_ctx, float scale,
                     int tile);

#endif /* TN_ATTN_TILED_H */
