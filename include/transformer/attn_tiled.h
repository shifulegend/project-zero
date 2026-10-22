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
 * Selection is process-wide (CLI --attn-kernel standard|tiled, --attn-tile N;
 * reported by GET /v1/config). Applies to the generic dense/GQA single-token
 * attention path.
 */

#include "kv_cache/sliding_window.h"

typedef enum {
    ATTN_KERNEL_STANDARD = 0,
    ATTN_KERNEL_TILED    = 1
} AttnKernel;

#define ATTN_TILE_MAX 1024

/* tile <= 0 = derive from the L1d size (cache_probe.h); clamped to
 * [8, ATTN_TILE_MAX]. */
void        attn_kernel_configure(AttnKernel kernel, int tile, int head_dim);
AttnKernel  attn_kernel_get(void);
int         attn_tile_get(void);
const char *attn_kernel_name(AttnKernel k);
/* Phase 27: attention heads are dispatched over the thread pool once the
 * valid context reaches this many positions (0 = never). Default
 * ATTN_PARALLEL_MIN_CTX_DEFAULT; CLI --attn-parallel-min-ctx. */
#define ATTN_PARALLEL_MIN_CTX_DEFAULT 64  /* 2026-10-18 sweep, SmolLM2-135M, 4 threads: 64 -> 92.6 tok/s vs 256 -> 77.8 (5-run means) */
void attn_parallel_min_ctx_set(int n);
int  attn_parallel_min_ctx_get(void);

/* "standard" | "tiled" -> kernel; returns -1 for anything else. */
int         attn_kernel_parse(const char *s);

/*
 * out[head_dim] = softmax(q·K^T * scale) · V over the valid context of one
 * (layer, kv_head) slab. k_head/v_head point at position 0 of that slab
 * (rows of head_dim floats, physical positions from sw_map_position()).
 */
void attn_head_tiled(float *out, const float *q, const float *k_head, const float *v_head,
                     int head_dim, const SlidingWindow *sw, int pos, int valid_ctx,
                     float scale, int tile);

#endif /* TN_ATTN_TILED_H */
