#ifndef TN_CORE_ATTN_CONFIG_H
#define TN_CORE_ATTN_CONFIG_H

/*
 * How one RunState computes attention (CLI --attn-kernel, --attn-tile, --attn-parallel-min-ctx;
 * reported by GET /v1/config). A zeroed RunState gets attn_config_default() when it is allocated;
 * the session overrides it on every state it owns (transformer/attn_tiled.h configures it).
 * Kept in core/ so run_state.h can embed it without including the transformer layer.
 */

typedef enum { ATTN_KERNEL_STANDARD = 0, ATTN_KERNEL_TILED = 1 } AttnKernel;

/* Phase 27: attention heads are dispatched over the thread pool once the valid context reaches
 * this many positions (0 = never). 2026-10-18 sweep, SmolLM2-135M, 4 threads: 64 -> 92.6 tok/s
 * vs 256 -> 77.8 (5-run means). */
#define ATTN_PARALLEL_MIN_CTX_DEFAULT 64
#define ATTN_TILE_DEFAULT 64

typedef struct {
    AttnKernel kernel;
    int tile;
    int parallel_min_ctx; /* heads go to the pool from this many valid positions; 0 = never */
} AttnConfig;

static inline void attn_config_default(AttnConfig *c) {
    c->kernel           = ATTN_KERNEL_STANDARD;
    c->tile             = ATTN_TILE_DEFAULT;
    c->parallel_min_ctx = ATTN_PARALLEL_MIN_CTX_DEFAULT;
}

#endif /* TN_CORE_ATTN_CONFIG_H */
