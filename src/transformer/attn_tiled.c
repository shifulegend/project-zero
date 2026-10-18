#include "transformer/attn_tiled.h"
#include "threading/cache_probe.h"
#include "math/simd_dispatch.h"
#include <math.h>
#include <string.h>

static AttnKernel g_kernel = ATTN_KERNEL_STANDARD;
static int g_tile = 64;
static int g_parallel_min_ctx = ATTN_PARALLEL_MIN_CTX_DEFAULT;

void attn_parallel_min_ctx_set(int n) { g_parallel_min_ctx = n < 0 ? 0 : n; }
int attn_parallel_min_ctx_get(void) { return g_parallel_min_ctx; }

void attn_kernel_configure(AttnKernel kernel, int tile, int head_dim) {
    g_kernel = kernel;
    if (tile <= 0) {
        /* K and V rows for one tile should sit in about half of L1d. */
        int row = head_dim > 0 ? head_dim * (int)sizeof(float) * 2 : 512;
        tile = tn_cache_rows_fitting_l1(row, 0.5);
    }
    if (tile < 8) tile = 8;
    if (tile > ATTN_TILE_MAX) tile = ATTN_TILE_MAX;
    g_tile = tile;
}

AttnKernel attn_kernel_get(void) { return g_kernel; }
int attn_tile_get(void) { return g_tile; }

const char *attn_kernel_name(AttnKernel k) {
    return k == ATTN_KERNEL_TILED ? "tiled" : "standard";
}

int attn_kernel_parse(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "standard") == 0) return ATTN_KERNEL_STANDARD;
    if (strcmp(s, "tiled") == 0) return ATTN_KERNEL_TILED;
    return -1;
}

void attn_head_tiled(float *out, const float *q, const float *k_head, const float *v_head,
                     int head_dim, const SlidingWindow *sw, int pos, int valid_ctx,
                     float scale, int tile) {
    float sc[ATTN_TILE_MAX];
    int phys[ATTN_TILE_MAX];
    if (tile <= 0 || tile > ATTN_TILE_MAX) tile = ATTN_TILE_MAX;
    memset(out, 0, (size_t)head_dim * sizeof(float));
    if (valid_ctx <= 0) return;

    float m = -INFINITY, l = 0.0f;
    for (int t0 = 0; t0 < valid_ctx; t0 += tile) {
        int n = valid_ctx - t0 < tile ? valid_ctx - t0 : tile;
        float tile_max = -INFINITY;
        for (int i = 0; i < n; i++) {
            int t = t0 + i;
            int logical = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
            phys[i] = sw_map_position(sw, logical);
            float s = tn_vec_dot(q, k_head + (size_t)phys[i] * head_dim, head_dim) * scale;
            sc[i] = s;
            if (s > tile_max) tile_max = s;
        }
        float m_new = tile_max > m ? tile_max : m;
        if (m != -INFINITY && m_new != m) {
            float corr = expf(m - m_new);
            l *= corr;
            for (int d = 0; d < head_dim; d++) out[d] *= corr;
        }
        for (int i = 0; i < n; i++) {
            float p = expf(sc[i] - m_new);
            l += p;
            tn_vec_saxpy(out, p, v_head + (size_t)phys[i] * head_dim, head_dim);
        }
        m = m_new;
    }
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = 0; d < head_dim; d++) out[d] *= inv;
}
