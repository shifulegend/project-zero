#include "runtime/run_state.h"

#include <stdio.h>
#include <stdlib.h> /* free() for k_rope_cache pointer array */
#include <string.h>

#include "core/log.h"
#include "core/platform.h"
#include "kv_cache/kv_quant_ops.h"
#include "kv_cache/kv_strategy.h"
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include "memory/file_backed.h"
#include "memory/safe_alloc.h" /* tn_alloc_too_large() — shared OOM guard */
#include "runtime/mamba_state.h"

/* Phase 31: carve `n` buffers of sizes[i] bytes out of one disk-backed
 * mapping in `dir`, each 64-byte aligned. On success out[i] point into
 * s->kvs.offload_base. */
static TernaryError kv_offload_carve(RunState *s, const char *dir, const size_t *sizes, void **out,
                                     int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        size_t a = (sizes[i] + (TN_SIMD_ALIGN - 1)) & ~(size_t)(TN_SIMD_ALIGN - 1);
        if (total + a < total) return TN_ERR_OOM;
        total += a;
    }
    tn_i64 disk = file_backed_free_space(dir);
    if (disk >= 0 && (tn_i64)total > disk) {
        TN_LOG_INF("kv_offload", "[kv-offload] need %.1f GiB in %s, only %.1f GiB free\n",
                   (double)total / 1073741824.0, dir, (double)disk / 1073741824.0);
        return TN_ERR_OOM;
    }
    FileBacked fb;
    TernaryError e = file_backed_alloc(&fb, dir, total);
    if (e != TN_OK) return e;
    s->kvs.offload_base  = fb.base;
    s->kvs.offload_bytes = fb.bytes;
    size_t off           = 0;
    for (int i = 0; i < n; i++) {
        out[i] = (char *)fb.base + off;
        off += (sizes[i] + (TN_SIMD_ALIGN - 1)) & ~(size_t)(TN_SIMD_ALIGN - 1);
    }
    return TN_OK;
}

TernaryError run_state_alloc(RunState *s, const Config *cfg, int max_seq_len) {
    return run_state_alloc_ex(s, cfg, max_seq_len, false, NULL);
}

/* Plan 8.5: quantized K/V rows plus F32 scales (i8 is 4x, i4 about 7x smaller than F32), carved
 * from the offload mapping or allocated. On failure the state is freed and the error returned. */
static TernaryError alloc_quantized_kv(RunState *s, const char *offload_dir, int qtype,
                                       size_t kv_cache_size, int head_dim, int row_bytes,
                                       int row_scales) {
    RunKvStore *kv    = &s->kvs;
    size_t n_rows     = kv_cache_size / (size_t)(head_dim > 0 ? head_dim : 1);
    size_t data_bytes = n_rows * (size_t)row_bytes;
    size_t scale_n    = n_rows * (size_t)row_scales;
    if (!offload_dir && tn_alloc_too_large(data_bytes, sizeof(tn_i8))) {
        run_state_free(s);
        return TN_ERR_OOM;
    }
    kv->type       = qtype;
    kv->row_bytes  = row_bytes;
    kv->row_scales = row_scales;
    if (offload_dir) {
        size_t sz[4]   = {data_bytes, data_bytes, scale_n * sizeof(float), scale_n * sizeof(float)};
        void *pp[4]    = {NULL, NULL, NULL, NULL};
        TernaryError e = kv_offload_carve(s, offload_dir, sz, pp, 4);
        if (e != TN_OK) {
            run_state_free(s);
            return e;
        }
        kv->key_cache_q   = (tn_i8 *)pp[0];
        kv->value_cache_q = (tn_i8 *)pp[1];
        kv->key_scale     = (float *)pp[2];
        kv->value_scale   = (float *)pp[3];
    } else {
        kv->key_cache_q   = (tn_i8 *)tn_aligned_calloc(data_bytes, sizeof(tn_i8), TN_SIMD_ALIGN);
        kv->value_cache_q = (tn_i8 *)tn_aligned_calloc(data_bytes, sizeof(tn_i8), TN_SIMD_ALIGN);
        kv->key_scale     = (float *)tn_aligned_calloc(scale_n, sizeof(float), TN_SIMD_ALIGN);
        kv->value_scale   = (float *)tn_aligned_calloc(scale_n, sizeof(float), TN_SIMD_ALIGN);
    }
    if (!run_kv_store_complete(&s->kvs)) {
        run_state_free(s);
        return TN_ERR_OOM;
    }
    return TN_OK;
}

/* Scratch buffers: 64-byte aligned for SIMD. False if any allocation failed. */
static bool alloc_scratch(RunState *s, const Config *cfg) {
    s->x   = (float *)tn_aligned_calloc(cfg->dim, sizeof(float), TN_SIMD_ALIGN);
    s->xb  = (float *)tn_aligned_calloc(cfg->dim, sizeof(float), TN_SIMD_ALIGN);
    s->xb2 = (float *)tn_aligned_calloc(cfg->dim, sizeof(float), TN_SIMD_ALIGN);
    s->hb  = (float *)tn_aligned_calloc(cfg->hidden_dim, sizeof(float), TN_SIMD_ALIGN);
    s->hb2 = (float *)tn_aligned_calloc(cfg->hidden_dim, sizeof(float), TN_SIMD_ALIGN);
    s->q   = (float *)tn_aligned_calloc(cfg->dim, sizeof(float), TN_SIMD_ALIGN);
    return s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q;
}

/* Attention scores (n_heads * max_seq_len) and the output logits. */
static bool alloc_scores_and_logits(RunState *s, const Config *cfg, int max_seq_len) {
    /* FM-005 fix: overflow-checked multiplication for att buffer */
    size_t att_count;
    if (tn_size_mul_overflow((size_t)cfg->n_heads, (size_t)max_seq_len, &att_count)) return false;
    if (tn_alloc_too_large(att_count, sizeof(float))) return false;
    s->att    = (float *)tn_aligned_calloc(att_count, sizeof(float), TN_SIMD_ALIGN);
    s->logits = (float *)tn_aligned_calloc(cfg->vocab_size, sizeof(float), TN_SIMD_ALIGN);
    return s->att && s->logits;
}

/* The F32 K and V caches, in RAM or carved from the offload mapping. */
static TernaryError alloc_f32_kv(RunState *s, const char *offload_dir, size_t kv_cache_size) {
    size_t kv_bytes;
    if (tn_size_mul_overflow(kv_cache_size, sizeof(float), &kv_bytes) ||
        (!offload_dir && tn_alloc_too_large(kv_cache_size, sizeof(float)))) {
        run_state_free(s);
        return TN_ERR_OOM;
    }
    if (offload_dir) {
        size_t sz[2]   = {kv_bytes, kv_bytes};
        void *pp[2]    = {NULL, NULL};
        TernaryError e = kv_offload_carve(s, offload_dir, sz, pp, 2);
        if (e != TN_OK) {
            run_state_free(s);
            return e;
        }
        s->key_cache   = (float *)pp[0];
        s->value_cache = (float *)pp[1];
    } else {
        s->key_cache   = (float *)tn_aligned_calloc(kv_cache_size, sizeof(float), TN_SIMD_ALIGN);
        s->value_cache = (float *)tn_aligned_calloc(kv_cache_size, sizeof(float), TN_SIMD_ALIGN);
    }
    if (s->key_cache && s->value_cache) return TN_OK;
    run_state_free(s);
    return TN_ERR_OOM;
}

/* The storage type for the generic KV cache and its row layout. i4 groups need
 * head_dim % TN_KV_I4_GROUP == 0; anything else falls back to i8. */
static int resolve_kv_type(int qtype, int head_dim, int *row_bytes, int *row_scales) {
    *row_bytes  = 0;
    *row_scales = 0;
    if (qtype == KV_TYPE_F32) return qtype;
    if (kvq_row_layout((KVCacheType)qtype, head_dim, row_bytes, row_scales) == 0) return qtype;
    TN_LOG_ERR("kv", "[kv] --kv-type %s needs head_dim %% %d == 0 (head_dim %d); using i8\n",
               kv_type_name((KVCacheType)qtype), TN_KV_I4_GROUP, head_dim);
    kvq_row_layout(KV_TYPE_I8, head_dim, row_bytes, row_scales);
    return KV_TYPE_I8;
}

/* KV Cache — transposed layout: [layer][head][pos][head_dim]
 * FM-002 fix: overflow-checked multiplication for all 4 factors. */
static TernaryError alloc_generic_kv(RunState *s, const Config *cfg, int head_dim, int max_seq_len,
                                     const KVStrategyOptions *kv_opt) {
    const char *offload_dir = kv_options_offload_dir(kv_opt);
    size_t kv_cache_size;
    if (tn_size_mul4((size_t)cfg->n_layers, (size_t)cfg->n_kv_heads, (size_t)max_seq_len,
                     (size_t)head_dim, &kv_cache_size)) {
        run_state_free(s);
        return TN_ERR_OOM;
    }
    int row_bytes, row_scales;
    int qtype = resolve_kv_type(kv_options_type(kv_opt), head_dim, &row_bytes, &row_scales);
    if (qtype == KV_TYPE_F32) return alloc_f32_kv(s, offload_dir, kv_cache_size);
    return alloc_quantized_kv(s, offload_dir, qtype, kv_cache_size, head_dim, row_bytes,
                              row_scales);
}

/* Precompute the RoPE frequency table — eliminates powf() from the hot loop. */
static bool alloc_rope_table(RunState *s, const Config *cfg, int head_dim) {
    s->rope_freq_len = head_dim / 2;
    s->rope_freq     = (float *)tn_aligned_calloc(s->rope_freq_len, sizeof(float), TN_SIMD_ALIGN);
    if (!s->rope_freq) return false;
    float theta = cfg->rope_theta > 0 ? cfg->rope_theta : 10000.0f;
    rope_precompute_freqs(s->rope_freq, head_dim, theta);
    return true;
}

TernaryError run_state_alloc_ex(RunState *s, const Config *cfg, int max_seq_len, bool skip_kv_cache,
                                const KVStrategyOptions *kv_opt) {
    memset(s, 0, sizeof(*s));
    attn_config_default(&s->attn);
    s->max_seq_len = max_seq_len;
    int head_dim   = config_head_dim(cfg);

    if (!alloc_scratch(s, cfg) || !alloc_scores_and_logits(s, cfg, max_seq_len)) {
        run_state_free(s);
        return TN_ERR_OOM;
    }

    /* Skipped entirely (not calloc'd, not calloc'd-then-freed — see run_state_alloc_ex's header
     * comment) when skip_kv_cache is set: the calloc's explicit memset would force real RSS
     * commit for the whole buffer even if freed moments later. */
    if (!skip_kv_cache) {
        TernaryError e = alloc_generic_kv(s, cfg, head_dim, max_seq_len, kv_opt);
        if (e != TN_OK) return e;
    }

    if (alloc_rope_table(s, cfg, head_dim)) return TN_OK;
    run_state_free(s);
    return TN_ERR_OOM;
}

void run_state_free(RunState *s) {
    if (s->mamba) {
        mamba_state_free(s->mamba);
        s->mamba = NULL;
    }
    if (s->kvs.offload_base) {
        /* KV buffers point into the disk-backed mapping, not the aligned heap. */
        FileBacked fb = {s->kvs.offload_base, s->kvs.offload_bytes};
        file_backed_free(&fb);
        s->kvs.offload_base = NULL;
        s->key_cache = s->value_cache = NULL;
        s->kvs.key_cache_q = s->kvs.value_cache_q = NULL;
        s->kvs.key_scale = s->kvs.value_scale = NULL;
    }
    tn_aligned_free(s->kvs.key_cache_q);
    s->kvs.key_cache_q = NULL;
    tn_aligned_free(s->kvs.value_cache_q);
    s->kvs.value_cache_q = NULL;
    tn_aligned_free(s->kvs.key_scale);
    s->kvs.key_scale = NULL;
    tn_aligned_free(s->kvs.value_scale);
    s->kvs.value_scale = NULL;
    tn_aligned_free(s->x);
    tn_aligned_free(s->xb);
    tn_aligned_free(s->xb2);
    tn_aligned_free(s->hb);
    tn_aligned_free(s->hb2);
    tn_aligned_free(s->q);
    tn_aligned_free(s->att);
    tn_aligned_free(s->logits);
    tn_aligned_free(s->key_cache);
    tn_aligned_free(s->value_cache);
    tn_aligned_free(s->rope_freq);
    tn_aligned_free(s->mla.mla_rope_freq);
    /* k_rope_cache is allocated per-layer by mla_run_state_alloc(); free if present */
    if (s->mla.k_rope_cache) {
        /* number of layers is unknown here; mla_run_state_alloc stores layers in cache[0..n-1].
     * We free the pointer array itself — the per-layer buffers are freed by
     * mla_run_state_free() which must be called before run_state_free(). */
        free(s->mla.k_rope_cache);
        s->mla.k_rope_cache = NULL;
    }
    memset(s, 0, sizeof(*s));
}

void run_state_kv_reserve(RunState *s, const Config *cfg, int n_pos) {
    if (!s || !cfg || s->kvs.offload_base) return;
    if (n_pos > s->max_seq_len) n_pos = s->max_seq_len;
    int from = s->kv_committed;
    if (n_pos <= from) return;
    int hd     = config_head_dim(cfg);
    size_t cnt = (size_t)(n_pos - from);
    for (int l = 0; l < cfg->n_layers; l++) {
        for (int h = 0; h < cfg->n_kv_heads; h++) {
            size_t row =
                ((size_t)l * (size_t)cfg->n_kv_heads + (size_t)h) * (size_t)s->max_seq_len +
                (size_t)from;
            if (s->kvs.type != KV_TYPE_F32) {
                if (!run_kv_store_complete(&s->kvs)) return;
                size_t rb = (size_t)s->kvs.row_bytes, rs = (size_t)s->kvs.row_scales;
                tn_mem_commit(s->kvs.key_cache_q + row * rb, cnt * rb);
                tn_mem_commit(s->kvs.value_cache_q + row * rb, cnt * rb);
                tn_mem_commit(s->kvs.key_scale + row * rs, cnt * rs * sizeof(float));
                tn_mem_commit(s->kvs.value_scale + row * rs, cnt * rs * sizeof(float));
            } else {
                if (!s->key_cache || !s->value_cache) return;
                tn_mem_commit(s->key_cache + row * (size_t)hd, cnt * (size_t)hd * sizeof(float));
                tn_mem_commit(s->value_cache + row * (size_t)hd, cnt * (size_t)hd * sizeof(float));
            }
        }
    }
    s->kv_committed = n_pos;
}

int run_state_kv_release(RunState *s, const Config *cfg) {
    if (!s || !cfg || s->kvs.offload_base || s->kv_committed <= 0) return 0;
    int n       = s->kv_committed;
    size_t rows = (size_t)s->max_seq_len * (size_t)cfg->n_layers * (size_t)cfg->n_kv_heads;
    int ok;
    if (s->kvs.type != KV_TYPE_F32) {
        if (!run_kv_store_complete(&s->kvs)) return 0;
        size_t rb = (size_t)s->kvs.row_bytes, rs = (size_t)s->kvs.row_scales;
        ok = tn_mem_release(s->kvs.key_cache_q, rows * rb) &
             tn_mem_release(s->kvs.value_cache_q, rows * rb) &
             tn_mem_release(s->kvs.key_scale, rows * rs * sizeof(float)) &
             tn_mem_release(s->kvs.value_scale, rows * rs * sizeof(float));
    } else {
        if (!s->key_cache || !s->value_cache) return 0;
        size_t bytes = rows * (size_t)config_head_dim(cfg) * sizeof(float);
        ok           = tn_mem_release(s->key_cache, bytes) & tn_mem_release(s->value_cache, bytes);
    }
    if (!ok) return 0;
    s->kv_committed = 0;
    return n;
}

void run_state_kv_prefetch_layer(const RunState *s, const Config *cfg, int layer, int n_pos) {
    if (!s || !s->kvs.offload_base || layer < 0 || layer >= cfg->n_layers || n_pos <= 0) return;
    int hd = config_head_dim(cfg);
    if (n_pos > s->max_seq_len) n_pos = s->max_seq_len;
    for (int h = 0; h < cfg->n_kv_heads; h++) {
        size_t row0 =
            ((size_t)layer * (size_t)cfg->n_kv_heads + (size_t)h) * (size_t)s->max_seq_len;
        if (s->kvs.type != KV_TYPE_F32) {
            size_t rb = (size_t)s->kvs.row_bytes, rs = (size_t)s->kvs.row_scales;
            file_backed_prefetch(s->kvs.key_cache_q + row0 * rb, (size_t)n_pos * rb);
            file_backed_prefetch(s->kvs.value_cache_q + row0 * rb, (size_t)n_pos * rb);
            file_backed_prefetch(s->kvs.key_scale + row0 * rs, (size_t)n_pos * rs * sizeof(float));
            file_backed_prefetch(s->kvs.value_scale + row0 * rs,
                                 (size_t)n_pos * rs * sizeof(float));
        } else {
            file_backed_prefetch(s->key_cache + row0 * (size_t)hd,
                                 (size_t)n_pos * (size_t)hd * sizeof(float));
            file_backed_prefetch(s->value_cache + row0 * (size_t)hd,
                                 (size_t)n_pos * (size_t)hd * sizeof(float));
        }
    }
}
