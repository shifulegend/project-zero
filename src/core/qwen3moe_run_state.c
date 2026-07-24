/**
 * qwen3moe_run_state.c
 *
 * Allocates the per-layer KV cache and RoPE frequency table for Qwen3-MoE
 * models (e.g. Qwen3-30B-A3B). Every layer uses the same plain-GQA +
 * QK-norm attention shape (see qwen3moe_attention.c) with head_dim
 * independent of dim/n_heads — reusing RunState's generic key_cache/
 * value_cache/rope_freq, which assume config_head_dim()==dim/n_heads,
 * would silently corrupt every layer. Same class of bug mla_run_state.c
 * and qwen35_run_state.c already document and fix for their own
 * independent head-dim cases.
 */

#include "core/run_state.h"
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>

TernaryError qwen3moe_run_state_alloc(RunState *s, const Config *cfg,
                                       const MoEConfig *mc, int max_seq_len) {
    if (!mc || !mc->has_qk_norm) return TN_OK; /* not this arch — no-op */

    /* s->key_cache/value_cache should already be NULL here: main.c calls
     * run_state_alloc_ex(..., skip_kv_cache=mc->has_qk_norm) precisely so
     * the *generic* key_cache/value_cache (sized with
     * config_head_dim(cfg)==dim/n_heads — wrong head_dim for this arch,
     * and never read by qwen3moe_attention.c, which uses
     * qwen3moe_key_cache/qwen3moe_value_cache below instead) is never
     * allocated at all. This free is a defensive no-op if that invariant is
     * ever violated — mirrors q35_run_state_alloc's identical comment. */
    tn_aligned_free(s->key_cache);   s->key_cache   = NULL;
    tn_aligned_free(s->value_cache); s->value_cache = NULL;

    int n_layers = cfg->n_layers;
    int n_kv_h   = cfg->n_kv_heads;
    int head_dim = mc->attn_head_dim;

    if (n_layers <= 0 || n_kv_h <= 0 || head_dim <= 0 || max_seq_len <= 0)
        return TN_ERR_INVALID_WEIGHTS;

    float **kc = (float **)calloc((size_t)n_layers, sizeof(float *));
    float **vc = (float **)calloc((size_t)n_layers, sizeof(float *));
    if (!kc || !vc) { free(kc); free(vc); return TN_ERR_OOM; }
    s->qwen3moe_key_cache   = kc;
    s->qwen3moe_value_cache = vc;

    size_t kv_elems = (size_t)n_kv_h * (size_t)max_seq_len * (size_t)head_dim;
    for (int l = 0; l < n_layers; l++) {
        kc[l] = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
        vc[l] = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
        if (!kc[l] || !vc[l]) { qwen3moe_run_state_free(s, cfg); return TN_ERR_OOM; }
    }

    int freq_len = head_dim / 2;
    s->qwen3moe_rope_freq = (float *)tn_aligned_calloc((size_t)freq_len, sizeof(float), 64);
    if (!s->qwen3moe_rope_freq) { qwen3moe_run_state_free(s, cfg); return TN_ERR_OOM; }
    float theta = cfg->rope_theta > 0.0f ? cfg->rope_theta : 10000.0f;
    rope_precompute_freqs(s->qwen3moe_rope_freq, head_dim, theta);

    return TN_OK;
}

void qwen3moe_run_state_free(RunState *s, const Config *cfg) {
    if (!s) return;
    int n_layers = cfg ? cfg->n_layers : 0;
    if (s->qwen3moe_key_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->qwen3moe_key_cache[l]);
        free(s->qwen3moe_key_cache); s->qwen3moe_key_cache = NULL;
    }
    if (s->qwen3moe_value_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->qwen3moe_value_cache[l]);
        free(s->qwen3moe_value_cache); s->qwen3moe_value_cache = NULL;
    }
    if (s->qwen3moe_rope_freq) {
        tn_aligned_free(s->qwen3moe_rope_freq);
        s->qwen3moe_rope_freq = NULL;
    }
}
