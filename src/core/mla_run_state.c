/**
 * mla_run_state.c — Phase 17.7
 *
 * Allocates and frees k_rope_cache and mla_rope_freq for MLA models.
 *
 * The critical fix: standard RoPE precomputes freq[i] = 1/(theta^(2i/head_dim))
 * using head_dim = dim/n_heads = 128. But MLA applies RoPE only to the
 * qk_rope_head_dim = 64 slice. llama.cpp calls rope_ext with n_rot=64, which
 * computes freq[i] = 1/(theta^(2i/64)). Using head_dim=128 vs rope_dim=64
 * makes freq[31] differ by a factor of ~147 — completely wrong positional encoding.
 *
 * Fix: allocate mla_rope_freq with rope_precompute_freqs(buf, rope_dim, theta)
 * so freq[i] = 1/(theta^(2i/rope_dim)), giving exactly 32 correct entries.
 */

#include "core/run_state.h"
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset */

TernaryError mla_run_state_alloc(RunState *s, const Config *cfg,
                                  const MoEConfig *mc, int max_seq_len) {
    if (!mc || !mc->has_mla) return TN_OK;   /* dense or non-MLA model — no-op */

    int n_layers = cfg->n_layers;
    int rope_dim = mc->qk_rope_head_dim;

    if (n_layers <= 0 || rope_dim <= 0 || max_seq_len <= 0) return TN_ERR_INVALID_WEIGHTS;

    /* Allocate pointer array for k_rope_cache */
    float **cache = (float **)malloc((size_t)n_layers * sizeof(float *));
    if (!cache) return TN_ERR_OOM;

    for (int l = 0; l < n_layers; l++) {
        size_t elems = (size_t)max_seq_len * rope_dim;
        cache[l] = (float *)tn_aligned_calloc(elems, sizeof(float), 64);
        if (!cache[l]) {
            /* Free what we already allocated */
            for (int i = 0; i < l; i++) tn_aligned_free(cache[i]);
            free(cache);
            return TN_ERR_OOM;
        }
    }

    s->k_rope_cache = cache;

    /* Allocate and fill MLA RoPE frequency table.
     * Must use rope_dim (64) as the "head_dim" argument so that:
     *   mla_rope_freq[i] = 1/(rope_theta ^ (2i / rope_dim))  for i in [0, rope_dim/2)
     * This matches llama.cpp ggml_rope_ext with n_rot = rope_dim. */
    int freq_len = rope_dim / 2;
    s->mla_rope_freq = (float *)tn_aligned_calloc((size_t)freq_len, sizeof(float), 64);
    if (!s->mla_rope_freq) {
        mla_run_state_free(s, n_layers);
        return TN_ERR_OOM;
    }
    float theta = cfg->rope_theta > 0.0f ? cfg->rope_theta : 10000.0f;
    rope_precompute_freqs(s->mla_rope_freq, rope_dim, theta);

    return TN_OK;
}

void mla_run_state_free(RunState *s, int n_layers) {
    if (!s) return;
    if (s->k_rope_cache) {
        for (int l = 0; l < n_layers; l++) {
            tn_aligned_free(s->k_rope_cache[l]);
        }
        free(s->k_rope_cache);
        s->k_rope_cache = NULL;
    }
    if (s->mla_rope_freq) {
        tn_aligned_free(s->mla_rope_freq);
        s->mla_rope_freq = NULL;
    }
}
