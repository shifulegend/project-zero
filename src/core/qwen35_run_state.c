/**
 * qwen35_run_state.c
 *
 * Allocates per-layer state for Qwen3.5/3.6 hybrid Gated-DeltaNet +
 * Gated-Attention models:
 *   - full-attention layers get their own [n_kv_heads][max_seq][attn_head_dim]
 *     K/V cache, correctly sized with mc->attn_head_dim (dim/n_heads is not
 *     an integer for this arch — reusing RunState's generic key_cache/
 *     value_cache/rope_freq, which assume config_head_dim()==dim/n_heads,
 *     would silently corrupt every full-attention layer; same class of bug
 *     mla_run_state.c already documents and fixes for MLA's rope_dim).
 *   - linear-attention layers get a small fixed-size causal-conv history
 *     buffer and a persistent delta-rule recurrent state matrix per head —
 *     no growing KV cache at all, since Gated DeltaNet's whole point is
 *     O(1)-per-token state instead of an O(context) cache.
 */

#include "core/run_state.h"
#include "core/platform.h"   /* tn_i64 */
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>

TernaryError q35_run_state_alloc(RunState *s, const Config *cfg,
                                  const MoEConfig *mc, int max_seq_len) {
    if (!mc || !mc->has_linear_attn) return TN_OK; /* not a hybrid model — no-op */

    /* s->key_cache/value_cache should already be NULL here: main.c calls
     * run_state_alloc_ex(..., skip_kv_cache=mc->has_linear_attn) precisely
     * so the *generic* key_cache/value_cache (sized with
     * config_head_dim(cfg)==dim/n_heads for ALL cfg->n_layers — wrong
     * head_dim for this arch, and never read by qwen35_attention.c, which
     * uses q35_key_cache/q35_value_cache below instead) is never allocated
     * at all for a hybrid model. This free is a defensive no-op if that
     * invariant is ever violated (e.g. a caller uses plain run_state_alloc()
     * instead of the _ex variant) — see docs/ai/mistakes.md for why a
     * calloc-then-immediately-free of that buffer is NOT equivalent to
     * never allocating it (tn_aligned_calloc's memset forces the RSS spike
     * at allocation time, not at free time). */
    tn_aligned_free(s->key_cache);   s->key_cache   = NULL;
    tn_aligned_free(s->value_cache); s->value_cache = NULL;

    int n_layers  = cfg->n_layers;
    int n_kv_h    = cfg->n_kv_heads;
    int head_dim  = mc->attn_head_dim;
    int rope_dim  = mc->attn_rope_dim > 0 ? mc->attn_rope_dim : head_dim;
    int conv_k    = mc->ssm_conv_kernel;
    int key_dim   = mc->ssm_group_count * mc->ssm_state_size;
    int value_dim = mc->ssm_inner_size;
    int conv_dim  = key_dim * 2 + value_dim;
    int n_v_heads = mc->ssm_time_step_rank;
    int state_sz  = mc->ssm_state_size;

    if (n_layers <= 0 || n_kv_h <= 0 || head_dim <= 0 || conv_k <= 1 ||
        conv_dim <= 0 || n_v_heads <= 0 || state_sz <= 0 || max_seq_len <= 0)
        return TN_ERR_INVALID_WEIGHTS;

    /* Defense-in-depth RAM safety cap for this arch's OWN K/V cache.
     *
     * select_kv_strategy()'s existing budget check (called earlier in
     * main.c, before this function) sizes RunState's *generic*
     * key_cache/value_cache using config_head_dim(cfg) == dim/n_heads,
     * which is wrong for this arch (5120/24 truncates to 213, not the real
     * 256) — and that generic cache is skipped entirely for qwen35 now
     * (run_state_alloc_ex's skip_kv_cache), but its budget check still
     * doesn't know this file allocates its own, additive K/V cache.
     *
     * This does NOT use tn_get_free_ram() (MemAvailable) the way
     * mla_run_state.c's sibling code effectively relies on elsewhere:
     * MemAvailable counts this process's OWN clean, MAP_POPULATE'd model
     * weight pages (src/memory/mapped_file.c pre-faults the whole ~7 GB
     * file at open time) as reclaimable/"available" RAM, even though this
     * process needs them resident to run at all. Sizing a NEW allocation
     * off that inflated number is what caused a real near-OOM (~15.5 GB
     * RSS on a 15 GB host, well past a 14.7 GB kill threshold) during
     * verification — see docs/ai/mistakes.md. A fixed, conservative
     * absolute ceiling sidesteps that self-referential accounting problem
     * entirely instead of trying to out-guess it. */
    {
        int n_full_layers = 0;
        for (int l = 0; l < n_layers; l++)
            if (q35_layer_is_full_attn(mc, l)) n_full_layers++;

        /* Fixed ceiling: 600 MB total for this cache. At head_dim=256,
         * n_kv_heads=4, 16 full-attention layers (Ternary-Bonsai-27B's
         * actual shape), that's ~600e6 / (4*256*2*4*16) = ~4577 tokens of
         * context — plenty for correctness verification and short
         * benchmark prompts, while leaving generous headroom on a 15 GB
         * host once the ~7 GB of MAP_POPULATE'd weights (always resident,
         * see comment above) and tokenizer/other overhead are accounted
         * for. Raise this if a host with more free RAM needs longer
         * context — it's intentionally conservative, not tuned per-host. */
        const tn_i64 FIXED_BUDGET_BYTES = 600000000LL;
        tn_i64 per_pos_bytes = (tn_i64)n_kv_h * (tn_i64)head_dim * 2 * (tn_i64)sizeof(float)
                             * (tn_i64)n_full_layers;
        if (per_pos_bytes > 0) {
            tn_i64 safe_max = FIXED_BUDGET_BYTES / per_pos_bytes;
            if (safe_max < 256) safe_max = 256; /* never below a minimal usable context */
            if (max_seq_len > (int)safe_max) max_seq_len = (int)safe_max;
        }
    }

    /* s->max_seq_len was already set (to the original, unclamped seq_len)
     * by the earlier run_state_alloc_ex() call in main.c. qwen35_attention.c
     * computes q35_key_cache/q35_value_cache byte offsets as
     * (kh * s->max_seq_len + pos) * head_dim — i.e. it trusts
     * s->max_seq_len as the per-head stride of the buffers allocated
     * below. If this function clamps its own *local* max_seq_len (for the
     * 600 MB budget above) without also updating s->max_seq_len, every
     * consumer keeps using the old, larger stride against the new,
     * smaller allocation: an out-of-bounds write for any kh>0, reliably
     * segfaulting in qwen35_attention_forward's very first full-attention
     * layer. See docs/ai/mistakes.md. */
    s->max_seq_len = max_seq_len;

    float **kc = (float **)calloc((size_t)n_layers, sizeof(float *));
    float **vc = (float **)calloc((size_t)n_layers, sizeof(float *));
    float **cs = (float **)calloc((size_t)n_layers, sizeof(float *));
    float **rs = (float **)calloc((size_t)n_layers, sizeof(float *));
    if (!kc || !vc || !cs || !rs) {
        free(kc); free(vc); free(cs); free(rs);
        return TN_ERR_OOM;
    }
    s->q35_key_cache   = kc;
    s->q35_value_cache = vc;
    s->q35_conv_state  = cs;
    s->q35_recur_state = rs;

    for (int l = 0; l < n_layers; l++) {
        if (q35_layer_is_full_attn(mc, l)) {
            size_t kv_elems = (size_t)n_kv_h * (size_t)max_seq_len * (size_t)head_dim;
            kc[l] = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
            vc[l] = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
            if (!kc[l] || !vc[l]) { q35_run_state_free(s, cfg, mc); return TN_ERR_OOM; }
        } else {
            size_t conv_elems  = (size_t)(conv_k - 1) * (size_t)conv_dim;
            size_t recur_elems = (size_t)n_v_heads * (size_t)state_sz * (size_t)state_sz;
            cs[l] = (float *)tn_aligned_calloc(conv_elems, sizeof(float), 64);
            rs[l] = (float *)tn_aligned_calloc(recur_elems, sizeof(float), 64);
            if (!cs[l] || !rs[l]) { q35_run_state_free(s, cfg, mc); return TN_ERR_OOM; }
        }
    }

    /* Shared partial-rotary RoPE frequency table, sized for rope_dim (64),
     * NOT head_dim (256) — mirrors mla_rope_freq's fix for the same class
     * of bug (see mla_run_state.c header comment). */
    int freq_len = rope_dim / 2;
    s->q35_rope_freq = (float *)tn_aligned_calloc((size_t)freq_len, sizeof(float), 64);
    if (!s->q35_rope_freq) { q35_run_state_free(s, cfg, mc); return TN_ERR_OOM; }
    float theta = cfg->rope_theta > 0.0f ? cfg->rope_theta : 10000.0f;
    rope_precompute_freqs(s->q35_rope_freq, rope_dim, theta);

    return TN_OK;
}

void q35_run_state_free(RunState *s, const Config *cfg, const MoEConfig *mc) {
    if (!s) return;
    int n_layers = cfg ? cfg->n_layers : 0;
    if (s->q35_key_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35_key_cache[l]);
        free(s->q35_key_cache); s->q35_key_cache = NULL;
    }
    if (s->q35_value_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35_value_cache[l]);
        free(s->q35_value_cache); s->q35_value_cache = NULL;
    }
    if (s->q35_conv_state) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35_conv_state[l]);
        free(s->q35_conv_state); s->q35_conv_state = NULL;
    }
    if (s->q35_recur_state) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35_recur_state[l]);
        free(s->q35_recur_state); s->q35_recur_state = NULL;
    }
    if (s->q35_rope_freq) {
        tn_aligned_free(s->q35_rope_freq);
        s->q35_rope_freq = NULL;
    }
    (void)mc;
}
