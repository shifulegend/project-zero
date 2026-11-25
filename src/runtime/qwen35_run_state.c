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

#include <stdlib.h>
#include <string.h>

#include "core/platform.h" /* tn_i64 */
#include "math/rope.h"
#include "memory/aligned_alloc.h"
#include "runtime/run_state.h"

/* Not a usable hybrid shape (or no context): refuse before allocating anything. */
static int q35_shape_invalid(const Config *cfg, const MoEConfig *mc, int max_seq_len) {
    int conv_dim = mc->ssm_group_count * mc->ssm_state_size * 2 + mc->ssm_inner_size;
    return cfg->n_layers <= 0 || cfg->n_kv_heads <= 0 || mc->attn_head_dim <= 0 ||
           mc->ssm_conv_kernel <= 1 || conv_dim <= 0 || mc->ssm_time_step_rank <= 0 ||
           mc->ssm_state_size <= 0 || max_seq_len <= 0;
}

/* Context length this arch's own K/V cache may use under the fixed RAM ceiling. */
static int q35_clamp_seq(const Config *cfg, const MoEConfig *mc, int max_seq_len) {
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
    int n_full_layers = 0;
    for (int l = 0; l < cfg->n_layers; l++)
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
    tn_i64 per_pos_bytes            = (tn_i64)cfg->n_kv_heads * (tn_i64)mc->attn_head_dim * 2 *
                           (tn_i64)sizeof(float) * (tn_i64)n_full_layers;
    if (per_pos_bytes > 0) {
        tn_i64 safe_max = FIXED_BUDGET_BYTES / per_pos_bytes;
        if (safe_max < 256) safe_max = 256; /* never below a minimal usable context */
        if (max_seq_len > (int)safe_max) max_seq_len = (int)safe_max;
    }
    return max_seq_len;
}

/* One layer's buffers: K/V cache for full attention, conv + recurrent state otherwise. */
static TernaryError q35_alloc_layer(RunState *s, const Config *cfg, const MoEConfig *mc, int l) {
    if (q35_layer_is_full_attn(mc, l)) {
        size_t kv_elems =
            (size_t)cfg->n_kv_heads * (size_t)s->max_seq_len * (size_t)mc->attn_head_dim;
        s->q35.key_cache[l]   = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
        s->q35.value_cache[l] = (float *)tn_aligned_calloc(kv_elems, sizeof(float), 64);
        return s->q35.key_cache[l] && s->q35.value_cache[l] ? TN_OK : TN_ERR_OOM;
    }
    size_t conv_dim =
        (size_t)mc->ssm_group_count * (size_t)mc->ssm_state_size * 2 + (size_t)mc->ssm_inner_size;
    size_t conv_elems = (size_t)(mc->ssm_conv_kernel - 1) * conv_dim;
    size_t recur_elems =
        (size_t)mc->ssm_time_step_rank * (size_t)mc->ssm_state_size * (size_t)mc->ssm_state_size;
    s->q35.conv_state[l]  = (float *)tn_aligned_calloc(conv_elems, sizeof(float), 64);
    s->q35.recur_state[l] = (float *)tn_aligned_calloc(recur_elems, sizeof(float), 64);
    return s->q35.conv_state[l] && s->q35.recur_state[l] ? TN_OK : TN_ERR_OOM;
}

/* The four per-layer pointer arrays (all NULL-filled), then every layer's buffers. */
static TernaryError q35_alloc_layers(RunState *s, const Config *cfg, const MoEConfig *mc) {
    size_t n   = (size_t)cfg->n_layers;
    float **kc = (float **)calloc(n, sizeof(float *));
    float **vc = (float **)calloc(n, sizeof(float *));
    float **cs = (float **)calloc(n, sizeof(float *));
    float **rs = (float **)calloc(n, sizeof(float *));
    if (!kc || !vc || !cs || !rs) {
        free(kc);
        free(vc);
        free(cs);
        free(rs);
        return TN_ERR_OOM;
    }
    s->q35.key_cache   = kc;
    s->q35.value_cache = vc;
    s->q35.conv_state  = cs;
    s->q35.recur_state = rs;
    for (int l = 0; l < cfg->n_layers; l++)
        if (q35_alloc_layer(s, cfg, mc, l) != TN_OK) return TN_ERR_OOM;
    return TN_OK;
}

/* RoPE table and attention scratch. */
static TernaryError q35_alloc_tables(RunState *s, const Config *cfg, const MoEConfig *mc) {
    int head_dim = mc->attn_head_dim;
    int rope_dim = mc->attn_rope_dim > 0 ? mc->attn_rope_dim : head_dim;
    /* Shared partial-rotary RoPE frequency table, sized for rope_dim (64),
     * NOT head_dim (256) — mirrors mla_rope_freq's fix for the same class
     * of bug (see mla_run_state.c header comment). */
    s->q35.rope_freq = (float *)tn_aligned_calloc((size_t)(rope_dim / 2), sizeof(float), 64);
    if (!s->q35.rope_freq) return TN_ERR_OOM;
    float theta = cfg->rope_theta > 0.0f ? cfg->rope_theta : 10000.0f;
    rope_precompute_freqs(s->q35.rope_freq, rope_dim, theta);

    /* Attention scratch (see RunState.q35.scratch): both layer kinds' buffers, laid out by
     * qwen35_attention.c from the same widths. */
    size_t conv_dim =
        (size_t)mc->ssm_group_count * (size_t)mc->ssm_state_size * 2 + (size_t)mc->ssm_inner_size;
    size_t q_w  = (size_t)cfg->n_heads * (size_t)head_dim;
    size_t kv_w = (size_t)cfg->n_kv_heads * (size_t)head_dim;
    size_t full = 2 * q_w + 2 * kv_w + q_w;
    size_t lin = conv_dim * 2 + (size_t)mc->ssm_inner_size * 2 + (size_t)mc->ssm_time_step_rank * 3;
    s->q35.scratch = (float *)tn_aligned_calloc(full + lin, sizeof(float), 64);
    return s->q35.scratch ? TN_OK : TN_ERR_OOM;
}

TernaryError q35_run_state_alloc(RunState *s, const Config *cfg, const MoEConfig *mc,
                                 int max_seq_len) {
    if (!mc || !mc->has_linear_attn) return TN_OK; /* not a hybrid model — no-op */

    /* s->key_cache/value_cache should already be NULL here: main.c calls
     * run_state_alloc_ex(..., skip_kv_cache=mc->has_linear_attn) precisely
     * so the *generic* key_cache/value_cache (sized with
     * config_head_dim(cfg)==dim/n_heads for ALL cfg->n_layers — wrong
     * head_dim for this arch, and never read by qwen35_attention.c, which
     * uses q35.key_cache/q35.value_cache below instead) is never allocated
     * at all for a hybrid model. This free is a defensive no-op if that
     * invariant is ever violated (e.g. a caller uses plain run_state_alloc()
     * instead of the _ex variant) — see docs/ai/mistakes.md for why a
     * calloc-then-immediately-free of that buffer is NOT equivalent to
     * never allocating it (tn_aligned_calloc's memset forces the RSS spike
     * at allocation time, not at free time). */
    tn_aligned_free(s->key_cache);
    s->key_cache = NULL;
    tn_aligned_free(s->value_cache);
    s->value_cache = NULL;

    if (q35_shape_invalid(cfg, mc, max_seq_len)) return TN_ERR_INVALID_WEIGHTS;
    max_seq_len = q35_clamp_seq(cfg, mc, max_seq_len);

    /* s->max_seq_len was already set (to the original, unclamped seq_len)
     * by the earlier run_state_alloc_ex() call in main.c. qwen35_attention.c
     * computes q35.key_cache/q35.value_cache byte offsets as
     * (kh * s->max_seq_len + pos) * head_dim — i.e. it trusts
     * s->max_seq_len as the per-head stride of the buffers allocated
     * below. If this function clamps its own *local* max_seq_len (for the
     * 600 MB budget above) without also updating s->max_seq_len, every
     * consumer keeps using the old, larger stride against the new,
     * smaller allocation: an out-of-bounds write for any kh>0, reliably
     * segfaulting in qwen35_attention_forward's very first full-attention
     * layer. See docs/ai/mistakes.md. */
    s->max_seq_len = max_seq_len;

    TernaryError rc = q35_alloc_layers(s, cfg, mc);
    if (rc == TN_OK) rc = q35_alloc_tables(s, cfg, mc);
    if (rc == TN_ERR_OOM && s->q35.key_cache) q35_run_state_free(s, cfg, mc);
    return rc;
}

void q35_run_state_free(RunState *s, const Config *cfg, const MoEConfig *mc) {
    if (!s) return;
    int n_layers = cfg ? cfg->n_layers : 0;
    if (s->q35.key_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35.key_cache[l]);
        free(s->q35.key_cache);
        s->q35.key_cache = NULL;
    }
    if (s->q35.value_cache) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35.value_cache[l]);
        free(s->q35.value_cache);
        s->q35.value_cache = NULL;
    }
    if (s->q35.conv_state) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35.conv_state[l]);
        free(s->q35.conv_state);
        s->q35.conv_state = NULL;
    }
    if (s->q35.recur_state) {
        for (int l = 0; l < n_layers; l++) tn_aligned_free(s->q35.recur_state[l]);
        free(s->q35.recur_state);
        s->q35.recur_state = NULL;
    }
    if (s->q35.rope_freq) {
        tn_aligned_free(s->q35.rope_freq);
        s->q35.rope_freq = NULL;
    }
    if (s->q35.scratch) {
        tn_aligned_free(s->q35.scratch);
        s->q35.scratch = NULL;
    }
    (void)mc;
}
