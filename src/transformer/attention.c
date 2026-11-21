#include "transformer/attention.h"

#include <math.h>
#include <string.h>

#include "core/clamp.h"
#include "core/step_timing.h"
#include "core/weights.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/lora_matmul.h"
#include "math/rope.h"
#include "math/simd_core_ops.h"
#include "math/simd_vec_ops.h"
#include "runtime/spec_scratch.h"
#include "transformer/attn_tiled.h"
#include "transformer/dense_matmul_dispatch.h"
#include "transformer/mla_attention.h"
#include "transformer/qwen35_attention.h"
#include "transformer/qwen3moe_attention.h"
#if TN_HAS_AVX512
#include <immintrin.h>
#endif

/*
 * K-3 R-3: Non-temporal store helper for KV cache writes.
 *
 * KV cache arrays are allocated with TN_SIMD_ALIGN=64 byte alignment.
 * head_dim is always a multiple of 16 floats (64 bytes) for this model
 * (head_dim=128 → 512 bytes).  Both conditions make _mm512_stream_ps safe.
 *
 * Non-temporal stores bypass the L3 cache write-back buffer, preventing
 * KV cache data from evicting weight rows that are needed for the next
 * layer.  This is critical when the model (1.1 GB) dwarfs L3 (12 MB).
 * An _mm_sfence() after the loop ensures the stores reach DRAM ordering.
 */
/* Quantized KV rows (plan 8.5, i8/i4): row r = KV_SCALE_IDX(...) -> its data
 * bytes and its F32 scales (see RunState.kv_row_bytes/kv_row_scales). */
static inline tn_i8 *kvq_k(const RunState *s, size_t r) {
    return s->kvs.key_cache_q + r * (size_t)s->kvs.row_bytes;
}
static inline tn_i8 *kvq_v(const RunState *s, size_t r) {
    return s->kvs.value_cache_q + r * (size_t)s->kvs.row_bytes;
}
static inline float *kvq_ks(const RunState *s, size_t r) {
    return s->kvs.key_scale + r * (size_t)s->kvs.row_scales;
}
static inline float *kvq_vs(const RunState *s, size_t r) {
    return s->kvs.value_scale + r * (size_t)s->kvs.row_scales;
}

/* y += b (n floats); no-op when the model has no bias for this projection. */

static inline void attn_add_bias(float *y, const float *b, int n) {
    if (!b) return;
    for (int i = 0; i < n; i++) y[i] += b[i];
}

static void kv_nt_store(float *dst, const float *src, int n) {
#if TN_HAS_AVX512
    int i = 0;
    for (; i + 15 < n; i += 16) _mm512_stream_ps(dst + i, _mm512_loadu_ps(src + i));
    _mm_sfence();
    for (; i < n; i++) dst[i] = src[i];
#else
    memcpy(dst, src, (size_t)n * sizeof(float));
#endif
}

/* ── Per-head attention (Step 5 of attention_forward) ───────────────────── */

/* One query head against one sequence's KV cache: scores Q . K[t] / sqrt(head_dim) over the valid
 * window, softmax, then the weighted sum of V into out_head. The scan the single-token path
 * (Step 5) and the batched path (per row) both run; positions are mapped through the sliding
 * window, so a wrapped cache reads the right slots (PRE10-BUG-004). */
typedef struct {
    const RunState *s;
    int layer, kv_h, n_kv_heads, head_dim;
    float inv_sqrt_head_dim;
} HeadScan;

static void attn_scan_head(const HeadScan *hs, const float *q_head, float *out_head,
                           float *att_scores, int pos) {
    const RunState *s  = hs->s;
    const int head_dim = hs->head_dim, layer = hs->layer, kv_h = hs->kv_h;
    const int n_kv_heads = hs->n_kv_heads, max_seq = s->max_seq_len;
    const int quant     = s->kvs.type != KV_TYPE_F32;
    const int valid_ctx = sw_valid_count(&s->sw, pos);

    for (int t = 0; t < valid_ctx; t++) {
        /* Reconstruct the historical logical position: after a wrap the oldest valid token is at
         * pos - valid_ctx + 1 + t, otherwise it is t. */
        int hist_logical = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t     = sw_map_position(&s->sw, hist_logical);
        float score;
        if (quant) {
            /* Plan 8.5: quantized row . F32 query, scales folded in. */
            size_t r = KV_SCALE_IDX(layer, kv_h, mapped_t, n_kv_heads, max_seq);
            score =
                kvq_dot_row((KVCacheType)s->kvs.type, q_head, kvq_k(s, r), kvq_ks(s, r), head_dim);
        } else {
            size_t k_offset = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, n_kv_heads, max_seq, head_dim);
            score           = tn_vec_dot(q_head, &s->key_cache[k_offset], head_dim);
        }
        att_scores[t] = score * hs->inv_sqrt_head_dim;
    }

    tn_softmax(att_scores, valid_ctx);

    memset(out_head, 0, head_dim * sizeof(float));
    for (int t = 0; t < valid_ctx; t++) {
        int hist_logical = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t;
        int mapped_t     = sw_map_position(&s->sw, hist_logical);
        float a_t        = att_scores[t];
        if (quant) {
            size_t r = KV_SCALE_IDX(layer, kv_h, mapped_t, n_kv_heads, max_seq);
            kvq_axpy_row((KVCacheType)s->kvs.type, out_head, a_t, kvq_v(s, r), kvq_vs(s, r),
                         head_dim);
        } else {
            size_t v_offset = KV_CACHE_IDX(layer, kv_h, mapped_t, 0, n_kv_heads, max_seq, head_dim);
            tn_vec_saxpy(out_head, a_t, &s->value_cache[v_offset], head_dim);
        }
    }
}


typedef struct {
    RunState *s;
    int layer, pos, head_dim, kv_mul, n_kv_heads, max_seq;
    float inv_sqrt_head_dim;
    int use_tiled, tile;
} AttnHeadArgs;

static void attn_one_head(const AttnHeadArgs *a, int h) {
    RunState *s        = a->s;
    const int head_dim = a->head_dim, max_seq = a->max_seq, pos = a->pos;
    const int layer = a->layer, n_kv_heads = a->n_kv_heads;
    const float inv_sqrt_head_dim = a->inv_sqrt_head_dim;

    /* Pointer to this head's query vector */
    float *q_head = s->q + h * head_dim;

    /* Resolve which KV head this query head maps to (GQA) */
    int kv_h = h / a->kv_mul;

    /* Number of valid context tokens to attend to */
    int valid_ctx = sw_valid_count(&s->sw, pos);

    /* Phase 27.3: single-pass tiled online-softmax kernel (opt-in; F32 KV only). */
    if (a->use_tiled && s->kvs.type == KV_TYPE_F32) {
        size_t slab = KV_CACHE_IDX(layer, kv_h, 0, 0, n_kv_heads, max_seq, head_dim);
        attn_head_tiled(s->xb + h * head_dim, q_head, &s->key_cache[slab], &s->value_cache[slab],
                        head_dim, &s->sw, pos, valid_ctx, inv_sqrt_head_dim, a->tile);
        return;
    }

    HeadScan hs = {s, layer, kv_h, n_kv_heads, head_dim, inv_sqrt_head_dim};
    attn_scan_head(&hs, q_head, s->xb + h * head_dim, s->att + h * max_seq, pos);
}

static void attn_heads_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    for (int h = start; h < end; h++) attn_one_head((const AttnHeadArgs *)arg, h);
}

/* Routes to the dedicated forward pass of the model's attention family, if it is not the generic
 * dense/GQA one handled below. The checks are a pointer deref and an int compare: the branch
 * predictor handles them in 0-1 cycles for BitNet, Llama and Mixtral. Returns 1 when handled.
 *   MLA (Phase 17.8); Qwen3.5/3.6 hybrid (Gated DeltaNet + Gated Attention); Qwen3-MoE (plain GQA +
 *   QK-norm, independent head_dim, its own function: see qwen3moe_attention.c's header). */
static int attn_route_family(RunState *s, const TransformerWeights *w, const Config *cfg,
                             const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    if (!mc) return 0;
    if (mc->mla.enabled) {
        mla_attention_forward(s, w, cfg, mc, layer, pos, tp);
        return 1;
    }
    if (mc->has_linear_attn) {
        qwen35_attention_forward(s, w, cfg, mc, layer, pos, tp);
        return 1;
    }
    if (mc->has_qk_norm) {
        qwen3moe_attention_forward(s, w, cfg, mc, layer, pos, tp);
        return 1;
    }
    return 0;
}

/* Per-head QK-norm of the Qwen3 dense family: RMSNorm on each Q and K head of one row, before
 * RoPE. */
static void attn_qk_norm(const TransformerWeights *w, const Config *cfg, int layer, float *q,
                         float *k) {
    int head_dim = config_head_dim(cfg);
    for (int h = 0; h < cfg->n_heads; h++) {
        float *q_h = q + (size_t)h * head_dim;
        tn_rmsnorm(q_h, q_h, w->q35.attn_q_norm[layer], head_dim, cfg->rms_norm_eps);
    }
    for (int kh = 0; kh < cfg->n_kv_heads; kh++) {
        float *k_h = k + (size_t)kh * head_dim;
        tn_rmsnorm(k_h, k_h, w->q35.attn_k_norm[layer], head_dim, cfg->rms_norm_eps);
    }
}

/* Phase 19: LoRA correction on top of the base Q/K/V projections (one call each, regardless of
 * which dispatch branch just ran; a no-op for every model that doesn't set s->active_lora, see
 * docs/ai/decision-log.md). Applied before QK-norm/RoPE, matching standard LoRA semantics: the
 * adapter modifies the linear projection's output and later normalization/rotation acts on the
 * combined (base + LoRA) result, as a real fine-tuned weight would produce. */
static void attn_lora_qkv(RunState *s, const Config *cfg, int layer, ThreadPool *tp, float *k_buf,
                          float *v_buf) {
    if (!s->active_lora) return;
    int dim    = cfg->dim;
    int kv_dim = config_kv_dim(cfg);
    lora_apply(s->q, s->xb, lora_mod(s->active_lora->q, layer), dim, dim, tp);
    lora_apply(k_buf, s->xb, lora_mod(s->active_lora->k, layer), dim, kv_dim, tp);
    lora_apply(v_buf, s->xb, lora_mod(s->active_lora->v, layer), dim, kv_dim, tp);
}

/* Steps 1-2b: pre-attention RMSNorm of s->x into s->xb, then Q (s->q), K and V projections with
 * the LoRA delta, the Qwen2/2.5 biases and, for Qwen3, per-head QK-norm. */
static void attn_project(RunState *s, const TransformerWeights *w, const Config *cfg, int layer,
                         ThreadPool *tp, float *k_buf, float *v_buf) {
    const WeightsAttn *wa = &w->attn;
    int dim               = cfg->dim;
    int kv_dim            = config_kv_dim(cfg);

    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, wa->norm[layer], dim, cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    /* Layer-level preq: quantise s->xb once, reuse for all three projections. Saves 2 redundant
     * quantisations per attention layer (3 calls -> 1 quantise). */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    attention_project_qkv(s, wa, w->layers_are_ternary, layer, dim, dim, kv_dim, s->q, k_buf, v_buf,
                          tp);

    attn_lora_qkv(s, cfg, layer, tp, k_buf, v_buf);

    /* Q/K/V biases (Qwen2/2.5), after any LoRA delta, before QK-norm/RoPE. */
    attn_add_bias(s->q, wa->bq ? wa->bq[layer] : NULL, dim);
    attn_add_bias(k_buf, wa->bk ? wa->bk[layer] : NULL, kv_dim);
    attn_add_bias(v_buf, wa->bv ? wa->bv[layer] : NULL, kv_dim);

    /* Absent (NULL) for architectures without QK-norm (Llama/Mistral/Gemma/Phi). */
    if (w->q35.attn_q_norm && w->q35.attn_q_norm[layer]) attn_qk_norm(w, cfg, layer, s->q, k_buf);
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
}

/* Step 3: RoPE on Q and K at `pos`, with full YaRN if configured. */
static void attn_rope(RunState *s, const Config *cfg, int pos, float *k_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    float corr[2];
    rope_corr_from_config(cfg, config_head_dim(cfg), corr);
    apply_rope_ex(s->q, k_buf, s->rope_freq, config_head_dim(cfg), pos, cfg->n_heads,
                  cfg->n_kv_heads, cfg->rope_freq_scale, cfg->rope_yarn_ext_factor,
                  cfg->rope_yarn_attn_factor, corr, cfg->rope_layout);
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);
}

/* Stores one row's K and V (n_kv_heads heads) into the transposed KV cache at
 * [layer][kv_head][mapped_pos][:], using non-temporal stores for an F32 cache (K-3 R-3: they keep
 * KV writes from evicting weight data from L3) or the quantised writer otherwise. */
static void store_kv_row(RunState *s, const Config *cfg, int layer, int mapped_pos,
                         const float *k_buf, const float *v_buf) {
    const RunKvStore *kvs = &s->kvs;
    int head_dim          = config_head_dim(cfg);
    int n_kv_heads        = cfg->n_kv_heads;
    int max_seq           = s->max_seq_len;
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        size_t cache_offset =
            KV_CACHE_IDX(layer, kv_h, mapped_pos, 0, n_kv_heads, max_seq, head_dim);
        if (kvs->type != KV_TYPE_F32) {
            size_t row = KV_SCALE_IDX(layer, kv_h, mapped_pos, n_kv_heads, max_seq);
            kvq_store((KVCacheType)kvs->type, kvq_k(s, row), kvq_ks(s, row),
                      &k_buf[kv_h * head_dim], head_dim);
            kvq_store((KVCacheType)kvs->type, kvq_v(s, row), kvq_vs(s, row),
                      &v_buf[kv_h * head_dim], head_dim);
        } else {
            kv_nt_store(&s->key_cache[cache_offset], &k_buf[kv_h * head_dim], head_dim);
            kv_nt_store(&s->value_cache[cache_offset], &v_buf[kv_h * head_dim], head_dim);
        }
    }
}

/* Step 4: store K and V at the current position, then advance the sliding window's write head. */
static void attn_store_kv(RunState *s, const Config *cfg, int layer, int pos, const float *k_buf,
                          const float *v_buf) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    store_kv_row(s, cfg, layer, sw_map_position(&s->sw, pos), k_buf, v_buf);
    sw_advance(&s->sw);
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Step 5: attention for every query head. Heads are independent (own score row in s->att, own
 * output slice of s->xb), so they are spread over the pool once the context is long enough for the
 * dispatch to pay off (--attn-parallel-min-ctx; Phase 27). */
static void attn_heads(RunState *s, const Config *cfg, int layer, int pos, ThreadPool *tp) {
    int head_dim    = config_head_dim(cfg);
    int n_heads     = cfg->n_heads;
    int64_t t_step  = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    AttnHeadArgs ha = {s,
                       layer,
                       pos,
                       head_dim,
                       n_heads / cfg->n_kv_heads,
                       cfg->n_kv_heads,
                       s->max_seq_len,
                       1.0f / sqrtf((float)head_dim),
                       s->attn.kernel == ATTN_KERNEL_TILED,
                       s->attn.tile};
    int valid_ctx   = sw_valid_count(&s->sw, pos);
    int min_ctx     = s->attn.parallel_min_ctx;
    if (tp && n_heads > 1 && min_ctx > 0 && valid_ctx >= min_ctx)
        threadpool_dispatch(tp, attn_heads_task, &ha, n_heads);
    else
        attn_heads_task(&ha, 0, 0, n_heads);
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);
}

/* Steps 6-8: attn_sub_norm (BitNet) if present, the output projection (s->xb -> s->xb2, with the
 * LoRA delta, Phase 19) and the residual add into s->x. */
static void attn_output(RunState *s, const TransformerWeights *w, const Config *cfg, int layer,
                        ThreadPool *tp) {
    const WeightsAttn *wa = &w->attn;
    int dim               = cfg->dim;
    int64_t t_step        = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (wa->sub_norm && wa->sub_norm[layer])
        tn_rmsnorm(s->xb, s->xb, wa->sub_norm[layer], dim, cfg->rms_norm_eps);

    if (w->layers_are_ternary)
        parallel_ternary_matmul_packed(s->xb2, s->xb, (const tn_u8 *)wa->wo[layer], dim, dim,
                                       wa->so[layer], tp);
    else
        tn_dense_matmul_dispatch(s->xb2, s->xb, wa->wo[layer], wa->wo_type[layer], dim, dim, tp);
    if (s->active_lora) lora_apply(s->xb2, s->xb, lora_mod(s->active_lora->o, layer), dim, dim, tp);

    tn_vec_add(s->x, s->x, s->xb2, dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

void attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                       const MoEConfig *mc, int layer, int pos, ThreadPool *tp) {
    if (attn_route_family(s, w, cfg, mc, layer, pos, tp)) return;
    /* k_buf: kv_dim floats, reusing xb2 (dim floats, kv_dim <= dim); v_buf: kv_dim floats,
     * reusing hb2 (hidden_dim floats >= kv_dim). */
    float *k_buf = s->xb2;
    float *v_buf = s->hb2;
    attn_project(s, w, cfg, layer, tp, k_buf, v_buf);
    attn_rope(s, cfg, pos, k_buf);
    attn_store_kv(s, cfg, layer, pos, k_buf, v_buf);
    attn_heads(s, cfg, layer, pos, tp);
    attn_output(s, w, cfg, layer, tp);
}

/* ── Phase 18 (speculative decoding): batched multi-token attention ────────
 *
 * Restructures the generic dense/GQA path above into two phases, mirroring
 * the single-token version step-for-step but batched:
 *   WRITE phase (per token k, independent across k): rmsnorm, Q/K/V
 *     projection (one batched matmul call across all n_tokens, not a
 *     per-token loop -- this is what saves RAM bandwidth), optional QK-norm,
 *     RoPE at position pos+k, KV cache store at sw_map_position(pos+k).
 *   READ phase (per token k, NOT batched across k): causal attention scan
 *     over sw_valid_count(pos+k) keys. This step has no weight matrix to
 *     amortize (the KV cache is small and already resident), and each
 *     token's causal window has a different length, so a plain loop over k
 *     reusing s->att is correct and simple -- matches the design plan.
 *
 * Two phases can be fully separated (not interleaved token-by-token) because
 * sw_map_position()/sw_valid_count() are pure functions of the logical
 * position argument alone (see src/kv_cache/sliding_window.c -- they never
 * read sw->write_head/sw->wrapped, only sw_advance() mutates those and
 * nothing reads the mutation back). So the read phase for token k, run
 * after every token's write phase has completed, still sees exactly the
 * causal window it should: real prior history plus the earlier same-batch
 * positions 0..k-1 (already written to the KV cache by the write phase).
 *
 * Q/K/V/O projections: ternary weights get the real batched kernel
 * (tn_ternary_matmul_packed_batch); F16 weights get the real batched kernel
 * via tn_dense_matmul_dispatch_batch(); every other format (Q4_K, Q4_K_X8,
 * Q2_0, F32) falls back to n_tokens sequential single-token calls inside
 * that same dispatch helper -- correct, documented, not the full bandwidth
 * win (Phase 18-B follow-up). The layer-level preq (shared Q8K
 * quantisation across Q/K/V) optimisation used by the single-token path is
 * intentionally not replicated here -- a real perf refinement, not a
 * correctness concern, left for Phase 18-B alongside the Q4_K/Q2_0/F32
 * batched kernels themselves. */
/* One head of the batched read phase for every row (see the call site in
 * attention_forward_rows). */
typedef struct {
    RunState *const *states;
    const int *pos_arr;
    SpecBatchScratch *sb;
    int layer, n_tokens, dim, head_dim, kv_mul, n_kv_heads;
    float inv_sqrt_head_dim;
} RowsHeadArgs;

static void rows_one_head(const RowsHeadArgs *a, int h) {
    const int head_dim = a->head_dim;
    const int kv_h     = h / a->kv_mul;
    for (int k = 0; k < a->n_tokens; k++) {
        RunState *s         = a->states[k];
        const float *q_head = a->sb->q + (size_t)k * a->dim + h * head_dim;
        float *out_head     = a->sb->xb + (size_t)k * a->dim + h * head_dim;
        HeadScan hs         = {s, a->layer, kv_h, a->n_kv_heads, head_dim, a->inv_sqrt_head_dim};
        attn_scan_head(&hs, q_head, out_head, s->att + h * s->max_seq_len, a->pos_arr[k]);
    }
}

static void rows_heads_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    for (int h = start; h < end; h++) rows_one_head((const RowsHeadArgs *)arg, h);
}

/* Write phase, steps 1-2 over all rows: per-row RMSNorm (each token normalises against its own
 * mean square, so it cannot be one batched matmul), one batched Q/K/V projection and the LoRA
 * deltas (Phase 19; the adapter is shared by every sequence in a step: states[0]'s). Whole-batch
 * step-timing granularity (Stage 5.5): one bracket per architectural step spanning all n_tokens,
 * mirroring attention_forward()'s step IDs. */
static void rows_project(RunState *const *states, SpecBatchScratch *sb, const TransformerWeights *w,
                         const Config *cfg, int layer, int n_tokens, float *k_buf_all,
                         float *v_buf_all, ThreadPool *tp) {
    int dim        = cfg->dim;
    int kv_dim     = config_kv_dim(cfg);
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++)
        tn_rmsnorm(sb->xb + (size_t)k * dim, sb->x + (size_t)k * dim, w->attn.norm[layer], dim,
                   cfg->rms_norm_eps);
    if (t_step) tn_step_timing_add(TN_STEP_4_PRE_ATTN_RMSNORM, tn_step_timing_now_ns() - t_step);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    attention_project_qkv_rows(sb, w, cfg, layer, n_tokens, k_buf_all, v_buf_all, tp);
    const RunState *s0 = states[0];
    if (s0->active_lora) {
        lora_apply_batch(sb->q, sb->xb, lora_mod(s0->active_lora->q, layer), dim, dim, n_tokens,
                         tp);
        lora_apply_batch(k_buf_all, sb->xb, lora_mod(s0->active_lora->k, layer), dim, kv_dim,
                         n_tokens, tp);
        lora_apply_batch(v_buf_all, sb->xb, lora_mod(s0->active_lora->v, layer), dim, kv_dim,
                         n_tokens, tp);
    }
    if (t_step) tn_step_timing_add(TN_STEP_5_Q_PROJECTION, tn_step_timing_now_ns() - t_step);
}

/* Biases, per-head QK-norm and RoPE at each row's own position, bracketed together as step 9 (in
 * the single-token path QK-norm sits in the step-5 bracket with the projections; here the
 * projections already closed theirs, so QK-norm's small cost is grouped with RoPE). */
static void rows_bias_norm_rope(RunState *const *states, const int *pos_arr, SpecBatchScratch *sb,
                                const TransformerWeights *w, const Config *cfg, int layer,
                                int n_tokens, float *k_buf_all, float *v_buf_all) {
    int dim        = cfg->dim;
    int kv_dim     = config_kv_dim(cfg);
    int head_dim   = config_head_dim(cfg);
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        float *q_k    = sb->q + (size_t)k * dim;
        float *kbuf_k = k_buf_all + (size_t)k * kv_dim;
        attn_add_bias(q_k, w->attn.bq ? w->attn.bq[layer] : NULL, dim);
        attn_add_bias(kbuf_k, w->attn.bk ? w->attn.bk[layer] : NULL, kv_dim);
        attn_add_bias(v_buf_all + (size_t)k * kv_dim, w->attn.bv ? w->attn.bv[layer] : NULL,
                      kv_dim);
        if (w->q35.attn_q_norm && w->q35.attn_q_norm[layer])
            attn_qk_norm(w, cfg, layer, q_k, kbuf_k);

        float corr[2];
        rope_corr_from_config(cfg, head_dim, corr);
        apply_rope_ex(q_k, kbuf_k, states[k]->rope_freq, head_dim, pos_arr[k], cfg->n_heads,
                      cfg->n_kv_heads, cfg->rope_freq_scale, cfg->rope_yarn_ext_factor,
                      cfg->rope_yarn_attn_factor, corr, cfg->rope_layout);
    }
    if (t_step) tn_step_timing_add(TN_STEP_9_YARN_ROPE, tn_step_timing_now_ns() - t_step);
}

/* KV cache store, per row. The write phase is fully separated from the read phase because
 * sw_map_position()/sw_valid_count() are pure functions of the logical position; sw_advance stays
 * for parity with the single-token path (read-phase correctness does not depend on it). */
static void rows_store_kv(RunState *const *states, const int *pos_arr, const Config *cfg, int layer,
                          int n_tokens, const float *k_buf_all, const float *v_buf_all) {
    int kv_dim     = config_kv_dim(cfg);
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    for (int k = 0; k < n_tokens; k++) {
        RunState *s = states[k];
        store_kv_row(s, cfg, layer, sw_map_position(&s->sw, pos_arr[k]),
                     k_buf_all + (size_t)k * kv_dim, v_buf_all + (size_t)k * kv_dim);
        sw_advance(&s->sw);
    }
    if (t_step) tn_step_timing_add(TN_STEP_10_KV_CACHE_WRITE, tn_step_timing_now_ns() - t_step);
}

/* Read phase: heads are spread over the pool like the single-token path (2026-10-22, Phase 23). A
 * head owns its own s->att row and its slice of every row's output, and walks the rows in order, so
 * rows that share a state (a prefill chunk) never touch the same score row at once. */
static void rows_heads(RunState *const *states, const int *pos_arr, SpecBatchScratch *sb,
                       const Config *cfg, int layer, int n_tokens, ThreadPool *tp) {
    int head_dim    = config_head_dim(cfg);
    int n_heads     = cfg->n_heads;
    int64_t t_step  = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    RowsHeadArgs ra = {states,
                       pos_arr,
                       sb,
                       layer,
                       n_tokens,
                       cfg->dim,
                       head_dim,
                       n_heads / cfg->n_kv_heads,
                       cfg->n_kv_heads,
                       1.0f / sqrtf((float)head_dim)};
    int work        = 0;
    for (int k = 0; k < n_tokens; k++) work += sw_valid_count(&states[k]->sw, pos_arr[k]);
    int min_ctx = states[0]->attn.parallel_min_ctx;
    if (tp && n_heads > 1 && min_ctx > 0 && work >= min_ctx)
        threadpool_dispatch(tp, rows_heads_task, &ra, n_heads);
    else
        rows_heads_task(&ra, 0, 0, n_heads);
    if (t_step) tn_step_timing_add(TN_STEP_11_ATTN_SCORE, tn_step_timing_now_ns() - t_step);
}

/* attn_sub_norm (BitNet) per row, the output projection as one batched call, the LoRA delta and the
 * residual (elementwise, so one call over the n_tokens*dim span equals n_tokens per-row calls). */
static void rows_output(RunState *const *states, SpecBatchScratch *sb, const TransformerWeights *w,
                        const Config *cfg, int layer, int n_tokens, ThreadPool *tp) {
    int dim        = cfg->dim;
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->attn.sub_norm && w->attn.sub_norm[layer]) {
        for (int k = 0; k < n_tokens; k++) {
            float *xb_k = sb->xb + (size_t)k * dim;
            tn_rmsnorm(xb_k, xb_k, w->attn.sub_norm[layer], dim, cfg->rms_norm_eps);
        }
    }
    if (w->layers_are_ternary)
        tn_ternary_matmul_packed_batch(sb->xb2, sb->xb, (const tn_u8 *)w->attn.wo[layer], dim, dim,
                                       w->attn.so[layer], n_tokens, tp);
    else
        tn_dense_matmul_dispatch_batch(sb->xb2, sb->xb, w->attn.wo[layer], w->attn.wo_type[layer],
                                       dim, dim, n_tokens, tp);
    const RunState *s0 = states[0];
    if (s0->active_lora)
        lora_apply_batch(sb->xb2, sb->xb, lora_mod(s0->active_lora->o, layer), dim, dim, n_tokens,
                         tp);
    tn_vec_add(sb->x, sb->x, sb->xb2, n_tokens * dim);
    if (t_step) tn_step_timing_add(TN_STEP_12_POST_ATTN, tn_step_timing_now_ns() - t_step);
}

/* Generic dense/GQA batched attention over n_tokens rows whose K/V live in
 * states[k] at position pos_arr[k] (Phase 23, 2026-10-19). One sequence's
 * consecutive tokens (speculative verify) and one token from each of several
 * sequences (continuous batching) are the same computation: weight matmuls
 * batched over all rows, KV writes then reads per row in its own cache. All
 * states share one adapter (states[0]->active_lora). k_buf/v_buf reuse sb->xb2/sb->hb2 exactly as
 * the single-token path reuses s->xb2/s->hb2 (kv_dim <= dim <= width(xb2), kv_dim <= hidden_dim <=
 * width(hb2)). */
static TernaryError attention_forward_rows(RunState *const *states, const int *pos_arr,
                                           SpecBatchScratch *sb, const TransformerWeights *w,
                                           const Config *cfg, int layer, int n_tokens,
                                           ThreadPool *tp) {
    float *k_buf_all = sb->xb2;
    float *v_buf_all = sb->hb2;
    rows_project(states, sb, w, cfg, layer, n_tokens, k_buf_all, v_buf_all, tp);
    rows_bias_norm_rope(states, pos_arr, sb, w, cfg, layer, n_tokens, k_buf_all, v_buf_all);
    rows_store_kv(states, pos_arr, cfg, layer, n_tokens, k_buf_all, v_buf_all);
    rows_heads(states, pos_arr, sb, cfg, layer, n_tokens, tp);
    rows_output(states, sb, w, cfg, layer, n_tokens, tp);
    return TN_OK;
}

TernaryError attention_forward_batch(RunState *s, SpecBatchScratch *sb, const TransformerWeights *w,
                                     const Config *cfg, const MoEConfig *mc, int layer, int pos,
                                     int n_tokens, ThreadPool *tp) {
    if (mc && mc->mla.enabled) {
        return mla_attention_forward_batch(s, sb, w, cfg, mc, layer, pos, n_tokens, tp);
    }
    if (mc && mc->has_linear_attn) {
        /* Qwen3.5/3.6 hybrid's Gated-DeltaNet linear-attention layers hold a
     * genuinely recurrent state (q35.recur_state) that position k+1's
     * update must read only after position k's update has landed -- this
     * cannot be batched with a local restructuring like the write/read
     * split above. Refuse cleanly; speculative_generate() checks this once
     * at startup so callers should never actually reach this branch in
     * practice (see docs/architecture/IMPLEMENTATION_PLAN.md Phase 18). */
        return TN_ERR_UNSUPPORTED;
    }
    if (mc && mc->has_qk_norm) {
        return qwen3moe_attention_forward_batch(s, sb, w, cfg, mc, layer, pos, n_tokens, tp);
    }

    enum { ROWS_ON_STACK = 64 };
    RunState *st_stack[ROWS_ON_STACK] = {0};
    int pos_stack[ROWS_ON_STACK]      = {0};
    RunState **st                     = st_stack;
    int *ps                           = pos_stack;
    if (n_tokens > ROWS_ON_STACK) {
        st = (RunState **)malloc((size_t)n_tokens * sizeof(*st));
        ps = (int *)malloc((size_t)n_tokens * sizeof(*ps));
        if (!st || !ps) {
            free(st);
            free(ps);
            return TN_ERR_OOM;
        }
    }
    for (int k = 0; k < n_tokens; k++) {
        st[k] = s;
        ps[k] = pos + k;
    }
    TernaryError e = attention_forward_rows(st, ps, sb, w, cfg, layer, n_tokens, tp);
    if (st != st_stack) {
        free(st);
        free(ps);
    }
    return e;
}

TernaryError attention_forward_multi(RunState *const *states, const int *pos_arr,
                                     SpecBatchScratch *sb, const TransformerWeights *w,
                                     const Config *cfg, const MoEConfig *mc, int layer,
                                     int n_tokens, ThreadPool *tp) {
    if (!states || !pos_arr || n_tokens <= 0) return TN_ERR_INVALID_ARGS;
    if (mc && (mc->mla.enabled || mc->has_linear_attn || mc->has_qk_norm || mc->is_ssm))
        return TN_ERR_UNSUPPORTED; /* generic dense/GQA path only (stage 2) */
    return attention_forward_rows(states, pos_arr, sb, w, cfg, layer, n_tokens, tp);
}
