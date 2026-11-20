#include "transformer/moe_ffn.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "core/debug.h"
#include "core/env_flag.h"
#include "core/gguf_reader.h"
#include "core/log.h"
#include "core/step_timing.h"
#include "math/matmul_q4k.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "memory/tls_scratch.h"
#include "transformer/moe_experts.h"
#include "transformer/moe_ffn_internal.h"
#include "transformer/moe_router.h"

/* Per-thread scratch (memory/tls_scratch.h): s->xb pre-quantized to Q8K once per MoE layer (Q4K
 * gate+up, shared w1+w3). Every user runs on the calling thread. */
static TnTlsScratch s_xb_scratch = TN_TLS_SCRATCH_INIT;

/* P4: Compute raw byte size of a quantized weight tensor.
 * Used to pass the correct length to madvise(MADV_WILLNEED). */
static size_t quant_tensor_bytes(int quant_type, size_t n_elems) {
    switch (quant_type) {
        case 10:
            return (n_elems / 256) * 84; /* Q2_K: 84 bytes/block */
        case 11:
            return (n_elems / 256) * 110; /* Q3_K: 110 bytes/block */
        case 12:
            return (n_elems / 256) * 144; /* Q4_K: 144 bytes/block */
        case 13:
            return (n_elems / 256) * 176; /* Q5_K: 176 bytes/block */
        case 14:
            return (n_elems / 256) * 210; /* Q6_K: 210 bytes/block */
        case 6:
            return (n_elems / 32) * 22; /* Q5_0: 22 bytes/block */
        case 7:
            return (n_elems / 32) * 24; /* Q5_1: 24 bytes/block */
        case 8:
            return (n_elems / 32) * 34; /* Q8_0: 34 bytes/block */
        case 20:
            return (n_elems / 32) * 18; /* IQ4_NL: 18 bytes/block */
        default:
            return n_elems * 4; /* F32 fallback */
    }
}

/* P4: Issue MADV_WILLNEED for all selected expert weight pages right after
 * routing (before the FFN loop) so the OS can page-in experts 1..top_k-1
 * while we compute expert 0. This amortises mmap cold-page-fault latency. */
static void prefetch_expert_weights(const TransformerWeights *w, int layer, const int *selected,
                                    int top_k, size_t w13_bytes, size_t w2_bytes) {
    for (int i = 0; i < top_k; i++) {
        int e = selected[i];
        if (e < 0) continue;
        if (w->moe.w1 && w->moe.w1[layer] && w->moe.w1[layer][e])
            madvise((void *)w->moe.w1[layer][e], w13_bytes, MADV_WILLNEED);
        if (w->moe.w3 && w->moe.w3[layer] && w->moe.w3[layer][e])
            madvise((void *)w->moe.w3[layer][e], w13_bytes, MADV_WILLNEED);
        if (w->moe.w2 && w->moe.w2[layer] && w->moe.w2[layer][e])
            madvise((void *)w->moe.w2[layer][e], w2_bytes, MADV_WILLNEED);
    }
}


/* 2026-10-21: fused two-dispatch expert path (moe_experts.c). Builds one job
 * per selected routed expert (+ the shared expert) whose weight types have
 * integer row-dot kernels, and runs them all in two thread-pool dispatches.
 * Sets *routed_done / *shared_done for the parts it handled; anything it
 * cannot handle (ternary .bin experts, Q2_K/Q3_K/IQ*, F32-dequantized shared
 * tensors) stays on the legacy path below. PZ_MOE_FUSED=0 forces the legacy
 * path (A/B and bisecting aid). */
static int moe_fused_enabled(void) {
    static atomic_int cached = TN_ENV_FLAG_UNSET;
    return tn_env_on_unless_zero(&cached, "PZ_MOE_FUSED");
}

static void moe_fused_try(RunState *s, const TransformerWeights *w, const Config *cfg,
                          const MoEConfig *mc, int layer, const int *selected, const float *scores,
                          int top_k, ThreadPool *tp, int *routed_done, int *shared_done) {
    *routed_done = 0;
    *shared_done = 0;
    if (!moe_fused_enabled()) return;
    const int dim = cfg->dim;
    TnMoEExpertJob jobs[MOE_SCORE_BUF_SIZE + 1];
    int nj = 0, routed = 0, shared = 0;

    if (w->expert_quant.present) {
        int w13 = (w->expert_quant.w13_per_layer && w->expert_quant.w13_per_layer[layer])
                      ? w->expert_quant.w13_per_layer[layer]
                      : w->expert_quant.w13_type;
        int w2  = (w->expert_quant.w2_per_layer && w->expert_quant.w2_per_layer[layer])
                      ? w->expert_quant.w2_per_layer[layer]
                      : w->expert_quant.w2_type;
        TnMoEExpertJob probe = {(const uint8_t *)1,
                                (const uint8_t *)1,
                                (const uint8_t *)1,
                                w13,
                                w13,
                                w2,
                                mc->expert_hidden_dim,
                                0.0f};
        if (moe_experts_job_ok(&probe, dim)) {
            for (int i = 0; i < top_k; i++) {
                int e = selected[i];
                if (e < 0 || e >= mc->num_experts) continue;
                TnMoEExpertJob j = {(const uint8_t *)w->moe.w1[layer][e],
                                    (const uint8_t *)w->moe.w3[layer][e],
                                    (const uint8_t *)w->moe.w2[layer][e],
                                    w13,
                                    w13,
                                    w2,
                                    mc->expert_hidden_dim,
                                    scores[i]};
                jobs[nj++]       = j;
            }
            routed = 1;
        }
    }
    if (mc->n_shared_experts > 0 && w->shared.w1 && w->shared.w1[layer] &&
        w->shared.w1_type_per_layer && w->shared.w2_type_per_layer && w->shared.w3_type_per_layer) {
        TnMoEExpertJob j = {
            (const uint8_t *)w->shared.w1[layer], (const uint8_t *)w->shared.w3[layer],
            (const uint8_t *)w->shared.w2[layer], w->shared.w1_type_per_layer[layer],
            w->shared.w3_type_per_layer[layer],   w->shared.w2_type_per_layer[layer],
            mc->shared_expert_hidden_dim,         1.0f};
        if (moe_experts_job_ok(&j, dim)) {
            jobs[nj++] = j;
            shared     = 1;
        }
    }
    if (nj == 0) return;
    if (moe_experts_forward(s->xb2, s->xb, jobs, nj, dim, cfg->act_type, tp) != 0) return;
    if (routed) {
        for (int i = 0; i < top_k; i++) {
            int e = selected[i];
            if (e >= 0 && e < mc->num_experts) moe_track_hit(layer, e);
        }
    }
    *routed_done = routed;
    *shared_done = shared;
}

/* P4: the raw-byte sizes of one expert's gate/up and down tensors, for madvise(). */
static void prefetch_selected(const TransformerWeights *w, int layer, int dim, int expert_hdim,
                              const int *selected, int top_k) {
    if (!w->expert_quant.present) return;
    int w2_qtype     = (w->expert_quant.w2_per_layer && w->expert_quant.w2_per_layer[layer])
                           ? w->expert_quant.w2_per_layer[layer]
                           : w->expert_quant.w2_type;
    size_t w13_bytes = quant_tensor_bytes(w->expert_quant.w13_type, (size_t)dim * expert_hdim);
    size_t w2_bytes  = quant_tensor_bytes(w2_qtype, (size_t)expert_hdim * dim);
    prefetch_expert_weights(w, layer, selected, top_k, w13_bytes, w2_bytes);
}

/* Steps 1-2: RMSNorm s->x into s->xb, then the router selects the top_k experts (gate weight
 * w->moe_gate_w[layer], F32 [num_experts x dim]). */
static void moe_norm_and_route(RunState *s, const TransformerWeights *w, const Config *cfg,
                               const MoEConfig *mc, int layer, ThreadPool *tp, float *scores,
                               int *selected, float *selected_scores) {
    const int dim         = cfg->dim;
    const int num_experts = mc->num_experts;
    const int top_k       = mc->num_experts_per_tok;

    /* Identical to the dense path; normalise s->x into s->xb. */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->ffn.norm[layer], dim, cfg->rms_norm_eps);
    if (t_step) {
        tn_step_timing_add(TN_STEP_13_PRE_FFN_RMSNORM, tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_norm", s->xb, dim);

    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    /* Dump raw router logits (pre-softmax) — only when tensor dump is active */
    if (g_dump_fp) {
        float raw_logits[MOE_SCORE_BUF_SIZE];
        parallel_matmul_float32(raw_logits, s->xb, (const float *)w->moe.gate_w[layer], dim,
                                num_experts, tp);
        DBG_DUMP(layer, "router_logits", raw_logits, num_experts);
    }

    /* Router computes gate matmul, gating, top-k (+ norm/scale) in one call */
    const MoERouterOpts ropts = {mc->expert_gating_func, mc->expert_weights_norm,
                                 mc->expert_weights_scale};
    moe_router_forward_opts(scores, selected, selected_scores, s->xb,
                            (const float *)w->moe.gate_w[layer], dim, num_experts, top_k, &ropts,
                            tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_15_MOE_ROUTING, tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_moe_probs", scores, num_experts);

    if (g_tn_verbose) {
        float xb_max = 0.0f;
        for (int ii = 0; ii < dim; ii++) {
            float a = s->xb[ii] < 0 ? -s->xb[ii] : s->xb[ii];
            if (a > xb_max) xb_max = a;
        }
        char sel[MOE_SCORE_BUF_SIZE * 16 + 1];
        size_t used = 0;
        sel[0]      = '\0';
        for (int ii = 0; ii < top_k && used + 16 < sizeof(sel); ii++)
            used += (size_t)snprintf(sel + used, sizeof(sel) - used, "%d(%.3f)%s", selected[ii],
                                     selected_scores[ii], ii < top_k - 1 ? ", " : "");
        TN_LOG_DBG("moe_ffn", "[DBG] L%02d router: xb_max_abs=%.4f  selected=[%s]\n", layer, xb_max,
                   sel);
    }
}

/* P1: s->xb quantised to Q8K once, reused by every Q4K dispatch (batched gate+up for routed
 * experts, shared expert w1+w3). NULL when nothing will use it. */
static TnQ8KActBlock *moe_quantize_xb(RunState *s, const TransformerWeights *w, int layer, int dim,
                                      int routed_done, int shared_done) {
    int xb_n_blocks = dim / TN_Q8K_BLOCK;
    if (!(routed_done && shared_done) && w->expert_quant.present &&
        (w->expert_quant.w13_type == GGUF_TYPE_Q4_K ||
         (w->shared.w1_type_per_layer && w->shared.w1_type_per_layer[layer] == GGUF_TYPE_Q4_K))) {
        TnQ8KActBlock *q = (TnQ8KActBlock *)tn_tls_scratch(
            &s_xb_scratch, (size_t)xb_n_blocks * sizeof(TnQ8KActBlock));
        if (q) tn_quantize_q8k(q, s->xb, xb_n_blocks);
        return q;
    }
    return NULL;
}

void moe_ffn_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
                     const MoEConfig *mc, int layer, ThreadPool *tp) {
    const int dim         = cfg->dim;
    const int expert_hdim = mc->expert_hidden_dim;
    const int top_k       = mc->num_experts_per_tok;

    /* Stack buffers for router outputs — tiny (≤512 floats / ints) */
    float expert_scores[MOE_SCORE_BUF_SIZE];
    int selected_experts[MOE_SCORE_BUF_SIZE];
    float selected_scores[MOE_SCORE_BUF_SIZE];

    moe_norm_and_route(s, w, cfg, mc, layer, tp, expert_scores, selected_experts, selected_scores);

    /* P4: Prefetch selected expert weight pages from mmap into OS page cache
     * immediately after routing. By the time expert 0's FFN finishes,
     * the OS will have page-faulted in experts 1..top_k-1's pages. */
    prefetch_selected(w, layer, dim, expert_hdim, selected_experts, top_k);

    /* Step 3: accumulate expert outputs into s->xb2 (zeroed first); each selected expert runs a
     * SwiGLU FFN, scaled by its routing score. Reuses s->hb (gate after activation) and s->hb2. */
    memset(s->xb2, 0, dim * sizeof(float));

    int routed_done = 0, shared_done = 0;
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    moe_fused_try(s, w, cfg, mc, layer, selected_experts, selected_scores, top_k, tp, &routed_done,
                  &shared_done);
    if (t_step && (routed_done || shared_done)) {
        tn_step_timing_add(TN_STEP_16_MOE_ROUTED, tn_step_timing_now_ns() - t_step);
    }

    const TnQ8KActBlock *xb_q8k = moe_quantize_xb(s, w, layer, dim, routed_done, shared_done);

    /* Batched path for Q4K gate/up: 3 threadpool dispatches per MoE layer (was top_k*3 = 18);
     * sequential per-expert loop for ternary / non-Q4K types. */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (routed_done) {
        /* handled by moe_fused_try() above */
    } else if (w->expert_quant.present && w->expert_quant.w13_type == GGUF_TYPE_Q4_K) {
        moe_legacy_routed_batched(s, w, cfg, mc, layer, selected_experts, selected_scores, top_k,
                                  xb_q8k, tp);
    } else {
        moe_legacy_routed_sequential(s, w, cfg, mc, layer, selected_experts, selected_scores, top_k,
                                     tp);
    }
    if (t_step && !routed_done) {
        tn_step_timing_add(TN_STEP_16_MOE_ROUTED, tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_moe_out", s->xb2, dim);

    /* Step 3b: the shared expert (DeepSeek-V2 style) is always active, on top of the routed ones. */
    if (!shared_done && mc->n_shared_experts > 0 && w->shared.w1 && w->shared.w1[layer]) {
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        moe_legacy_shared(s, w, cfg, mc, layer, xb_q8k, tp);
        if (t_step) {
            tn_step_timing_add(TN_STEP_17_SHARED_EXPERT, tn_step_timing_now_ns() - t_step);
        }
        DBG_DUMP(layer, "ffn_shexp", s->q, dim);
    }

    /* Step 4: residual connection, s->x += s->xb2 (same as the dense FFN). */
    tn_vec_add(s->x, s->x, s->xb2, dim);
    DBG_DUMP(layer, "ffn_out", s->x, dim);
}
