#include "transformer/moe_ffn.h"
#include "transformer/moe_router.h"
#include "math/parallel_matmul.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q2k.h"
#include "math/matmul_q5_1.h"
#include "math/matmul_q5_0.h"
#include "math/matmul_q8_0.h"
#include "math/simd_dispatch.h"
#include "core/gguf_quant.h"
#include "core/gguf_reader.h"
#include "core/debug.h"
#include "core/step_timing.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>

/* Maximum experts supported without heap allocation (stack-allocated buffers) */
#define MOE_MAX_EXPERTS_STACK 512

/* Maximum expert hidden dim supported on stack (for score & index buffers only,
 * not for the FFN output — that uses RunState's hb/hb2 which are heap-allocated) */
#define MOE_SCORE_BUF_SIZE    MOE_MAX_EXPERTS_STACK

/* Lazy-allocated dequantization scratch buffer — holds one expert weight matrix
 * (dim × expert_hdim floats).  Grown as needed; freed at process exit. */
static float  *s_dequant_buf       = NULL;
static size_t  s_dequant_buf_elems = 0;

/* Module-level scratch buffers for batched expert matmul outputs.
 * moe_ffn_forward is not re-entrant (one call at a time from the main thread;
 * internal parallelism is via tp only), so static buffers are safe. */
static float *s_gate_buf   = NULL;  /* [top_k * expert_hdim] */
static float *s_up_buf     = NULL;  /* [top_k * expert_hdim] */
static float *s_down_buf   = NULL;  /* [top_k * dim] */
static int    s_batch_top_k  = 0;
static int    s_batch_ehdim  = 0;
static int    s_batch_dim    = 0;

/* P1: Pre-quantized Q8K activation buffer for s->xb.
 * Quantized once per MoE layer; reused for all Q4K matmuls sharing xb
 * (Q4K batched gate+up, shared expert w1+w3). */
static TnQ8KActBlock *s_xb_q8k        = NULL;
static int            s_xb_q8k_blocks = 0;

static float *dequant_expert_weight(
        const tn_i8 *raw, int quant_type, size_t n_elems) {
    if (s_dequant_buf_elems < n_elems) {
        free(s_dequant_buf);
        s_dequant_buf = (float *)malloc(n_elems * sizeof(float));
        s_dequant_buf_elems = s_dequant_buf ? n_elems : 0;
        if (!s_dequant_buf) return NULL;
    }
    switch (quant_type) {
        case 10: gguf_dequant_q2_k(s_dequant_buf, raw, n_elems); break; /* Q2_K */
        case 11: gguf_dequant_q3_k(s_dequant_buf, raw, n_elems); break; /* Q3_K */
        case 12: gguf_dequant_q4_k(s_dequant_buf, raw, n_elems); break; /* Q4_K */
        case  6: gguf_dequant_q5_0(s_dequant_buf, raw, n_elems); break; /* Q5_0 */
        case  7: gguf_dequant_q5_1(s_dequant_buf, raw, n_elems); break; /* Q5_1 */
        case 13: gguf_dequant_q5_k(s_dequant_buf, raw, n_elems); break; /* Q5_K */
        case 14: gguf_dequant_q6_k(s_dequant_buf, raw, n_elems); break; /* Q6_K */
        case  8: gguf_dequant_q8_0(s_dequant_buf, raw, n_elems); break; /* Q8_0 */
        case 20: gguf_dequant_iq4_nl(s_dequant_buf, raw, n_elems); break; /* IQ4_NL */
        default: /* F32 already decoded (ternary / other) */
            for (size_t i = 0; i < n_elems; i++) {
                ((float *)s_dequant_buf)[i] = ((const float *)raw)[i];
            }
            break;
    }
    return s_dequant_buf;
}

/* P4: Compute raw byte size of a quantized weight tensor.
 * Used to pass the correct length to madvise(MADV_WILLNEED). */
static size_t quant_tensor_bytes(int quant_type, size_t n_elems) {
    switch (quant_type) {
        case 10: return (n_elems / 256) * 84;   /* Q2_K: 84 bytes/block */
        case 11: return (n_elems / 256) * 110;  /* Q3_K: 110 bytes/block */
        case 12: return (n_elems / 256) * 144;  /* Q4_K: 144 bytes/block */
        case 13: return (n_elems / 256) * 176;  /* Q5_K: 176 bytes/block */
        case 14: return (n_elems / 256) * 210;  /* Q6_K: 210 bytes/block */
        case  6: return (n_elems / 32)  * 22;   /* Q5_0: 22 bytes/block */
        case  7: return (n_elems / 32)  * 24;   /* Q5_1: 24 bytes/block */
        case  8: return (n_elems / 32)  * 34;   /* Q8_0: 34 bytes/block */
        case 20: return (n_elems / 32)  * 18;   /* IQ4_NL: 18 bytes/block */
        default: return n_elems * 4;             /* F32 fallback */
    }
}

/* P4: Issue MADV_WILLNEED for all selected expert weight pages right after
 * routing (before the FFN loop) so the OS can page-in experts 1..top_k-1
 * while we compute expert 0. This amortises mmap cold-page-fault latency. */
static void prefetch_expert_weights(
        const TransformerWeights *w,
        int layer, const int *selected, int top_k,
        size_t w13_bytes, size_t w2_bytes) {
    for (int i = 0; i < top_k; i++) {
        int e = selected[i];
        if (e < 0) continue;
        if (w->moe_w1 && w->moe_w1[layer] && w->moe_w1[layer][e])
            madvise((void *)w->moe_w1[layer][e], w13_bytes, MADV_WILLNEED);
        if (w->moe_w3 && w->moe_w3[layer] && w->moe_w3[layer][e])
            madvise((void *)w->moe_w3[layer][e], w13_bytes, MADV_WILLNEED);
        if (w->moe_w2 && w->moe_w2[layer] && w->moe_w2[layer][e])
            madvise((void *)w->moe_w2[layer][e], w2_bytes, MADV_WILLNEED);
    }
}


static int g_track_n_layers    = 0;
static int g_track_n_experts   = 0;
static uint32_t *g_expert_hits = NULL;  /* [n_layers * num_experts] */

void moe_expert_tracking_reset(int n_layers, int num_experts) {
    int needed = n_layers * num_experts;
    if (g_track_n_layers != n_layers || g_track_n_experts != num_experts) {
        free(g_expert_hits);
        g_expert_hits = (uint32_t *)calloc((size_t)needed, sizeof(uint32_t));
        g_track_n_layers  = n_layers;
        g_track_n_experts = num_experts;
    } else if (g_expert_hits) {
        memset(g_expert_hits, 0, (size_t)needed * sizeof(uint32_t));
    }
}

void moe_expert_tracking_print(int n_layers, int num_experts) {
    if (!g_expert_hits || n_layers <= 0 || num_experts <= 0) return;

    printf("\n--- Expert Utilisation Summary (%d MoE layers × %d experts) ---\n",
           n_layers, num_experts);

    int total_unique_global = 0;
    uint32_t global_hits[MOE_MAX_EXPERTS_STACK] = {0};

    for (int l = 0; l < n_layers; l++) {
        uint32_t *row = g_expert_hits + l * num_experts;
        int unique = 0;
        uint32_t total = 0;
        int top5[5] = {-1,-1,-1,-1,-1};
        uint32_t top5v[5] = {0};

        for (int e = 0; e < num_experts; e++) {
            if (row[e] > 0) { unique++; global_hits[e] += row[e]; }
            total += row[e];
            /* maintain top-5 */
            for (int k = 0; k < 5; k++) {
                if (row[e] > top5v[k]) {
                    /* shift down */
                    for (int j = 4; j > k; j--) { top5[j]=top5[j-1]; top5v[j]=top5v[j-1]; }
                    top5[k] = e; top5v[k] = row[e]; break;
                }
            }
        }

        printf("  L%02d: %3d unique / %2d total  (invocations: %4u)  top:", l, unique, num_experts, total);
        for (int k = 0; k < 5 && top5[k] >= 0; k++) printf(" E%d(%u)", top5[k], top5v[k]);
        printf("\n");
    }

    /* Global expert utilisation across all layers */
    int global_unique = 0;
    for (int e = 0; e < num_experts; e++) if (global_hits[e] > 0) global_unique++;
    printf("  Global: %d / %d experts activated (%.1f%%)\n",
           global_unique, num_experts, 100.0f * global_unique / num_experts);
    (void)total_unique_global;
    printf("---\n");
}

void moe_expert_tracking_free(void) {
    free(g_expert_hits);
    g_expert_hits     = NULL;
    g_track_n_layers  = 0;
    g_track_n_experts = 0;
}
/* -------------------------------------------------------------------------- */

void moe_ffn_forward(RunState              *s,
                     const TransformerWeights *w,
                     const Config          *cfg,
                     const MoEConfig       *mc,
                     int                    layer,
                     ThreadPool            *tp) {
    int dim            = cfg->dim;
    int expert_hdim    = mc->expert_hidden_dim;
    int num_experts    = mc->num_experts;
    int top_k          = mc->num_experts_per_tok;

    /* Stack buffers for router outputs — tiny (≤512 floats / ints) */
    float expert_scores[MOE_SCORE_BUF_SIZE];
    int   selected_experts[MOE_SCORE_BUF_SIZE];
    float selected_scores[MOE_SCORE_BUF_SIZE];

    /* ----------------------------------------------------------------
     * Step 1: RMSNorm  →  s->xb
     * Identical to the dense path; normalise s->x into s->xb.
     * ---------------------------------------------------------------- */
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    tn_rmsnorm(s->xb, s->x, w->rms_ffn_weight[layer], dim, cfg->rms_norm_eps);
    if (t_step) {
        tn_step_timing_add(TN_STEP_13_PRE_FFN_RMSNORM,
                           tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_norm", s->xb, dim);

    /* ----------------------------------------------------------------
     * Step 2: Router — select top_k experts
     * Gate weight is w->moe_gate_w[layer] — F32 [num_experts × dim].
     * ---------------------------------------------------------------- */
    /* Dump raw router logits (pre-softmax) — only when tensor dump is active */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (g_dump_fp) {
        float raw_logits[MOE_SCORE_BUF_SIZE];
        parallel_matmul_float32(raw_logits, s->xb,
                                (const float *)w->moe_gate_w[layer], dim, num_experts, tp);
        DBG_DUMP(layer, "router_logits", raw_logits, num_experts);
    }

    /* Router computes gate matmul, softmax, top-k in one call */
    moe_router_forward(expert_scores,
                       selected_experts,
                       selected_scores,
                       s->xb,
                       (const float *)w->moe_gate_w[layer],
                       dim, num_experts, top_k, tp);
    if (t_step) {
        tn_step_timing_add(TN_STEP_15_MOE_ROUTING,
                           tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_moe_probs", expert_scores, num_experts);

    if (g_tn_verbose) {
        fprintf(stderr, "[DBG] L%02d router: xb_max=%.4f  experts=[",
                layer, (float)0.0f);
        float xb_max = 0.0f;
        for (int ii = 0; ii < dim; ii++) { float a = s->xb[ii] < 0 ? -s->xb[ii] : s->xb[ii]; if (a > xb_max) xb_max = a; }
        fprintf(stderr, "[DBG] L%02d router: xb_max_abs=%.4f  selected=[", layer, xb_max);
        for (int ii = 0; ii < top_k; ii++)
            fprintf(stderr, "%d(%.3f)%s", selected_experts[ii], selected_scores[ii], ii<top_k-1?", ":"");
        fprintf(stderr, "]\n");
    }

    /* P4: Prefetch selected expert weight pages from mmap into OS page cache
     * immediately after routing. By the time expert 0's FFN finishes,
     * the OS will have page-faulted in experts 1..top_k-1's pages. */
    if (w->has_expert_quant) {
        int w2_qtype = (w->expert_w2_quant_per_layer &&
                        w->expert_w2_quant_per_layer[layer])
                       ? w->expert_w2_quant_per_layer[layer]
                       : w->expert_w2_quant_type;
        size_t w13_bytes = quant_tensor_bytes(w->expert_w13_quant_type,
                                              (size_t)dim * expert_hdim);
        size_t w2_bytes  = quant_tensor_bytes(w2_qtype,
                                              (size_t)expert_hdim * dim);
        prefetch_expert_weights(w, layer, selected_experts, top_k,
                                w13_bytes, w2_bytes);
    }

    /* ----------------------------------------------------------------
     * Step 3: Accumulate expert outputs into s->xb2 (zeroed first)
     * For each selected expert: run SwiGLU FFN, scale, accumulate.
     * Reuses s->hb (gate after activation) and s->hb2 (up proj).
     * ---------------------------------------------------------------- */
    memset(s->xb2, 0, dim * sizeof(float));

    /* P1: Pre-quantize s->xb to Q8K once — reused by all Q4K dispatches
     * (batched gate+up for routed experts, shared expert w1+w3). */
    int xb_n_blocks = dim / TN_Q8K_BLOCK;
    if (s_xb_q8k_blocks < xb_n_blocks) {
        free(s_xb_q8k);
        s_xb_q8k = (TnQ8KActBlock *)malloc(
            (size_t)xb_n_blocks * sizeof(TnQ8KActBlock));
        s_xb_q8k_blocks = s_xb_q8k ? xb_n_blocks : 0;
    }
    if (s_xb_q8k && w->has_expert_quant &&
        (w->expert_w13_quant_type == GGUF_TYPE_Q4_K ||
         (w->shared_w1_type_per_layer &&
          w->shared_w1_type_per_layer[layer] == GGUF_TYPE_Q4_K)))
        tn_quantize_q8k(s_xb_q8k, s->xb, xb_n_blocks);

    /* ----------------------------------------------------------------
     * Expert FFN — batched path for Q4K gate/up + Q5_1 down:
     *   3 threadpool dispatches per MoE layer (was top_k×3 = 18).
     *   Workers stay active across all expert rows in a single dispatch,
     *   eliminating spin-wait gaps between sequential per-expert calls.
     *
     * Fallback: sequential per-expert loop for ternary / non-Q4K types.
     * ---------------------------------------------------------------- */
    t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->has_expert_quant &&
        w->expert_w13_quant_type == GGUF_TYPE_Q4_K) {

        int w2_qtype = (w->expert_w2_quant_per_layer &&
                        w->expert_w2_quant_per_layer[layer])
                       ? w->expert_w2_quant_per_layer[layer]
                       : w->expert_w2_quant_type;

        /* Grow module-level scratch buffers if dimensions increased */
        if (s_batch_top_k < top_k || s_batch_ehdim < expert_hdim ||
            s_batch_dim < dim) {
            free(s_gate_buf); free(s_up_buf); free(s_down_buf);
            s_gate_buf = (float *)malloc((size_t)top_k * expert_hdim * sizeof(float));
            s_up_buf   = (float *)malloc((size_t)top_k * expert_hdim * sizeof(float));
            s_down_buf = (float *)malloc((size_t)top_k * dim          * sizeof(float));
            s_batch_top_k = top_k;
            s_batch_ehdim = expert_hdim;
            s_batch_dim   = dim;
        }

        /* Build pointer arrays — one entry per valid selected expert */
        float         *gate_ptrs[MOE_SCORE_BUF_SIZE];
        float         *up_ptrs  [MOE_SCORE_BUF_SIZE];
        float         *down_ptrs[MOE_SCORE_BUF_SIZE];
        const uint8_t *w1_ptrs  [MOE_SCORE_BUF_SIZE];
        const uint8_t *w3_ptrs  [MOE_SCORE_BUF_SIZE];
        const uint8_t *w2_ptrs  [MOE_SCORE_BUF_SIZE];
        float          sel_sc   [MOE_SCORE_BUF_SIZE];
        int            valid_k  = 0;

        for (int i = 0; i < top_k; i++) {
            int e = selected_experts[i];
            if (e < 0 || e >= num_experts) continue;
            if (g_expert_hits && layer < g_track_n_layers && e < g_track_n_experts)
                g_expert_hits[layer * g_track_n_experts + e]++;
            gate_ptrs[valid_k] = s_gate_buf + (size_t)valid_k * expert_hdim;
            up_ptrs  [valid_k] = s_up_buf   + (size_t)valid_k * expert_hdim;
            down_ptrs[valid_k] = s_down_buf + (size_t)valid_k * dim;
            w1_ptrs  [valid_k] = (const uint8_t *)w->moe_w1[layer][e];
            w3_ptrs  [valid_k] = (const uint8_t *)w->moe_w3[layer][e];
            w2_ptrs  [valid_k] = (const uint8_t *)w->moe_w2[layer][e];
            sel_sc   [valid_k] = selected_scores[i];
            valid_k++;
        }

        if (valid_k > 0) {
            int64_t ts_gate = 0, ts_up = 0, ts_silu = 0, ts_down = 0, ts_acc = 0;
            if (tn_step_timing_enabled()) ts_gate = tn_step_timing_now_ns();

            /* Dispatch 1: all gate projections (Q4K, pre-quantized xb) */
            if (s_xb_q8k)
                parallel_matmul_q4k_batch_preq((float * const *)gate_ptrs,
                                                s_xb_q8k,
                                                (const uint8_t * const *)w1_ptrs,
                                                dim, expert_hdim, valid_k, tp);
            else
                parallel_matmul_q4k_batch((float * const *)gate_ptrs, s->xb,
                                           (const uint8_t * const *)w1_ptrs,
                                           dim, expert_hdim, valid_k, tp);
            if (tn_step_timing_enabled()) { ts_up = tn_step_timing_now_ns(); }

            /* Dispatch 2: all up projections (Q4K, pre-quantized xb — no re-quantise) */
            if (s_xb_q8k)
                parallel_matmul_q4k_batch_preq((float * const *)up_ptrs,
                                                s_xb_q8k,
                                                (const uint8_t * const *)w3_ptrs,
                                                dim, expert_hdim, valid_k, tp);
            else
                parallel_matmul_q4k_batch((float * const *)up_ptrs, s->xb,
                                           (const uint8_t * const *)w3_ptrs,
                                           dim, expert_hdim, valid_k, tp);
            if (tn_step_timing_enabled()) { ts_silu = tn_step_timing_now_ns(); }

            /* SiLU + gate×up fusion (sequential, ~8K ops per expert) */
            for (int i = 0; i < valid_k; i++) {
                if (cfg->act_type == 1) tn_relu2(gate_ptrs[i], expert_hdim);
                else                    tn_silu(gate_ptrs[i], expert_hdim);
                tn_vec_mul(gate_ptrs[i], gate_ptrs[i], up_ptrs[i], expert_hdim);
            }
            if (tn_step_timing_enabled()) { ts_down = tn_step_timing_now_ns(); }

            /* Dispatch 3: all down projections (per-expert inputs) */
            if (w2_qtype == GGUF_TYPE_Q5_1) {
                parallel_matmul_q5_1_batch(down_ptrs, gate_ptrs,
                                            (const uint8_t * const *)w2_ptrs,
                                            expert_hdim, dim, valid_k, tp);
            } else if (w2_qtype == GGUF_TYPE_Q5_0) {
                parallel_matmul_q5_0_batch(down_ptrs, gate_ptrs,
                                            (const uint8_t * const *)w2_ptrs,
                                            expert_hdim, dim, valid_k, tp);
            } else if (w2_qtype == GGUF_TYPE_Q8_0) {
                parallel_matmul_q8_0_batch(down_ptrs, gate_ptrs,
                                            (const uint8_t * const *)w2_ptrs,
                                            expert_hdim, dim, valid_k, tp);
            } else if (w2_qtype == GGUF_TYPE_Q4_K) {
                /* Q4K down: sequential (input is per-expert, not shared) */
                for (int i = 0; i < valid_k; i++)
                    parallel_matmul_q4k(down_ptrs[i], gate_ptrs[i],
                                        w2_ptrs[i], expert_hdim, dim, tp);
            } else {
                /* Other types: dequant + F32 sequential */
                for (int i = 0; i < valid_k; i++) {
                    size_t n_el = (size_t)expert_hdim * dim;
                    float *fw = dequant_expert_weight(
                        w->moe_w2[layer][selected_experts[i]], w2_qtype, n_el);
                    if (fw) parallel_matmul_float32(down_ptrs[i], gate_ptrs[i],
                                                    fw, expert_hdim, dim, tp);
                }
            }

            if (tn_step_timing_enabled()) { ts_acc = tn_step_timing_now_ns(); }

            /* Weighted accumulate: xb2 += sel_sc[i] * down_ptrs[i] */
            for (int i = 0; i < valid_k; i++)
                tn_vec_saxpy(s->xb2, sel_sc[i], down_ptrs[i], dim);

            /* Print sub-timing for first 3 MoE calls (diagnostic only) */
            static int sub_timing_count = 0;
            if (tn_step_timing_enabled() && sub_timing_count < 3) {
                sub_timing_count++;
                int64_t t_end = tn_step_timing_now_ns();
                fprintf(stderr,
                    "[sub-step16 #%d] gate=%.3fms up=%.3fms silu=%.3fms down=%.3fms acc=%.3fms\n",
                    sub_timing_count,
                    (ts_up   - ts_gate) * 1e-6,
                    (ts_silu - ts_up  ) * 1e-6,
                    (ts_down - ts_silu) * 1e-6,
                    (ts_acc  - ts_down) * 1e-6,
                    (t_end   - ts_acc ) * 1e-6);
            }
        }

    } else {
        /* Sequential fallback for ternary or non-Q4K expert weights */
        for (int i = 0; i < top_k; i++) {
            int   e   = selected_experts[i];
            float sc  = selected_scores[i];
            if (e < 0 || e >= num_experts) continue;

            if (g_expert_hits && layer < g_track_n_layers && e < g_track_n_experts)
                g_expert_hits[layer * g_track_n_experts + e]++;

            if (w->has_expert_quant) {
                size_t w13_elems = (size_t)dim * (size_t)expert_hdim;
                size_t w2_elems  = (size_t)expert_hdim * (size_t)dim;

                /* Resolve per-layer w13 quant type (fixes mixed-type models like DASH variant) */
                int w13_qtype = (w->expert_w13_quant_per_layer &&
                                 w->expert_w13_quant_per_layer[layer])
                                ? w->expert_w13_quant_per_layer[layer]
                                : w->expert_w13_quant_type;

                /* P7: Fused Q2K matvec — dequant + dot in one pass, no intermediate
                 * float32 buffer.  Saves ~11.5 MB write + ~11.5 MB re-read vs old path. */
                if (w13_qtype == GGUF_TYPE_Q2_K) {
                    parallel_matvec_q2k(s->hb, s->xb,
                                        (const uint8_t *)w->moe_w1[layer][e],
                                        dim, expert_hdim, tp);
                    parallel_matvec_q2k(s->hb2, s->xb,
                                        (const uint8_t *)w->moe_w3[layer][e],
                                        dim, expert_hdim, tp);
                } else {
                    float *fw1 = dequant_expert_weight(w->moe_w1[layer][e],
                                                       w13_qtype, w13_elems);
                    if (fw1)
                        parallel_matmul_float32(s->hb, s->xb, fw1, dim, expert_hdim, tp);

                    float *fw3 = dequant_expert_weight(w->moe_w3[layer][e],
                                                       w13_qtype, w13_elems);
                    if (fw3)
                        parallel_matmul_float32(s->hb2, s->xb, fw3, dim, expert_hdim, tp);
                }

                if (cfg->act_type == 1) tn_relu2(s->hb, expert_hdim);
                else                    tn_silu(s->hb, expert_hdim);
                tn_vec_mul(s->hb, s->hb, s->hb2, expert_hdim);

                int w2_qtype = (w->expert_w2_quant_per_layer &&
                                w->expert_w2_quant_per_layer[layer])
                               ? w->expert_w2_quant_per_layer[layer]
                               : w->expert_w2_quant_type;
                if (w2_qtype == GGUF_TYPE_Q2_K) {
                    /* P7: fused Q2K down projection */
                    parallel_matvec_q2k(s->q, s->hb,
                                        (const uint8_t *)w->moe_w2[layer][e],
                                        expert_hdim, dim, tp);
                } else if (w2_qtype == GGUF_TYPE_Q4_K) {
                    parallel_matmul_q4k(s->q, s->hb,
                                        (const uint8_t *)w->moe_w2[layer][e],
                                        expert_hdim, dim, tp);
                } else if (w2_qtype == GGUF_TYPE_Q5_1) {
                    parallel_matmul_q5_1(s->q, s->hb,
                                         (const uint8_t *)w->moe_w2[layer][e],
                                         expert_hdim, dim, tp);
                } else {
                    float *fw2 = dequant_expert_weight(w->moe_w2[layer][e],
                                                       w2_qtype, w2_elems);
                    if (fw2)
                        parallel_matmul_float32(s->q, s->hb, fw2,
                                                expert_hdim, dim, tp);
                }
            } else {
                /* Ternary-packed weights (native .bin format) */
                parallel_ternary_matmul_packed(
                    s->hb, s->xb,
                    (const tn_u8 *)w->moe_w1[layer][e],
                    dim, expert_hdim, w->moe_s1[layer][e], tp);

                parallel_ternary_matmul_packed(
                    s->hb2, s->xb,
                    (const tn_u8 *)w->moe_w3[layer][e],
                    dim, expert_hdim, w->moe_s3[layer][e], tp);

                if (cfg->act_type == 1) tn_relu2(s->hb, expert_hdim);
                else                    tn_silu(s->hb, expert_hdim);
                tn_vec_mul(s->hb, s->hb, s->hb2, expert_hdim);

                parallel_ternary_matmul_packed(
                    s->q, s->hb,
                    (const tn_u8 *)w->moe_w2[layer][e],
                    expert_hdim, dim, w->moe_s2[layer][e], tp);
            }

            tn_vec_saxpy(s->xb2, sc, s->q, dim);
        }
    }
    if (t_step) {
        tn_step_timing_add(TN_STEP_16_MOE_ROUTED,
                           tn_step_timing_now_ns() - t_step);
    }
    DBG_DUMP(layer, "ffn_moe_out", s->xb2, dim);

    /* ----------------------------------------------------------------
     * Step 3b: Shared expert FFN (DeepSeek-V2 style)
     * Always-active: runs in addition to top-k routed experts.
     * Accumulates directly into xb2 with weight 1.0 (no routing scale).
     * Each projection dispatches on shared_w{1,2,3}_type_per_layer[layer]:
     *   GGUF_TYPE_Q4_K (12) → fused Q4K kernel (raw bytes, on-the-fly dequant)
     *   any other type      → F32 dequant path (tensor was dequantized at load)
     * This handles the mixed-type case: e.g. ffn_down_shexp is Q5_K on layer 1
     * but Q4_K on all other MoE layers.
     * ---------------------------------------------------------------- */
    if (mc->n_shared_experts > 0 && w->moe_shared_w1 && w->moe_shared_w1[layer]) {
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int shdim = mc->shared_expert_hidden_dim;
        int w1t = w->shared_w1_type_per_layer ? w->shared_w1_type_per_layer[layer] : 0;
        int w2t = w->shared_w2_type_per_layer ? w->shared_w2_type_per_layer[layer] : 0;
        int w3t = w->shared_w3_type_per_layer ? w->shared_w3_type_per_layer[layer] : 0;

        /* P2: fuse shared expert gate (w1) + up (w3) into one batch dispatch
         * when both are Q4K and pre-quantized acts are available.
         * Saves 1 thread-pool dispatch per MoE layer (26 total per token). */
        if (w1t == GGUF_TYPE_Q4_K && w3t == GGUF_TYPE_Q4_K && s_xb_q8k) {
            float         *sh_outs[2] = { s->hb,  s->hb2 };
            const uint8_t *sh_ws[2]   = {
                (const uint8_t *)w->moe_shared_w1[layer],
                (const uint8_t *)w->moe_shared_w3[layer]
            };
            parallel_matmul_q4k_batch_preq(sh_outs, s_xb_q8k,
                                            sh_ws, dim, shdim, 2, tp);
        } else {
            /* Gate projection (w1) */
            if (w1t == GGUF_TYPE_Q4_K && s_xb_q8k)
                parallel_matmul_q4k_preq(s->hb, s_xb_q8k,
                                         (const uint8_t *)w->moe_shared_w1[layer],
                                         dim, shdim, tp);
            else if (w1t == GGUF_TYPE_Q4_K)
                parallel_matmul_q4k(s->hb, s->xb,
                                    (const uint8_t *)w->moe_shared_w1[layer],
                                    dim, shdim, tp);
            else
                parallel_matmul_float32(s->hb, s->xb,
                                        (const float *)w->moe_shared_w1[layer],
                                        dim, shdim, tp);

            /* Up projection (w3) */
            if (w3t == GGUF_TYPE_Q4_K && s_xb_q8k)
                parallel_matmul_q4k_preq(s->hb2, s_xb_q8k,
                                         (const uint8_t *)w->moe_shared_w3[layer],
                                         dim, shdim, tp);
            else if (w3t == GGUF_TYPE_Q4_K)
                parallel_matmul_q4k(s->hb2, s->xb,
                                    (const uint8_t *)w->moe_shared_w3[layer],
                                    dim, shdim, tp);
            else
                parallel_matmul_float32(s->hb2, s->xb,
                                        (const float *)w->moe_shared_w3[layer],
                                        dim, shdim, tp);
        }

        if (cfg->act_type == 1) tn_relu2(s->hb, shdim);
        else                    tn_silu(s->hb, shdim);
        tn_vec_mul(s->hb, s->hb, s->hb2, shdim);

        /* Down projection (w2) */
        if (w2t == GGUF_TYPE_Q4_K)
            parallel_matmul_q4k(s->q, s->hb,
                                (const uint8_t *)w->moe_shared_w2[layer],
                                shdim, dim, tp);
        else
            parallel_matmul_float32(s->q, s->hb,
                                    (const float *)w->moe_shared_w2[layer],
                                    shdim, dim, tp);

        tn_vec_saxpy(s->xb2, 1.0f, s->q, dim);
        if (t_step) {
            tn_step_timing_add(TN_STEP_17_SHARED_EXPERT,
                               tn_step_timing_now_ns() - t_step);
        }
        DBG_DUMP(layer, "ffn_shexp", s->q, dim);
    }

    /* ----------------------------------------------------------------
     * Step 4: Residual connection  →  s->x += s->xb2
     * Same as the dense FFN residual.
     * ---------------------------------------------------------------- */
    tn_vec_add(s->x, s->x, s->xb2, dim);
    DBG_DUMP(layer, "ffn_out", s->x, dim);
}
