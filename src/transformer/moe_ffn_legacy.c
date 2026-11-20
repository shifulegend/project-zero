/* The expert FFN paths used when the fused two-dispatch path (moe_experts.c) does not handle an
 * expert: Q4_K-gate batched dispatch, the sequential per-expert fallback, and the shared expert. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_quant.h"
#include "core/gguf_reader.h"
#include "core/log.h"
#include "core/step_timing.h"
#include "math/matmul_q2k.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q5_0.h"
#include "math/matmul_q5_1.h"
#include "math/matmul_q8_0.h"
#include "math/parallel_matmul.h"
#include "math/simd_vec_ops.h"
#include "memory/tls_scratch.h"
#include "transformer/moe_ffn_internal.h"

/* Per-thread scratch (memory/tls_scratch.h; was file-scope statics that two concurrent callers
 * corrupted). Every user runs on the calling thread; pool workers receive the pointers by value.
 *   s_dequant_scratch  one expert weight matrix (dim x expert_hdim floats)
 *   s_batch_scratch    batched expert matmul outputs: gate [top_k*ehdim] | up | down [top_k*dim] */
static TnTlsScratch s_dequant_scratch = TN_TLS_SCRATCH_INIT;
static TnTlsScratch s_batch_scratch   = TN_TLS_SCRATCH_INIT;

static float *dequant_expert_weight(const tn_i8 *raw, int quant_type, size_t n_elems) {
    float *dq = (float *)tn_tls_scratch(&s_dequant_scratch, n_elems * sizeof(float));
    if (!dq) return NULL;
    switch (quant_type) {
        case 10:
            gguf_dequant_q2_k(dq, raw, n_elems);
            break; /* Q2_K */
        case 11:
            gguf_dequant_q3_k(dq, raw, n_elems);
            break; /* Q3_K */
        case 12:
            gguf_dequant_q4_k(dq, raw, n_elems);
            break; /* Q4_K */
        case 6:
            gguf_dequant_q5_0(dq, raw, n_elems);
            break; /* Q5_0 */
        case 7:
            gguf_dequant_q5_1(dq, raw, n_elems);
            break; /* Q5_1 */
        case 13:
            gguf_dequant_q5_k(dq, raw, n_elems);
            break; /* Q5_K */
        case 14:
            gguf_dequant_q6_k(dq, raw, n_elems);
            break; /* Q6_K */
        case 8:
            gguf_dequant_q8_0(dq, raw, n_elems);
            break; /* Q8_0 */
        case 20:
            gguf_dequant_iq4_nl(dq, raw, n_elems);
            break; /* IQ4_NL */
        default:   /* F32 already decoded (ternary / other) */
            for (size_t i = 0; i < n_elems; i++) {
                ((float *)dq)[i] = ((const float *)raw)[i];
            }
            break;
    }
    return dq;
}

/* The quant type of the experts' down projection in `layer` (per-layer override first). */
static int expert_w2_type(const TransformerWeights *w, int layer) {
    return (w->expert_quant.w2_per_layer && w->expert_quant.w2_per_layer[layer])
               ? w->expert_quant.w2_per_layer[layer]
               : w->expert_quant.w2_type;
}

/* ── Batched Q4_K path ──────────────────────────────────────────────────── */

/* Pointers for the valid selected experts: outputs into the batch buffer, weights, scores. */
typedef struct {
    float *gate[MOE_SCORE_BUF_SIZE];
    float *up[MOE_SCORE_BUF_SIZE];
    float *down[MOE_SCORE_BUF_SIZE];
    const uint8_t *w1[MOE_SCORE_BUF_SIZE];
    const uint8_t *w3[MOE_SCORE_BUF_SIZE];
    const uint8_t *w2[MOE_SCORE_BUF_SIZE];
    float score[MOE_SCORE_BUF_SIZE];
    int n; /* number of valid experts */
} ExpertBatch;

/* One entry per valid selected expert; `batch` holds gate | up | down outputs back to back. */
static void batch_collect(ExpertBatch *b, const TransformerWeights *w, const MoEConfig *mc, int dim,
                          int layer, const int *selected, const float *scores, int top_k,
                          float *batch) {
    const int expert_hdim = mc->expert_hidden_dim;
    const size_t gate_n   = (size_t)top_k * (size_t)expert_hdim;
    float *batch_gate     = batch;
    float *batch_up       = batch + gate_n;
    float *batch_down     = batch + 2 * gate_n;
    b->n                  = 0;
    for (int i = 0; i < top_k; i++) {
        int e = selected[i];
        if (e < 0 || e >= mc->num_experts) continue;
        moe_track_hit(layer, e);
        int k       = b->n;
        b->gate[k]  = batch_gate + (size_t)k * expert_hdim;
        b->up[k]    = batch_up + (size_t)k * expert_hdim;
        b->down[k]  = batch_down + (size_t)k * dim;
        b->w1[k]    = (const uint8_t *)w->moe.w1[layer][e];
        b->w3[k]    = (const uint8_t *)w->moe.w3[layer][e];
        b->w2[k]    = (const uint8_t *)w->moe.w2[layer][e];
        b->score[k] = scores[i];
        b->n++;
    }
}

/* Gate or up projections of all selected experts in one dispatch (pre-quantised xb when given). */
static void batch_q4k_proj(float *const *out, const uint8_t *const *wts, const float *xb,
                           const TnQ8KActBlock *xb_q8k, int dim, int expert_hdim, int n,
                           ThreadPool *tp) {
    if (xb_q8k)
        parallel_matmul_q4k_batch_preq(out, xb_q8k, wts, dim, expert_hdim, n, tp);
    else
        parallel_matmul_q4k_batch(out, xb, wts, dim, expert_hdim, n, tp);
}

/* Down projections (per-expert inputs): Q5_1 / Q5_0 / Q8_0 batched, Q4_K sequential, else
 * dequantised to F32. */
static void batch_down_proj(ExpertBatch *b, const TransformerWeights *w, int layer,
                            const int *selected, int w2_qtype, int dim, int expert_hdim,
                            ThreadPool *tp) {
    const int valid_k = b->n;
    if (w2_qtype == GGUF_TYPE_Q5_1) {
        parallel_matmul_q5_1_batch(b->down, b->gate, (const uint8_t *const *)b->w2, expert_hdim,
                                   dim, valid_k, tp);
    } else if (w2_qtype == GGUF_TYPE_Q5_0) {
        parallel_matmul_q5_0_batch(b->down, b->gate, (const uint8_t *const *)b->w2, expert_hdim,
                                   dim, valid_k, tp);
    } else if (w2_qtype == GGUF_TYPE_Q8_0) {
        parallel_matmul_q8_0_batch(b->down, b->gate, (const uint8_t *const *)b->w2, expert_hdim,
                                   dim, valid_k, tp);
    } else if (w2_qtype == GGUF_TYPE_Q4_K) {
        /* Q4K down: sequential (input is per-expert, not shared) */
        for (int i = 0; i < valid_k; i++)
            parallel_matmul_q4k(b->down[i], b->gate[i], b->w2[i], expert_hdim, dim, tp);
    } else {
        /* Other types: dequant + F32 sequential */
        for (int i = 0; i < valid_k; i++) {
            size_t n_el = (size_t)expert_hdim * dim;
            float *fw   = dequant_expert_weight(w->moe.w2[layer][selected[i]], w2_qtype, n_el);
            if (fw) parallel_matmul_float32(b->down[i], b->gate[i], fw, expert_hdim, dim, tp);
        }
    }
}

void moe_legacy_routed_batched(RunState *s, const TransformerWeights *w, const Config *cfg,
                               const MoEConfig *mc, int layer, const int *selected,
                               const float *scores, int top_k, const TnQ8KActBlock *xb_q8k,
                               ThreadPool *tp) {
    const int dim         = cfg->dim;
    const int expert_hdim = mc->expert_hidden_dim;
    const int w2_qtype    = expert_w2_type(w, layer);

    const size_t gate_n = (size_t)top_k * (size_t)expert_hdim;
    const size_t down_n = (size_t)top_k * (size_t)dim;
    float *batch = (float *)tn_tls_scratch(&s_batch_scratch, (2 * gate_n + down_n) * sizeof(float));
    if (!batch) {
        TN_LOG_ERR("moe", "[moe] out of memory for the expert batch buffers\n");
        abort();
    }
    ExpertBatch b;
    batch_collect(&b, w, mc, dim, layer, selected, scores, top_k, batch);
    if (b.n == 0) return;

    int64_t ts_gate = 0, ts_up = 0, ts_silu = 0, ts_down = 0, ts_acc = 0;
    if (tn_step_timing_enabled()) ts_gate = tn_step_timing_now_ns();

    /* Dispatch 1: all gate projections (Q4K, pre-quantized xb) */
    batch_q4k_proj(b.gate, b.w1, s->xb, xb_q8k, dim, expert_hdim, b.n, tp);
    if (tn_step_timing_enabled()) ts_up = tn_step_timing_now_ns();

    /* Dispatch 2: all up projections (Q4K, pre-quantized xb — no re-quantise) */
    batch_q4k_proj(b.up, b.w3, s->xb, xb_q8k, dim, expert_hdim, b.n, tp);
    if (tn_step_timing_enabled()) ts_silu = tn_step_timing_now_ns();

    /* SiLU + gate×up fusion (sequential, ~8K ops per expert) */
    for (int i = 0; i < b.n; i++) {
        if (cfg->act_type == 1)
            tn_relu2(b.gate[i], expert_hdim);
        else
            tn_silu(b.gate[i], expert_hdim);
        tn_vec_mul(b.gate[i], b.gate[i], b.up[i], expert_hdim);
    }
    if (tn_step_timing_enabled()) ts_down = tn_step_timing_now_ns();

    /* Dispatch 3: all down projections */
    batch_down_proj(&b, w, layer, selected, w2_qtype, dim, expert_hdim, tp);
    if (tn_step_timing_enabled()) ts_acc = tn_step_timing_now_ns();

    /* Weighted accumulate: xb2 += score[i] * down[i] */
    for (int i = 0; i < b.n; i++) tn_vec_saxpy(s->xb2, b.score[i], b.down[i], dim);

    /* Print sub-timing for first 3 MoE calls (diagnostic only) */
    static int sub_timing_count = 0;
    if (tn_step_timing_enabled() && sub_timing_count < 3) {
        sub_timing_count++;
        int64_t t_end = tn_step_timing_now_ns();
        TN_LOG_INF("sub_step16",
                   "[sub-step16 #%d] gate=%.3fms up=%.3fms silu=%.3fms down=%.3fms acc=%.3fms\n",
                   sub_timing_count, (ts_up - ts_gate) * 1e-6, (ts_silu - ts_up) * 1e-6,
                   (ts_down - ts_silu) * 1e-6, (ts_acc - ts_down) * 1e-6, (t_end - ts_acc) * 1e-6);
    }
}

/* ── Sequential per-expert path ─────────────────────────────────────────── */

/* The layer's gate / up quant type (per-layer override first; fixes mixed-type models). */
static int expert_w13_type(const TransformerWeights *w, int layer) {
    return (w->expert_quant.w13_per_layer && w->expert_quant.w13_per_layer[layer])
               ? w->expert_quant.w13_per_layer[layer]
               : w->expert_quant.w13_type;
}

/* One quantised expert: gate and up (fused Q2_K, else dequantised to F32), activation, down.
 * The result is left in s->q. */
static void expert_ffn_quantised(RunState *s, const TransformerWeights *w, const Config *cfg,
                                 int layer, int e, int dim, int expert_hdim, ThreadPool *tp) {
    size_t w13_elems = (size_t)dim * (size_t)expert_hdim;
    size_t w2_elems  = (size_t)expert_hdim * (size_t)dim;
    int w13_qtype    = expert_w13_type(w, layer);

    /* P7: Fused Q2K matvec — dequant + dot in one pass, no intermediate
     * float32 buffer.  Saves ~11.5 MB write + ~11.5 MB re-read vs old path. */
    if (w13_qtype == GGUF_TYPE_Q2_K) {
        parallel_matvec_q2k(s->hb, s->xb, (const uint8_t *)w->moe.w1[layer][e], dim, expert_hdim,
                            tp);
        parallel_matvec_q2k(s->hb2, s->xb, (const uint8_t *)w->moe.w3[layer][e], dim, expert_hdim,
                            tp);
    } else {
        float *fw1 = dequant_expert_weight(w->moe.w1[layer][e], w13_qtype, w13_elems);
        if (fw1) parallel_matmul_float32(s->hb, s->xb, fw1, dim, expert_hdim, tp);

        float *fw3 = dequant_expert_weight(w->moe.w3[layer][e], w13_qtype, w13_elems);
        if (fw3) parallel_matmul_float32(s->hb2, s->xb, fw3, dim, expert_hdim, tp);
    }

    if (cfg->act_type == 1)
        tn_relu2(s->hb, expert_hdim);
    else
        tn_silu(s->hb, expert_hdim);
    tn_vec_mul(s->hb, s->hb, s->hb2, expert_hdim);

    int w2_qtype = expert_w2_type(w, layer);
    if (w2_qtype == GGUF_TYPE_Q2_K) {
        /* P7: fused Q2K down projection */
        parallel_matvec_q2k(s->q, s->hb, (const uint8_t *)w->moe.w2[layer][e], expert_hdim, dim,
                            tp);
    } else if (w2_qtype == GGUF_TYPE_Q4_K) {
        parallel_matmul_q4k(s->q, s->hb, (const uint8_t *)w->moe.w2[layer][e], expert_hdim, dim,
                            tp);
    } else if (w2_qtype == GGUF_TYPE_Q5_1) {
        parallel_matmul_q5_1(s->q, s->hb, (const uint8_t *)w->moe.w2[layer][e], expert_hdim, dim,
                             tp);
    } else {
        float *fw2 = dequant_expert_weight(w->moe.w2[layer][e], w2_qtype, w2_elems);
        if (fw2) parallel_matmul_float32(s->q, s->hb, fw2, expert_hdim, dim, tp);
    }
}

/* One ternary-packed expert (native .bin format); the result is left in s->q. */
static void expert_ffn_ternary(RunState *s, const TransformerWeights *w, const Config *cfg,
                               int layer, int e, int dim, int expert_hdim, ThreadPool *tp) {
    parallel_ternary_matmul_packed(s->hb, s->xb, (const tn_u8 *)w->moe.w1[layer][e], dim,
                                   expert_hdim, w->moe.s1[layer][e], tp);

    parallel_ternary_matmul_packed(s->hb2, s->xb, (const tn_u8 *)w->moe.w3[layer][e], dim,
                                   expert_hdim, w->moe.s3[layer][e], tp);

    if (cfg->act_type == 1)
        tn_relu2(s->hb, expert_hdim);
    else
        tn_silu(s->hb, expert_hdim);
    tn_vec_mul(s->hb, s->hb, s->hb2, expert_hdim);

    parallel_ternary_matmul_packed(s->q, s->hb, (const tn_u8 *)w->moe.w2[layer][e], expert_hdim,
                                   dim, w->moe.s2[layer][e], tp);
}

void moe_legacy_routed_sequential(RunState *s, const TransformerWeights *w, const Config *cfg,
                                  const MoEConfig *mc, int layer, const int *selected,
                                  const float *scores, int top_k, ThreadPool *tp) {
    const int dim         = cfg->dim;
    const int expert_hdim = mc->expert_hidden_dim;
    for (int i = 0; i < top_k; i++) {
        int e    = selected[i];
        float sc = scores[i];
        if (e < 0 || e >= mc->num_experts) continue;
        moe_track_hit(layer, e);

        if (w->expert_quant.present)
            expert_ffn_quantised(s, w, cfg, layer, e, dim, expert_hdim, tp);
        else
            expert_ffn_ternary(s, w, cfg, layer, e, dim, expert_hdim, tp);

        tn_vec_saxpy(s->xb2, sc, s->q, dim);
    }
}

/* ── Shared expert ──────────────────────────────────────────────────────── */

/* Gate (w1) and up (w3) of the shared expert into s->hb / s->hb2. */
static void shared_gate_up(RunState *s, const TransformerWeights *w, int layer, int dim, int shdim,
                           int w1t, int w3t, const TnQ8KActBlock *xb_q8k, ThreadPool *tp) {
    /* P2: fuse shared expert gate (w1) + up (w3) into one batch dispatch
     * when both are Q4K and pre-quantized acts are available.
     * Saves 1 thread-pool dispatch per MoE layer (26 total per token). */
    if (w1t == GGUF_TYPE_Q4_K && w3t == GGUF_TYPE_Q4_K && xb_q8k) {
        float *sh_outs[2]       = {s->hb, s->hb2};
        const uint8_t *sh_ws[2] = {(const uint8_t *)w->shared.w1[layer],
                                   (const uint8_t *)w->shared.w3[layer]};
        parallel_matmul_q4k_batch_preq(sh_outs, xb_q8k, sh_ws, dim, shdim, 2, tp);
        return;
    }
    /* Gate projection (w1) */
    if (w1t == GGUF_TYPE_Q4_K && xb_q8k)
        parallel_matmul_q4k_preq(s->hb, xb_q8k, (const uint8_t *)w->shared.w1[layer], dim, shdim,
                                 tp);
    else if (w1t == GGUF_TYPE_Q4_K)
        parallel_matmul_q4k(s->hb, s->xb, (const uint8_t *)w->shared.w1[layer], dim, shdim, tp);
    else
        parallel_matmul_float32(s->hb, s->xb, (const float *)w->shared.w1[layer], dim, shdim, tp);

    /* Up projection (w3) */
    if (w3t == GGUF_TYPE_Q4_K && xb_q8k)
        parallel_matmul_q4k_preq(s->hb2, xb_q8k, (const uint8_t *)w->shared.w3[layer], dim, shdim,
                                 tp);
    else if (w3t == GGUF_TYPE_Q4_K)
        parallel_matmul_q4k(s->hb2, s->xb, (const uint8_t *)w->shared.w3[layer], dim, shdim, tp);
    else
        parallel_matmul_float32(s->hb2, s->xb, (const float *)w->shared.w3[layer], dim, shdim, tp);
}

/* Each projection dispatches on shared_w{1,2,3}_type_per_layer[layer]: GGUF_TYPE_Q4_K runs the
 * fused Q4K kernel (raw bytes, on-the-fly dequant), any other type the F32 path (tensor was
 * dequantised at load). Handles the mixed-type case: e.g. ffn_down_shexp is Q5_K on layer 1 but
 * Q4_K on every other MoE layer. */
void moe_legacy_shared(RunState *s, const TransformerWeights *w, const Config *cfg,
                       const MoEConfig *mc, int layer, const TnQ8KActBlock *xb_q8k,
                       ThreadPool *tp) {
    const int dim   = cfg->dim;
    const int shdim = mc->shared_expert_hidden_dim;
    int w1t         = w->shared.w1_type_per_layer ? w->shared.w1_type_per_layer[layer] : 0;
    int w2t         = w->shared.w2_type_per_layer ? w->shared.w2_type_per_layer[layer] : 0;
    int w3t         = w->shared.w3_type_per_layer ? w->shared.w3_type_per_layer[layer] : 0;

    shared_gate_up(s, w, layer, dim, shdim, w1t, w3t, xb_q8k, tp);

    if (cfg->act_type == 1)
        tn_relu2(s->hb, shdim);
    else
        tn_silu(s->hb, shdim);
    tn_vec_mul(s->hb, s->hb, s->hb2, shdim);

    /* Down projection (w2) */
    if (w2t == GGUF_TYPE_Q4_K)
        parallel_matmul_q4k(s->q, s->hb, (const uint8_t *)w->shared.w2[layer], shdim, dim, tp);
    else
        parallel_matmul_float32(s->q, s->hb, (const float *)w->shared.w2[layer], shdim, dim, tp);

    tn_vec_saxpy(s->xb2, 1.0f, s->q, dim);
}
