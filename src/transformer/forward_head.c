/*
 * src/transformer/forward_head.c
 *
 * The model's output head: final RMSNorm and the classifier matmul that turns the last hidden state
 * into logits, for one token (transformer_forward) and for a batch of rows (speculative and
 * multi-sequence steps). Split out of forward.c; the dispatch is unchanged.
 */

#include "transformer/forward_head.h"

#include <stdint.h>

#include "core/gguf_reader.h"
#include "core/step_timing.h"
#include "math/batched_matmul.h"
#include "math/matmul_q2_0.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q6k.h"
#include "math/parallel_matmul.h"
#include "math/simd_core_ops.h"
#include "runtime/hardware_profile.h"
#include "runtime/spec_scratch.h"

/* Which form of the LM head can run: a zero-copy quantized head of this type, or a materialized
 * INT4 / INT8 copy when the selected classifier format asks for one. */
static int cls_raw_is(const TransformerWeights *w, int type) {
    return w->cls.raw && w->cls.raw_type == type;
}
static int cls_i4_ok(const TransformerWeights *w, TnClassifierFormat fmt) {
    return fmt == TN_CLS_INT4 && w->cls.i4 && w->cls.i4_scales;
}
static int cls_i8_ok(const TransformerWeights *w, TnClassifierFormat fmt) {
    return fmt >= TN_CLS_INT8 && w->cls.i8 && w->cls.i8_scales;
}

/* Classifier of a Q2_0 model. Uses a materialized BF16/INT8/INT4 classifier copy when one was
 * built (only happens when the user explicitly passed --classifier — see gguf_loader.c) instead of
 * always falling back to the raw zero-copy Q2_0 path regardless of the requested format (the bug
 * this replaces; see docs/ai/mistakes.md). Absent an explicit request, stay zero-copy. */
static int classify_materialized(const Config *cfg, const TransformerWeights *w, RunState *s,
                                 ThreadPool *tp, TnClassifierFormat fmt) {
    int dim = cfg->dim;
    if (cls_i4_ok(w, fmt)) {
        parallel_matmul_i4(s->logits, s->x, w->cls.i4, w->cls.i4_scales, dim, cfg->vocab_size, tp);
    } else if (cls_i8_ok(w, fmt)) {
        parallel_matmul_i8(s->logits, s->x, w->cls.i8, w->cls.i8_scales, dim, cfg->vocab_size, tp);
    } else if (w->cls.w) {
        parallel_matmul_bf16(s->logits, s->x, w->cls.w, dim, cfg->vocab_size, tp);
    } else {
        return 0;
    }
    return 1;
}

static void classify_q2_0(const Config *cfg, const TransformerWeights *w, RunState *s,
                          ThreadPool *tp) {
    const TnHardwareProfile *hp = tn_hardware_profile_get();
    TnClassifierFormat fmt      = hp ? hp->classifier_fmt : TN_CLS_BF16;
    if (hp && hp->classifier_explicit && classify_materialized(cfg, w, s, tp, fmt)) return;
    parallel_matmul_q2_0(s->logits, s->x, (const uint8_t *)w->q35.output_raw, cfg->dim,
                         cfg->vocab_size, tp);
}

/* Hardware-aware dispatch via TnHardwareProfile:
 *   - INT4 when VNNI available (0.5 bytes/weight, 164 MB bandwidth)
 *   - INT8 when VNNI or ARM dotprod available (1 byte, 328 MB)
 *   - BF16 fallback for all other hardware (2 bytes, 656 MB)
 * The hardware profile auto-selects the optimal format at startup. Weights are only quantized to
 * the selected format (saves startup time). Quality preserved: the LM head only needs relative
 * ordering for sampling. */
static void classify_generic(const Config *cfg, const TransformerWeights *w, RunState *s,
                             ThreadPool *tp) {
    int dim                     = cfg->dim;
    const TnHardwareProfile *hp = tn_hardware_profile_get();
    TnClassifierFormat fmt      = hp ? hp->classifier_fmt : TN_CLS_BF16;
    if (cls_raw_is(w, GGUF_TYPE_Q6_K)) { /* zero-copy quantized head (2026-10-21, see weights.h) */
        parallel_matmul_q6k(s->logits, s->x, (const uint8_t *)w->cls.raw, dim, cfg->vocab_size, tp);
        return;
    }
    if (cls_raw_is(w, GGUF_TYPE_Q4_K)) {
        parallel_matmul_q4k(s->logits, s->x, (const uint8_t *)w->cls.raw, dim, cfg->vocab_size, tp);
        return;
    }
    if (cls_i4_ok(w, fmt)) {
        parallel_matmul_i4(s->logits, s->x, w->cls.i4, w->cls.i4_scales, dim, cfg->vocab_size, tp);
        return;
    }
    if (cls_i8_ok(w, fmt)) {
        parallel_matmul_i8(s->logits, s->x, w->cls.i8, w->cls.i8_scales, dim, cfg->vocab_size, tp);
        return;
    }
    parallel_matmul_bf16(s->logits, s->x, w->cls.w, dim, cfg->vocab_size, tp);
}

/* Step 4: classifier matmul -> logits (Q2_0, ternary or the generic dispatch above). */
void forward_head_classify(const Config *cfg, const TransformerWeights *w, RunState *s,
                           ThreadPool *tp) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->q35.is_q2_0_model) {
        classify_q2_0(cfg, w, s, tp);
    } else if (w->cls.is_ternary) {
        parallel_ternary_matmul_packed(s->logits, s->x, (const tn_u8 *)w->cls.w, cfg->dim,
                                       cfg->vocab_size, w->cls.scale, tp);
    } else {
        classify_generic(cfg, w, s, tp);
    }
    if (t_step) {
        tn_step_timing_add(TN_STEP_19_LM_HEAD, tn_step_timing_now_ns() - t_step);
    }
}

/* Final RMSNorm, per token (a per-row statistic, not batchable). */
static void rows_final_norm(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                            int n_tokens) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    if (w->rms_final_weight) {
        for (int k = 0; k < n_tokens; k++) {
            float *x_k = sb->x + (size_t)k * cfg->dim;
            tn_rmsnorm(x_k, x_k, w->rms_final_weight, cfg->dim, cfg->rms_norm_eps);
        }
    }
    if (t_step) tn_step_timing_add(TN_STEP_18_FINAL_RMSNORM, tn_step_timing_now_ns() - t_step);
}

/* Classifier matmul -> logits_out, batched. Mirrors transformer_forward()'s non-q35 branches
 * (wcls_is_ternary, then the hardware-profile-selected INT4/INT8/BF16 generic path); the
 * q35_is_q2_0_model branch there is unreachable here (see Step 1). */
static void rows_classifier(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                            int n_tokens, ThreadPool *tp, float *logits_out) {
    int dim = cfg->dim;
    if (w->cls.is_ternary) {
        tn_ternary_matmul_packed_batch(logits_out, sb->x, (const tn_u8 *)w->cls.w, dim,
                                       cfg->vocab_size, w->cls.scale, n_tokens, tp);
        return;
    }
    const TnHardwareProfile *hp = tn_hardware_profile_get();
    TnClassifierFormat fmt      = hp ? hp->classifier_fmt : TN_CLS_BF16;
    if (cls_raw_is(w, GGUF_TYPE_Q6_K)) {
        parallel_matmul_q6k_tokens(logits_out, sb->x, (const uint8_t *)w->cls.raw, dim,
                                   cfg->vocab_size, n_tokens, tp);
        return;
    }
    if (cls_raw_is(w, GGUF_TYPE_Q4_K)) {
        parallel_matmul_q4k_tokens(logits_out, sb->x, (const uint8_t *)w->cls.raw, dim,
                                   cfg->vocab_size, n_tokens, tp);
        return;
    }
    if (cls_i4_ok(w, fmt)) {
        tn_matmul_i4_batch(logits_out, sb->x, w->cls.i4, w->cls.i4_scales, dim, cfg->vocab_size,
                           n_tokens, tp);
        return;
    }
    if (cls_i8_ok(w, fmt)) {
        tn_matmul_i8_batch(logits_out, sb->x, w->cls.i8, w->cls.i8_scales, dim, cfg->vocab_size,
                           n_tokens, tp);
        return;
    }
    tn_matmul_bf16_batch(logits_out, sb->x, w->cls.w, dim, cfg->vocab_size, n_tokens, tp);
}

/* Final RMSNorm + classifier over n_tokens rows of sb->x -> logits_out
 * ([n_tokens][vocab]); shared by the speculative batch and multi-sequence
 * steps. */
void forward_head_rows(SpecBatchScratch *sb, const TransformerWeights *w, const Config *cfg,
                       int n_tokens, ThreadPool *tp, float *logits_out) {
    rows_final_norm(sb, w, cfg, n_tokens);
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    rows_classifier(sb, w, cfg, n_tokens, tp, logits_out);
    if (t_step) tn_step_timing_add(TN_STEP_19_LM_HEAD, tn_step_timing_now_ns() - t_step);
}
