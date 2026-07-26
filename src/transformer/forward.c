#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "transformer/forward.h"
#include "core/moe_config.h"
#include "core/debug.h"
#include "core/step_timing.h"
#include "math/parallel_matmul.h"
#include "math/matmul_q2_0.h"
#include "math/simd_dispatch.h"
#include "transformer/attention.h"
#include "transformer/embedding.h"
#include "transformer/ffn.h"
#include "core/hardware_profile.h"

/*
 * Lightweight per-component profiling, activated by TN_PROFILE=1.
 * Accumulates time across all tokens, prints summary on every Nth token.
 * Zero cost when disabled (checked once at first call).
 */
static int g_profile_enabled = -1; /* -1 = unchecked */
static int64_t g_prof_embed_ns, g_prof_attn_ns;
static int64_t g_prof_norm_ns, g_prof_cls_ns, g_prof_total_ns;
static int64_t g_prof_attn_only_ns, g_prof_ffn_only_ns;
static int g_prof_tokens;

static inline int64_t prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

float *transformer_forward(int token, int pos, const Config *cfg,
                           const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, ThreadPool *tp) {
  int dim = cfg->dim;

  /* One-time check for profiling env var */
  if (g_profile_enabled < 0) {
      const char *env = getenv("TN_PROFILE");
      g_profile_enabled = (env && env[0] == '1') ? 1 : 0;
  }

  int64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
  if (g_profile_enabled) t0 = prof_now();

    /* Step 1: Embed token → s->x (skip if token < 0, assuming caller injected s->x) */
    if (token >= 0) {
      if (token >= cfg->vocab_size) {
          return s->logits;
      }
      int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
      if (w->q35_is_q2_0_model) {
        embed_token_q2_0(s->x, token, w->q35_token_embd_raw, dim);
      } else {
        embed_token(s->x, token, w->embd_f32, w->token_embedding_table, dim);
      }
      if (t_step) {
          tn_step_timing_add(TN_STEP_3_TOKEN_EMBEDDING,
                             tn_step_timing_now_ns() - t_step);
      }
      DBG_MSG("token=%d  pos=%d\n", token, pos);
    }

  if (g_profile_enabled) t1 = prof_now();

    /* Step 2: Run through all transformer layers */
    for (int l = 0; l < cfg->n_layers; l++) {
      int64_t la = 0, lb = 0;
      if (g_profile_enabled) la = prof_now();
      attention_forward(s, w, cfg, mc, l, pos, tp);
      if (g_profile_enabled) lb = prof_now();
      ffn_forward(s, w, cfg, mc, l, tp);
      if (g_profile_enabled) { int64_t lc = prof_now(); g_prof_attn_only_ns += (lb - la); g_prof_ffn_only_ns += (lc - lb); }
      DBG_DUMP(l, "l_out", s->x, dim);
      if (g_tn_verbose) {
          char tag[48];
          snprintf(tag, sizeof(tag), "L%02d post-layer x[dim=%d]", l, dim);
          dbg_vec_stats(tag, s->x, dim);
      }
    }
    sw_advance(&s->sw);

  if (g_profile_enabled) t2 = prof_now();

    /* Step 3: Final RMSNorm */
    if (w->rms_final_weight) {
      int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
      tn_rmsnorm(s->x, s->x, w->rms_final_weight, dim, cfg->rms_norm_eps);
      if (t_step) {
          tn_step_timing_add(TN_STEP_18_FINAL_RMSNORM,
                             tn_step_timing_now_ns() - t_step);
      }
    }
    DBG_DUMP(-1, "result_norm", s->x, dim);

  if (g_profile_enabled) t3 = prof_now();

  /* Verbose: logits stats before classifier */
  DBG_VEC("final-norm x", s->x, dim);

  /* Step 4: Classifier matmul → logits
   *
   * Hardware-aware dispatch via TnHardwareProfile:
   *   - INT4 when VNNI available (0.5 bytes/weight, 164 MB bandwidth)
   *   - INT8 when VNNI or ARM dotprod available (1 byte, 328 MB)
   *   - BF16 fallback for all other hardware (2 bytes, 656 MB)
   *
   * The hardware profile auto-selects the optimal format at startup.
   * Weights are only quantized to the selected format (saves startup time).
   * Quality preserved: LM head only needs relative ordering for sampling.
   */
  if (w->q35_is_q2_0_model) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    /* Use a materialized BF16/INT8/INT4 classifier copy when one was built
     * (only happens when the user explicitly passed --classifier — see
     * gguf_loader.c) instead of always falling back to the raw zero-copy
     * Q2_0 path regardless of the requested format (the bug this replaces;
     * see docs/ai/mistakes.md). Absent an explicit request, stay zero-copy. */
    const TnHardwareProfile *hp_q2 = tn_hardware_profile_get();
    TnClassifierFormat fmt_q2 = hp_q2 ? hp_q2->classifier_fmt : TN_CLS_BF16;
    bool used_materialized = false;
    if (hp_q2 && hp_q2->classifier_explicit) {
      if (fmt_q2 == TN_CLS_INT4 && w->wcls_i4 && w->wcls_i4_scales) {
        parallel_matmul_i4(s->logits, s->x, w->wcls_i4, w->wcls_i4_scales,
                            dim, cfg->vocab_size, tp);
        used_materialized = true;
      } else if (fmt_q2 >= TN_CLS_INT8 && w->wcls_i8 && w->wcls_i8_scales) {
        parallel_matmul_i8(s->logits, s->x, w->wcls_i8, w->wcls_i8_scales,
                            dim, cfg->vocab_size, tp);
        used_materialized = true;
      } else if (w->wcls) {
        parallel_matmul_bf16(s->logits, s->x, w->wcls, dim, cfg->vocab_size, tp);
        used_materialized = true;
      }
    }
    if (!used_materialized) {
      parallel_matmul_q2_0(s->logits, s->x, (const uint8_t *)w->q35_output_raw,
                            dim, cfg->vocab_size, tp);
    }
    if (t_step) {
      tn_step_timing_add(TN_STEP_19_LM_HEAD, tn_step_timing_now_ns() - t_step);
    }
  } else if (w->wcls_is_ternary) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    parallel_ternary_matmul_packed(s->logits, s->x, (const tn_u8 *)w->wcls, dim, cfg->vocab_size,
                            w->wcls_scale, tp);
    if (t_step) {
      tn_step_timing_add(TN_STEP_19_LM_HEAD, tn_step_timing_now_ns() - t_step);
    }
  } else {
    const TnHardwareProfile *hp = tn_hardware_profile_get();
    TnClassifierFormat fmt = hp ? hp->classifier_fmt : TN_CLS_BF16;
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;

    if (fmt == TN_CLS_INT4 && w->wcls_i4 && w->wcls_i4_scales) {
      parallel_matmul_i4(s->logits, s->x, w->wcls_i4, w->wcls_i4_scales,
                          dim, cfg->vocab_size, tp);
    } else if (fmt >= TN_CLS_INT8 && w->wcls_i8 && w->wcls_i8_scales) {
      parallel_matmul_i8(s->logits, s->x, w->wcls_i8, w->wcls_i8_scales,
                          dim, cfg->vocab_size, tp);
    } else {
      parallel_matmul_bf16(s->logits, s->x, w->wcls, dim, cfg->vocab_size, tp);
    }
    if (t_step) {
      tn_step_timing_add(TN_STEP_19_LM_HEAD, tn_step_timing_now_ns() - t_step);
    }
  }

  /* Step 19: dump top-8 logit values + their token indices for LM head verification */
  if (g_dump_fp) {
      /* Find top-8 logit indices by partial selection sort */
      float top_vals[8];
      int   top_idx[8];
      int   visited[8] = {0,0,0,0,0,0,0,0};
      /* store top-8 by scanning vocab once per pick (8 passes, each O(vocab)) */
      int vocab = cfg->vocab_size;
      for (int k = 0; k < 8; k++) {
          float best = -1e30f; int bi = 0;
          for (int i = 0; i < vocab; i++) {
              int already = 0;
              for (int j = 0; j < k; j++) if (top_idx[j] == i) { already=1; break; }
              if (!already && s->logits[i] > best) { best = s->logits[i]; bi = i; }
          }
          top_vals[k] = best; top_idx[k] = bi;
          (void)visited;
      }
      /* Dump as a synthetic 8-element vector tagged "lm_head_top8" */
      DBG_DUMP(-1, "lm_head_top8", top_vals, 8);
      /* Also dump the corresponding token indices as floats for reference */
      float idx_f[8];
      for (int k = 0; k < 8; k++) idx_f[k] = (float)top_idx[k];
      DBG_DUMP(-1, "lm_head_top8_idx", idx_f, 8);
  }

  if (g_profile_enabled) {
      t4 = prof_now();
      g_prof_embed_ns += (t1 - t0);
      g_prof_attn_ns  += (t2 - t1);  /* attn + ffn combined (layers) */
      g_prof_norm_ns  += (t3 - t2);
      g_prof_cls_ns   += (t4 - t3);
      g_prof_total_ns += (t4 - t0);
      g_prof_tokens++;

      if (g_prof_tokens % 20 == 0) {
          double ms = 1e-6;
          fprintf(stderr, "[PROFILE %d tok] embed=%.2f ms  layers=%.2f ms "
                  "(attn=%.2f ffn=%.2f)  "
                  "norm=%.2f ms  classifier=%.2f ms  TOTAL=%.2f ms\n",
                  g_prof_tokens,
                  g_prof_embed_ns * ms / g_prof_tokens,
                  g_prof_attn_ns  * ms / g_prof_tokens,
                  g_prof_attn_only_ns * ms / g_prof_tokens,
                  g_prof_ffn_only_ns * ms / g_prof_tokens,
                  g_prof_norm_ns  * ms / g_prof_tokens,
                  g_prof_cls_ns   * ms / g_prof_tokens,
                  g_prof_total_ns * ms / g_prof_tokens);
      }
  }

  return s->logits;
}
