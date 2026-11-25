#include "runtime/model_load.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/log.h"
#include "core/moe_weights.h"
#include "core/progress.h"
#include "kv_cache/kv_strategy.h"
#include "runtime/hardware_profile.h"
#include "runtime/mamba_state.h"

/* Phase 18 (speculative decoding) Stage 3: this file is the GGUF half of
 * main()'s original inline model-load block (roughly its old lines
 * 199-529), factored out unchanged in behavior so a second (draft) model
 * can reuse it. main()'s native raw-ternary `.bin` branch is NOT handled
 * here — draft models are GGUF-only (see include/runtime/model_load.h) — and
 * stays fully inline in main() as its own separate path. */

/* How far a load got; release_to() frees everything up to and including that stage. */
typedef enum { STAGE_FILE, STAGE_WEIGHTS, STAGE_STATE } LoadStage;

/* Frees what a (partial) load built, newest first. Every free below is NULL-safe and the model
 * starts zeroed, so it is correct at any stage. mla_run_state_free / q35_run_state_free /
 * qwen3moe_run_state_free must run before run_state_free() (they free k_rope_cache and the q35_ /
 * qwen3moe_ pointer arrays that run_state_free() doesn't know about) -- mirrors main()'s own
 * end-of-program cleanup sequence. */
static void release_state(LoadedModel *m) {
    if (!m->state) return;
    if (m->moe_config.mla.enabled) mla_run_state_free(m->state, m->config.n_layers);
    if (m->moe_config.has_linear_attn) q35_run_state_free(m->state, &m->config, &m->moe_config);
    if (m->moe_config.has_qk_norm) qwen3moe_run_state_free(m->state, &m->config);
    run_state_free(m->state);
    free(m->state);
    m->state = NULL;
}

static void release_to(LoadedModel *m, LoadStage stage) {
    if (stage >= STAGE_STATE) release_state(m);
    if (stage >= STAGE_WEIGHTS) {
        if (m->moe_config.is_moe) moe_weights_free(&m->weights, &m->moe_config);
        weights_free_pointers(&m->weights);
        if (m->gguf_store) weights_free_gguf(m->gguf_store);
        m->gguf_store = NULL;
    }
    gguf_header_free(&m->gguf_hdr);
    mapped_file_close(&m->mf);
}

/* Maps the file, reads the GGUF header, the Config and the MoEConfig. */
static TernaryError read_model_info(LoadedModel *out, const char *model_path, bool is_primary) {
    if (mapped_file_open(&out->mf, model_path) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to map model file: %s\n", model_path);
        return TN_ERR_FILE_OPEN;
    }

    if (is_primary) TN_OUT_INF("model_load", "Model format: GGUF\n");

    if (gguf_read_header(&out->gguf_hdr, out->mf.data, out->mf.size) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to parse GGUF header.\n");
        return TN_ERR_INVALID_ARGS;
    }
    if (config_from_gguf(&out->config, &out->gguf_hdr) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to read config from GGUF metadata.\n");
        return TN_ERR_INVALID_ARGS;
    }
    if (is_primary) config_print(&out->config);

    /* Populate MoEConfig (MoE/MLA/hybrid/SSM flags) ahead of weight loading.
     * moe_config_from_gguf() returns the dense default for plain
     * architectures, so it is called for every model. 2026-10-18: this used
     * to be gated on a hardcoded arch list (deepseek2/qwen35/qwen3moe), so a
     * newly supported architecture (mamba) silently ran as a dense
     * transformer with NULL attention weights. */
    if (moe_config_from_gguf(&out->moe_config, &out->gguf_hdr) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to read MoE config from GGUF.\n");
        return TN_ERR_INVALID_ARGS;
    }
    if (is_primary &&
        (out->moe_config.is_moe || out->moe_config.mla.enabled || out->moe_config.has_linear_attn ||
         out->moe_config.has_qk_norm || out->moe_config.is_ssm))
        moe_config_print(&out->moe_config);
    return TN_OK;
}

/* Weight pointer arrays (per layer, per expert for MoE) and the weights themselves. On failure
 * `*stage` says what to release. */
static TernaryError load_model_weights(LoadedModel *out, LoadStage *stage) {
    *stage = STAGE_FILE;
    if (weights_alloc_pointers(&out->weights, &out->config) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to allocate weight pointers.\n");
        return TN_ERR_OOM;
    }
    *stage = STAGE_WEIGHTS;
    /* For MoE GGUF models (DeepSeek-V2), allocate per-layer/expert arrays. */
    if (out->moe_config.is_moe &&
        moe_weights_alloc(&out->weights, &out->config, &out->moe_config) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to allocate MoE weight pointer arrays.\n");
        return TN_ERR_OOM;
    }
    if (weights_from_gguf(&out->weights, &out->config, &out->gguf_hdr, &out->gguf_store) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to load GGUF weights.\n");
        return TN_ERR_INVALID_ARGS;
    }
    return TN_OK;
}

/* Correct the profiler's per-token traffic + ceiling with the real
 * loaded model (2026-07-17): tn_hardware_profile_init() runs before
 * the model file is opened and seeds Data/token with compile-time
 * BitNet-2B constants (~1149 MB), overstating the ceiling ~6x for
 * multi-GB GGUF models (docs/ai/mistakes.md). Accounting per class:
 * Q2_0-native models subtract the embedding table (read one row per
 * token, not streamed) and swap the raw Q2_0 LM head for the
 * materialized classifier's bytes when one was explicitly requested.
 * Generic GGUF uses the raw file size — an upper bound that still
 * bills the full embedding table, since its tensor format (and thus
 * byte size) isn't known generically here; tight for tied-embedding
 * models, conservative otherwise. MoE models overcount (all experts,
 * not just routed) — TODO(PZ-TODO-1): subtract inactive-expert bytes via
 * MoEConfig.
 *
 * Only ever measured for the primary (verifier) model — a draft model
 * loaded alongside it must not overwrite the profiler's ceiling display
 * with its own (much smaller) bytes/token. */

static void report_model_bytes(const LoadedModel *out) {
    double per_tok                = (double)out->mf.size;
    double cls_bytes              = 0.0;
    const TnHardwareProfile *hp_m = tn_hardware_profile_get();
    if (out->weights.q35.is_q2_0_model) {
        double q2_row = (double)out->config.vocab_size * out->config.dim * (34.0 / 128.0);
        cls_bytes     = q2_row; /* zero-copy raw Q2_0 head (default) */
        if (hp_m && hp_m->classifier_explicit) {
            switch (hp_m->classifier_fmt) {
                case TN_CLS_INT4:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim * 0.5;
                    break;
                case TN_CLS_INT8:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim;
                    break;
                default:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim * 2.0;
                    break;
            }
        }
        /* drop embedding (one row/token) + in-file head, add the
         * head actually used */
        per_tok -= 2.0 * q2_row;
        per_tok += cls_bytes;
    }
    tn_hardware_profile_set_model_bytes(per_tok, cls_bytes);
    if (hp_m) {
        TN_OUT_INF("profile",
                   "[profile] Data/token (loaded model): %.0f MB -> "
                   "ceiling %.1f tok/s at %.1f GB/s\n",
                   per_tok / (1024.0 * 1024.0), hp_m->theoretical_ceiling, hp_m->measured_bw_gbps);
    }
}

/* The context length: sized from the RAM left after loading for the primary model, or the hint for
 * a draft model. */
static void size_context(LoadedModel *out, bool is_primary, int max_seq_len_hint,
                         const KVStrategyOptions *kv_opt) {
    /* KV Strategy — measured AFTER model weights are loaded so that any F32-dequantised
 * weight allocations (MLA projections, shared experts, norms) are already counted in
 * consumed RAM.  For mmap'd GGUF models the raw quantised expert bytes don't use
 * physical RAM until accessed, so only the upfront malloc'd F32 blocks matter here.
 * Re-measuring at this point prevents DeepSeek-style OOM where a 163840-token context
 * would require 7+ GB KV cache on a machine that only has 2–3 GB left after load.
 *
 * A non-primary (draft) model skips the free-RAM probe entirely and takes
 * max_seq_len_hint directly — a draft model never needs a large context,
 * and a second independent free-RAM probe during the same startup would
 * double-count RAM the primary model's own sizing already accounted for. */
    if (is_primary) {
        tn_i64 post_load_ram = tn_get_free_ram();
        KVStrategyResult kv_res =
            select_kv_strategy_ex(&out->config, post_load_ram, &out->moe_config, kv_opt);
        out->config.seq_len = kv_res.max_seq_len;
        /* Qwen35 hybrid models keep their own F32 K/V caches
         * (q35.key_cache/q35.value_cache, only the full-attention layers) —
         * the quantized-KV strategy machinery is not wired into that path,
         * so printing e.g. "Quantized I8" for them misreported what actually
         * happens (2026-07-17, found during the ceiling-gap attribution;
         * see docs/ai/mistakes.md). The RAM-aware max-context clamp from
         * select_kv_strategy() still applies either way. Qwen3-MoE's own
         * qwen3moe.key_cache/qwen3moe.value_cache are the same situation —
         * independent head_dim, own F32 cache, not the quantized-KV path. */
        kv_strategy_print(&kv_res, out->config.seq_len, kv_opt);
    } else {
        out->config.seq_len = (max_seq_len_hint > 0) ? max_seq_len_hint : 1;
    }
}

/* RunState and the per-architecture state (MLA, Mamba, Qwen3.5 hybrid, Qwen3-MoE). On failure the
 * state is released here; the caller releases the weights. */
static TernaryError alloc_run_state(LoadedModel *out, const KVStrategyOptions *kv_opt) {
    Config *c     = &out->config;
    MoEConfig *mc = &out->moe_config;
    out->state    = (RunState *)malloc(sizeof(RunState));
    if (!out->state) {
        TN_LOG_ERR("model_load", "Failed to allocate RunState header.\n");
        return TN_ERR_OOM;
    }

    /* Qwen3.5/3.6 hybrid models and Qwen3-MoE models never read the generic
     * key_cache/value_cache (they use their own correctly-sized
     * q35.key_cache/q35.value_cache or qwen3moe.key_cache/
     * qwen3moe.value_cache, allocated below) — skip_kv_cache=true avoids a
     * multi-GB calloc+memset for those models' context. See
     * run_state_alloc_ex's header comment and docs/ai/mistakes.md. */
    if (run_state_alloc_ex(out->state, c, c->seq_len,
                           mc->has_linear_attn || mc->has_qk_norm || mc->is_ssm, kv_opt) != TN_OK) {
        TN_LOG_ERR("model_load", "Failed to allocate RunState buffers.\n");
        free(out->state); /* nothing inside it to release */
        out->state = NULL;
        return TN_ERR_OOM;
    }

    const char *what = NULL;
    if (mc->mla.enabled &&
        mla_run_state_alloc(out->state, c, mc, c->seq_len) != TN_OK) /* k_rope_cache */
        what = "MLA k_rope_cache";
    else if (mc->is_ssm && mamba_state_alloc(&out->state->mamba, c->n_layers, mc) != TN_OK)
        what = "Mamba SSM state"; /* freed by run_state_free */
    else if (mc->has_linear_attn && q35_run_state_alloc(out->state, c, mc, c->seq_len) != TN_OK)
        what = "Qwen3.5/3.6 hybrid attention state";
    else if (mc->has_qk_norm && qwen3moe_run_state_alloc(out->state, c, mc, c->seq_len) != TN_OK)
        what = "Qwen3-MoE attention state";
    if (what) {
        TN_LOG_ERR("model_load", "Failed to allocate %s.\n", what);
        release_state(out);
        return TN_ERR_OOM;
    }
    return TN_OK;
}

TernaryError load_gguf_model(LoadedModel *out, const char *model_path, bool is_primary,
                             int max_seq_len_hint, ThreadPool *tp, bool stdout_is_tty,
                             const KVStrategyOptions *kv_opt) {
    memset(out, 0, sizeof(*out));
    moe_config_init_dense(&out->moe_config); /* dense by default; overridden below for MoE models */
    (void)tp; /* not currently consumed by weights_from_gguf() (which takes no ThreadPool
               * argument) -- kept in the signature for interface symmetry with the rest
               * of this codebase's loader-style functions and in case a future weight
               * format needs it; not a dead parameter to remove. */

    TernaryError e = read_model_info(out, model_path, is_primary);
    if (e != TN_OK) {
        release_to(out, STAGE_FILE);
        return e;
    }
    LoadStage stage;
    e = load_model_weights(out, &stage);
    if (e != TN_OK) {
        release_to(out, stage);
        return e;
    }
    if (is_primary) {
        report_model_bytes(out);
        tn_progress_stage(3, 4, "Preparing runtime...", stdout_is_tty);
    }
    size_context(out, is_primary, max_seq_len_hint, kv_opt);
    e = alloc_run_state(out, kv_opt);
    if (e != TN_OK) {
        release_to(out, STAGE_WEIGHTS);
        return e;
    }
    return TN_OK;
}

void loaded_model_free(LoadedModel *m) {
    if (!m->state) {
        /* Nothing beyond this was ever allocated (either load_gguf_model()
         * never returned TN_OK, or this is a zero-initialized LoadedModel
         * the caller never attempted to load into) -- but the mmap may
         * still be open if a caller manually populated `mf` without going
         * through load_gguf_model(). mapped_file_close() is itself safe on
         * a zeroed MappedFile (see its own doc comment / 2026-09-22 fix). */
        mapped_file_close(&m->mf);
        return;
    }
    release_to(m, STAGE_STATE);
}
