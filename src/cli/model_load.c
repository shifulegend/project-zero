#include "cli/model_load.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/hardware_profile.h"
#include "core/moe_weights.h"
#include "cli/progress.h"
#include "kv_cache/kv_strategy.h"

/* Phase 18 (speculative decoding) Stage 3: this file is the GGUF half of
 * main()'s original inline model-load block (roughly its old lines
 * 199-529), factored out unchanged in behavior so a second (draft) model
 * can reuse it. main()'s native raw-ternary `.bin` branch is NOT handled
 * here — draft models are GGUF-only (see include/cli/model_load.h) — and
 * stays fully inline in main() as its own separate path. */

TernaryError load_gguf_model(LoadedModel *out, const char *model_path,
                              bool is_primary, int max_seq_len_hint,
                              ThreadPool *tp, bool stdout_is_tty) {
    memset(out, 0, sizeof(*out));
    moe_config_init_dense(&out->moe_config); /* dense by default; overridden below for MoE models */
    (void)tp; /* not currently consumed by weights_from_gguf() (which takes no ThreadPool
               * argument) -- kept in the signature for interface symmetry with the rest
               * of this codebase's loader-style functions and in case a future weight
               * format needs it; not a dead parameter to remove. */

    if (mapped_file_open(&out->mf, model_path) != TN_OK) {
        fprintf(stderr, "Failed to map model file: %s\n", model_path);
        return TN_ERR_FILE_OPEN;
    }

    if (is_primary) printf("Model format: GGUF\n");

    if (gguf_read_header(&out->gguf_hdr, out->mf.data, out->mf.size) != TN_OK) {
        fprintf(stderr, "Failed to parse GGUF header.\n");
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_INVALID_ARGS;
    }
    if (config_from_gguf(&out->config, &out->gguf_hdr) != TN_OK) {
        fprintf(stderr, "Failed to read config from GGUF metadata.\n");
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_INVALID_ARGS;
    }
    if (is_primary) config_print(&out->config);

    /* For DeepSeek-V2 (MLA+MoE), Qwen3.5/3.6 (hybrid Gated-DeltaNet), or
     * Qwen3-MoE (routed MoE + QK-norm) GGUF: populate MoEConfig ahead of
     * weight loading. */
    if (strcmp(out->gguf_hdr.arch, "deepseek2") == 0 || strcmp(out->gguf_hdr.arch, "qwen35") == 0 ||
        strcmp(out->gguf_hdr.arch, "qwen3moe") == 0) {
        if (moe_config_from_gguf(&out->moe_config, &out->gguf_hdr) != TN_OK) {
            fprintf(stderr, "Failed to read MoE config from GGUF.\n");
            gguf_header_free(&out->gguf_hdr);
            mapped_file_close(&out->mf);
            return TN_ERR_INVALID_ARGS;
        }
        if (is_primary) moe_config_print(&out->moe_config);
    }

    if (weights_alloc_pointers(&out->weights, &out->config) != TN_OK) {
        fprintf(stderr, "Failed to allocate weight pointers.\n");
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_OOM;
    }
    /* For MoE GGUF models (DeepSeek-V2), allocate per-layer/expert arrays. */
    if (out->moe_config.is_moe) {
        if (moe_weights_alloc(&out->weights, &out->config, &out->moe_config) != TN_OK) {
            fprintf(stderr, "Failed to allocate MoE weight pointer arrays.\n");
            moe_weights_free(&out->weights, &out->moe_config);
            weights_free_pointers(&out->weights);
            gguf_header_free(&out->gguf_hdr);
            mapped_file_close(&out->mf);
            return TN_ERR_OOM;
        }
    }
    if (weights_from_gguf(&out->weights, &out->config, &out->gguf_hdr, &out->gguf_store) != TN_OK) {
        fprintf(stderr, "Failed to load GGUF weights.\n");
        if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
        weights_free_pointers(&out->weights);
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_INVALID_ARGS;
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
     * not just routed) — TODO: subtract inactive-expert bytes via
     * MoEConfig.
     *
     * Only ever measured for the primary (verifier) model — a draft model
     * loaded alongside it must not overwrite the profiler's ceiling display
     * with its own (much smaller) bytes/token. */
    if (is_primary) {
        double per_tok = (double)out->mf.size;
        double cls_bytes = 0.0;
        const TnHardwareProfile *hp_m = tn_hardware_profile_get();
        if (out->weights.q35_is_q2_0_model) {
            double q2_row = (double)out->config.vocab_size * out->config.dim * (34.0 / 128.0);
            cls_bytes = q2_row; /* zero-copy raw Q2_0 head (default) */
            if (hp_m && hp_m->classifier_explicit) {
                switch (hp_m->classifier_fmt) {
                case TN_CLS_INT4:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim * 0.5; break;
                case TN_CLS_INT8:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim;       break;
                default:
                    cls_bytes = (double)out->config.vocab_size * out->config.dim * 2.0; break;
                }
            }
            /* drop embedding (one row/token) + in-file head, add the
             * head actually used */
            per_tok -= 2.0 * q2_row;
            per_tok += cls_bytes;
        }
        tn_hardware_profile_set_model_bytes(per_tok, cls_bytes);
        if (hp_m) {
            printf("[profile] Data/token (loaded model): %.0f MB -> "
                   "ceiling %.1f tok/s at %.1f GB/s\n",
                   per_tok / (1024.0 * 1024.0),
                   hp_m->theoretical_ceiling, hp_m->measured_bw_gbps);
        }
    }

    if (is_primary) tn_progress_stage(3, 4, "Preparing runtime...", stdout_is_tty);

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
        KVStrategyResult kv_res = select_kv_strategy(&out->config, post_load_ram, &out->moe_config);
        out->config.seq_len = kv_res.max_seq_len;
        /* Qwen35 hybrid models keep their own F32 K/V caches
         * (q35_key_cache/q35_value_cache, only the full-attention layers) —
         * the quantized-KV strategy machinery is not wired into that path,
         * so printing e.g. "Quantized I8" for them misreported what actually
         * happens (2026-07-17, found during the ceiling-gap attribution;
         * see docs/ai/mistakes.md). The RAM-aware max-context clamp from
         * select_kv_strategy() still applies either way. Qwen3-MoE's own
         * qwen3moe_key_cache/qwen3moe_value_cache are the same situation —
         * independent head_dim, own F32 cache, not the quantized-KV path. */
        if (out->moe_config.has_linear_attn) {
            printf("KV Strategy: F32 (Qwen35 hybrid path; quantized-KV "
                   "strategy not wired in), max context: %d tokens\n",
                   out->config.seq_len);
        } else if (out->moe_config.has_qk_norm) {
            printf("KV Strategy: F32 (Qwen3-MoE path; quantized-KV "
                   "strategy not wired in), max context: %d tokens\n",
                   out->config.seq_len);
        } else {
            printf("KV Strategy: %s, max context: %d tokens\n",
                   kv_strategy_name(kv_res.strategy), out->config.seq_len);
        }
    } else {
        out->config.seq_len = (max_seq_len_hint > 0) ? max_seq_len_hint : 1;
    }

    /* Setup RunState */
    out->state = (RunState *)malloc(sizeof(RunState));
    if (!out->state) {
        fprintf(stderr, "Failed to allocate RunState header.\n");
        if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
        weights_free_pointers(&out->weights);
        if (out->gguf_store) weights_free_gguf(out->gguf_store);
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_OOM;
    }

    /* Qwen3.5/3.6 hybrid models and Qwen3-MoE models never read the generic
     * key_cache/value_cache (they use their own correctly-sized
     * q35_key_cache/q35_value_cache or qwen3moe_key_cache/
     * qwen3moe_value_cache, allocated below) — skip_kv_cache=true avoids a
     * multi-GB calloc+memset for those models' context. See
     * run_state_alloc_ex's header comment and docs/ai/mistakes.md. */
    if (run_state_alloc_ex(out->state, &out->config, out->config.seq_len,
                            out->moe_config.has_linear_attn || out->moe_config.has_qk_norm) != TN_OK) {
        fprintf(stderr, "Failed to allocate RunState buffers.\n");
        free(out->state);
        out->state = NULL;
        if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
        weights_free_pointers(&out->weights);
        if (out->gguf_store) weights_free_gguf(out->gguf_store);
        gguf_header_free(&out->gguf_hdr);
        mapped_file_close(&out->mf);
        return TN_ERR_OOM;
    }

    /* Phase 17.7: MLA k_rope_cache (allocated only when has_mla=1) */
    if (out->moe_config.has_mla) {
        if (mla_run_state_alloc(out->state, &out->config, &out->moe_config, out->config.seq_len) != TN_OK) {
            fprintf(stderr, "Failed to allocate MLA k_rope_cache.\n");
            run_state_free(out->state);
            free(out->state);
            out->state = NULL;
            if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
            weights_free_pointers(&out->weights);
            if (out->gguf_store) weights_free_gguf(out->gguf_store);
            gguf_header_free(&out->gguf_hdr);
            mapped_file_close(&out->mf);
            return TN_ERR_OOM;
        }
    }

    /* Qwen3.5/3.6 hybrid attention state (allocated only when has_linear_attn=1) */
    if (out->moe_config.has_linear_attn) {
        if (q35_run_state_alloc(out->state, &out->config, &out->moe_config, out->config.seq_len) != TN_OK) {
            fprintf(stderr, "Failed to allocate Qwen3.5/3.6 hybrid attention state.\n");
            if (out->moe_config.has_mla) mla_run_state_free(out->state, out->config.n_layers);
            run_state_free(out->state);
            free(out->state);
            out->state = NULL;
            if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
            weights_free_pointers(&out->weights);
            if (out->gguf_store) weights_free_gguf(out->gguf_store);
            gguf_header_free(&out->gguf_hdr);
            mapped_file_close(&out->mf);
            return TN_ERR_OOM;
        }
    }

    /* Qwen3-MoE attention state (allocated only when has_qk_norm=1) */
    if (out->moe_config.has_qk_norm) {
        if (qwen3moe_run_state_alloc(out->state, &out->config, &out->moe_config, out->config.seq_len) != TN_OK) {
            fprintf(stderr, "Failed to allocate Qwen3-MoE attention state.\n");
            if (out->moe_config.has_mla) mla_run_state_free(out->state, out->config.n_layers);
            if (out->moe_config.has_linear_attn) q35_run_state_free(out->state, &out->config, &out->moe_config);
            run_state_free(out->state);
            free(out->state);
            out->state = NULL;
            if (out->moe_config.is_moe) moe_weights_free(&out->weights, &out->moe_config);
            weights_free_pointers(&out->weights);
            if (out->gguf_store) weights_free_gguf(out->gguf_store);
            gguf_header_free(&out->gguf_hdr);
            mapped_file_close(&out->mf);
            return TN_ERR_OOM;
        }
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

    /* mla_run_state_free/q35_run_state_free/qwen3moe_run_state_free must run
     * before run_state_free() (they free k_rope_cache and the q35_/qwen3moe_
     * pointer arrays that run_state_free() doesn't know about) -- mirrors
     * main()'s own end-of-program cleanup sequence. */
    if (m->moe_config.has_mla) mla_run_state_free(m->state, m->config.n_layers);
    if (m->moe_config.has_linear_attn) q35_run_state_free(m->state, &m->config, &m->moe_config);
    if (m->moe_config.has_qk_norm) qwen3moe_run_state_free(m->state, &m->config);
    run_state_free(m->state);
    free(m->state);
    m->state = NULL;

    if (m->moe_config.is_moe) moe_weights_free(&m->weights, &m->moe_config);
    weights_free_pointers(&m->weights);
    if (m->gguf_store) weights_free_gguf(m->gguf_store);
    gguf_header_free(&m->gguf_hdr);
    mapped_file_close(&m->mf);
}
