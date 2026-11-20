/* Model loading and the one release function every later phase relies on. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/session.h"
#include "core/log.h"
#include "core/lora.h"
#include "core/moe_weights.h"
#include "core/progress.h"
#include "core/weights.h"
#include "kv_cache/kv_strategy.h"
#include "memory/free_ram.h"
#include "memory/mapped_file.h"
#include "rag/embedder.h"
#include "rag/vector_db.h"
#include "reasoning/prm.h"
#include "runtime/hardware_profile.h"
#include "runtime/model_load.h"
#include "speculative/draft_model.h"
#include "threading/thread_pool.h"

/* GGUF: the shared loader (also used for the draft model) does everything. */
static int load_gguf(PzSession *S) {
    LoadedModel lm;
    TernaryError lerr = load_gguf_model(&lm, S->args.input.model_path, /*is_primary=*/true, 0,
                                        S->tp, S->stdout_is_tty);
    if (lerr != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to load model: %s\n", tn_error_str(lerr));
        return 1;
    }
    S->p          = lm.config;
    S->w          = lm.weights;
    S->mc         = lm.moe_config;
    S->s          = lm.state;
    S->mf         = lm.mf;
    S->gguf_hdr   = lm.gguf_hdr;
    S->gguf_store = lm.gguf_store;
    return 0;
}

/* Native ternary `.bin`: mapped, parsed and its run state sized here (no draft-model use case,
 * see runtime/model_load.h, so it does not share the GGUF helper). */
static int load_native(PzSession *S) {
    TN_OUT_INF("session_model", "Model format: native ternary\n");

    if (mapped_file_open(&S->mf, S->args.input.model_path) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to map model file: %s\n", S->args.input.model_path);
        return 1;
    }

    if (config_read(&S->p, S->mf.data, S->mf.size) != TN_OK) {
        TN_LOG_ERR("session_model", "Invalid model configuration.\n");
        mapped_file_close(&S->mf);
        return 1;
    }
    config_print(&S->p);

    /* MoE model: scale_mode=2 → read 32-byte MoE header at offset 64,
     * weight data starts at byte 128 (64 header + 32 MoE + 32 padding).
     * Dense model: scale_mode=0 → weight data starts at byte 64. */
    tn_i8 *weight_data;
    size_t data_size;
    if (S->p.scale_mode == 2) {
        const size_t MOE_HDR_OFFSET = 64;
        const size_t WEIGHT_OFFSET  = 128;
        if (S->mf.size < WEIGHT_OFFSET) {
            TN_LOG_ERR("session_model", "MoE model file too small.\n");
            mapped_file_close(&S->mf);
            return 1;
        }
        size_t moe_size = S->mf.size - MOE_HDR_OFFSET;
        size_t moe_off  = 0;
        if (moe_config_read(&S->mc, (char *)S->mf.data + MOE_HDR_OFFSET, moe_size, &moe_off) !=
            TN_OK) {
            TN_LOG_ERR("session_model", "Failed to read MoE config header.\n");
            mapped_file_close(&S->mf);
            return 1;
        }
        moe_config_print(&S->mc);
        weight_data = (tn_i8 *)S->mf.data + WEIGHT_OFFSET;
        data_size   = S->mf.size - WEIGHT_OFFSET;
    } else {
        weight_data = (tn_i8 *)S->mf.data + 64;
        data_size   = S->mf.size - 64;
    }

    if (weights_alloc_pointers(&S->w, &S->p) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to allocate weight pointers.\n");
        mapped_file_close(&S->mf);
        return 1;
    }

    if (S->mc.is_moe) {
        if (moe_weights_alloc(&S->w, &S->p, &S->mc) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate MoE weight pointers.\n");
            if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
            weights_free_pointers(&S->w);
            mapped_file_close(&S->mf);
            return 1;
        }
    }

    if (weights_map(&S->w, &S->p, &S->mc, weight_data, data_size) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to map weight structures from file.\n");
        moe_weights_free(&S->w, &S->mc);
        weights_free_pointers(&S->w);
        mapped_file_close(&S->mf);
        return 1;
    }

    /* Native-format twin of the GGUF set_model_bytes call above
     * (2026-07-17, independent-review finding: without this, any native
     * model that isn't exactly BitNet-2B kept the hardcoded pre-load
     * estimate forever). Per-token traffic = ternary layers + norms
     * (weight data minus the BF16 embedding table, which is read one
     * row per token) + the classifier at its selected precision (the
     * classifier is derived from that embedding: BF16 = full table,
     * INT8/INT4 = half/quarter). Native MoE (scale_mode==2) overcounts
     * inactive experts here — same TODO(PZ-TODO-1) as the GGUF branch. */
    {
        double embed_bytes            = (double)S->p.vocab_size * S->p.dim * 2.0;
        const TnHardwareProfile *hp_n = tn_hardware_profile_get();
        double cls_bytes              = embed_bytes; /* BF16 default */
        if (hp_n && hp_n->classifier_fmt == TN_CLS_INT8) cls_bytes = embed_bytes / 2.0;
        if (hp_n && hp_n->classifier_fmt == TN_CLS_INT4) cls_bytes = embed_bytes / 4.0;
        double per_tok = ((double)data_size - embed_bytes) + cls_bytes;
        tn_hardware_profile_set_model_bytes(per_tok, cls_bytes);
        if (hp_n && per_tok > 0.0) {
            TN_OUT_INF("profile",
                       "[profile] Data/token (loaded model): %.0f MB -> "
                       "ceiling %.1f tok/s at %.1f GB/s\n",
                       per_tok / (1024.0 * 1024.0), hp_n->theoretical_ceiling,
                       hp_n->measured_bw_gbps);
        }
    }

    tn_progress_stage(3, 4, "Preparing runtime...", S->stdout_is_tty);

    /* KV Strategy — measured AFTER model weights are loaded so that any F32-dequantised
     * weight allocations (MLA projections, shared experts, norms) are already counted in
     * consumed RAM.  For mmap'd GGUF models the raw quantised expert bytes don't use
     * physical RAM until accessed, so only the upfront malloc'd F32 blocks matter here.
     * Re-measuring at this point prevents DeepSeek-style OOM where a 163840-token context
     * would require 7+ GB KV cache on a machine that only has 2–3 GB left after load.
     *
     * This is the native-format twin of load_gguf_model()'s identical KV-strategy/
     * RunState-allocation sequence (Phase 18 Stage 3) — native `.bin` models have no
     * draft-model use case (see runtime/model_load.h), so this stays inline here rather
     * than sharing that GGUF-only helper. */
    {
        tn_i64 post_load_ram    = tn_get_free_ram();
        KVStrategyResult kv_res = select_kv_strategy(&S->p, post_load_ram, &S->mc);
        S->p.seq_len            = kv_res.max_seq_len;
        kv_strategy_print(&kv_res, S->p.seq_len);
    }

    /* Setup RunState */
    S->s = (RunState *)malloc(sizeof(RunState));
    if (!S->s) {
        TN_LOG_ERR("session_model", "Failed to allocate RunState header.\n");
        if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
        weights_free_pointers(&S->w);
        mapped_file_close(&S->mf);
        return 1;
    }

    /* Qwen3.5/3.6 hybrid models and Qwen3-MoE models never read the generic
     * key_cache/value_cache (they use their own correctly-sized
     * q35_key_cache/q35_value_cache or qwen3moe_key_cache/
     * qwen3moe_value_cache, allocated below) — skip_kv_cache=true avoids a
     * multi-GB calloc+memset for those models' context. See
     * run_state_alloc_ex's header comment and docs/ai/mistakes.md. */
    if (run_state_alloc_ex(S->s, &S->p, S->p.seq_len, S->mc.has_linear_attn || S->mc.has_qk_norm) !=
        TN_OK) {
        TN_LOG_ERR("session_model", "Failed to allocate RunState buffers.\n");
        free(S->s);
        if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
        weights_free_pointers(&S->w);
        mapped_file_close(&S->mf);
        return 1;
    }

    /* Phase 17.7: MLA k_rope_cache (allocated only when mla.enabled=1) */
    if (S->mc.mla.enabled) {
        if (mla_run_state_alloc(S->s, &S->p, &S->mc, S->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate MLA k_rope_cache.\n");
            run_state_free(S->s);
            free(S->s);
            if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
            weights_free_pointers(&S->w);
            mapped_file_close(&S->mf);
            return 1;
        }
    }

    /* Qwen3.5/3.6 hybrid attention state (allocated only when has_linear_attn=1) */
    if (S->mc.has_linear_attn) {
        if (q35_run_state_alloc(S->s, &S->p, &S->mc, S->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate Qwen3.5/3.6 hybrid attention state.\n");
            run_state_free(S->s);
            free(S->s);
            if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
            weights_free_pointers(&S->w);
            mapped_file_close(&S->mf);
            return 1;
        }
    }

    /* Qwen3-MoE attention state (allocated only when has_qk_norm=1) */
    if (S->mc.has_qk_norm) {
        if (qwen3moe_run_state_alloc(S->s, &S->p, &S->mc, S->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate Qwen3-MoE attention state.\n");
            run_state_free(S->s);
            free(S->s);
            if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
            weights_free_pointers(&S->w);
            mapped_file_close(&S->mf);
            return 1;
        }
    }

    return 0;
}

/* Opens the model file and builds weights and run state. Returns 1 when loading failed (the
 * reason is printed and nothing needs releasing except the thread pool). */
int session_load_model(PzSession *S) {
    tn_progress_stage(1, 4, "Opening model file...", S->stdout_is_tty);

    tn_progress_stage(2, 4, "Loading weights...", S->stdout_is_tty);

    memset(&S->w, 0, sizeof(S->w)); /* required before weights_alloc_pointers */
    moe_config_init_dense(&S->mc);  /* initialise to dense; overridden below for MoE models */
    memset(&S->gguf_hdr, 0, sizeof(S->gguf_hdr));
    return S->is_gguf ? load_gguf(S) : load_native(S);
}

/* Frees whatever the run built, in reverse order. Safe at any stage: a part that was never set up
 * is zeroed (the session starts zeroed) or flagged off. Needs S->s set, i.e. a loaded model. */
void session_release(PzSession *S) {
    if (S->draft_active) draft_model_free(&S->draft);
    prm_free(&S->prm);
    if (S->lora_active) lora_free(&S->lora);
    if (S->rag_ok) {
        embedder_free(&S->rag.emb);
        vector_db_close(&S->rag.db);
    }
    /* tokenizer_free is safe unconditionally: t was memset to zero above,
     * and both load paths (--tokenizer file, or GGUF auto-load) populate
     * the same fields it frees. Previously gated on args.tokenizer_path,
     * which skipped cleanup entirely for the GGUF-auto-load path — the
     * common case when no external --tokenizer is passed — leaking the
     * whole vocab (~49k strings) every run. */
    tokenizer_free(&S->t);
    /* mla_run_state_free/q35_run_state_free/qwen3moe_run_state_free must run
     * before run_state_free() (they free k_rope_cache and the q35_ / qwen3moe_
     * pointer arrays that run_state_free() doesn't know about).
     * mla_run_state_free was previously never called anywhere in the
     * shutdown path — a real leak of k_rope_cache (n_layers malloc'd
     * buffers) + mla_rope_freq on every DeepSeek/MLA run, only its own
     * OOM-cleanup path called it. Fixed here alongside the new
     * q35_run_state_free wiring (see docs/ai/mistakes.md). */
    if (S->mc.mla.enabled) mla_run_state_free(S->s, S->p.n_layers);
    if (S->mc.has_linear_attn) q35_run_state_free(S->s, &S->p, &S->mc);
    if (S->mc.has_qk_norm) qwen3moe_run_state_free(S->s, &S->p);
    run_state_free(S->s);
    free(S->s);
    if (S->mc.is_moe) moe_weights_free(&S->w, &S->mc);
    weights_free_pointers(&S->w);
    if (S->gguf_store) weights_free_gguf(S->gguf_store);
    if (S->is_gguf) gguf_header_free(&S->gguf_hdr);
    mapped_file_close(&S->mf);
    threadpool_destroy(S->tp);
}
