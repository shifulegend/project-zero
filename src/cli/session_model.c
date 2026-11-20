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
static int load_gguf(PzSession *ses) {
    LoadedModel lm;
    TernaryError lerr = load_gguf_model(&lm, ses->args.input.model_path, /*is_primary=*/true, 0,
                                        ses->tp, ses->stdout_is_tty);
    if (lerr != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to load model: %s\n", tn_error_str(lerr));
        return 1;
    }
    ses->p          = lm.config;
    ses->w          = lm.weights;
    ses->mc         = lm.moe_config;
    ses->s          = lm.state;
    ses->mf         = lm.mf;
    ses->gguf_hdr   = lm.gguf_hdr;
    ses->gguf_store = lm.gguf_store;
    return 0;
}

/* Native ternary `.bin`: mapped, parsed and its run state sized here (no draft-model use case,
 * see runtime/model_load.h, so it does not share the GGUF helper). */
static int load_native(PzSession *ses) {
    TN_OUT_INF("session_model", "Model format: native ternary\n");

    if (mapped_file_open(&ses->mf, ses->args.input.model_path) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to map model file: %s\n", ses->args.input.model_path);
        return 1;
    }

    if (config_read(&ses->p, ses->mf.data, ses->mf.size) != TN_OK) {
        TN_LOG_ERR("session_model", "Invalid model configuration.\n");
        mapped_file_close(&ses->mf);
        return 1;
    }
    config_print(&ses->p);

    /* MoE model: scale_mode=2 → read 32-byte MoE header at offset 64,
     * weight data starts at byte 128 (64 header + 32 MoE + 32 padding).
     * Dense model: scale_mode=0 → weight data starts at byte 64. */
    tn_i8 *weight_data;
    size_t data_size;
    if (ses->p.scale_mode == 2) {
        const size_t MOE_HDR_OFFSET = 64;
        const size_t WEIGHT_OFFSET  = 128;
        if (ses->mf.size < WEIGHT_OFFSET) {
            TN_LOG_ERR("session_model", "MoE model file too small.\n");
            mapped_file_close(&ses->mf);
            return 1;
        }
        size_t moe_size = ses->mf.size - MOE_HDR_OFFSET;
        size_t moe_off  = 0;
        if (moe_config_read(&ses->mc, (char *)ses->mf.data + MOE_HDR_OFFSET, moe_size, &moe_off) !=
            TN_OK) {
            TN_LOG_ERR("session_model", "Failed to read MoE config header.\n");
            mapped_file_close(&ses->mf);
            return 1;
        }
        moe_config_print(&ses->mc);
        weight_data = (tn_i8 *)ses->mf.data + WEIGHT_OFFSET;
        data_size   = ses->mf.size - WEIGHT_OFFSET;
    } else {
        weight_data = (tn_i8 *)ses->mf.data + 64;
        data_size   = ses->mf.size - 64;
    }

    if (weights_alloc_pointers(&ses->w, &ses->p) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to allocate weight pointers.\n");
        mapped_file_close(&ses->mf);
        return 1;
    }

    if (ses->mc.is_moe) {
        if (moe_weights_alloc(&ses->w, &ses->p, &ses->mc) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate MoE weight pointers.\n");
            if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
            weights_free_pointers(&ses->w);
            mapped_file_close(&ses->mf);
            return 1;
        }
    }

    if (weights_map(&ses->w, &ses->p, &ses->mc, weight_data, data_size) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to map weight structures from file.\n");
        moe_weights_free(&ses->w, &ses->mc);
        weights_free_pointers(&ses->w);
        mapped_file_close(&ses->mf);
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
        double embed_bytes            = (double)ses->p.vocab_size * ses->p.dim * 2.0;
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

    tn_progress_stage(3, 4, "Preparing runtime...", ses->stdout_is_tty);

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
        KVStrategyResult kv_res = select_kv_strategy(&ses->p, post_load_ram, &ses->mc);
        ses->p.seq_len          = kv_res.max_seq_len;
        kv_strategy_print(&kv_res, ses->p.seq_len);
    }

    /* Setup RunState */
    ses->s = (RunState *)malloc(sizeof(RunState));
    if (!ses->s) {
        TN_LOG_ERR("session_model", "Failed to allocate RunState header.\n");
        if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
        weights_free_pointers(&ses->w);
        mapped_file_close(&ses->mf);
        return 1;
    }

    /* Qwen3.5/3.6 hybrid models and Qwen3-MoE models never read the generic
     * key_cache/value_cache (they use their own correctly-sized
     * q35.key_cache/q35.value_cache or qwen3moe.key_cache/
     * qwen3moe.value_cache, allocated below) — skip_kv_cache=true avoids a
     * multi-GB calloc+memset for those models' context. See
     * run_state_alloc_ex's header comment and docs/ai/mistakes.md. */
    if (run_state_alloc_ex(ses->s, &ses->p, ses->p.seq_len,
                           ses->mc.has_linear_attn || ses->mc.has_qk_norm) != TN_OK) {
        TN_LOG_ERR("session_model", "Failed to allocate RunState buffers.\n");
        free(ses->s);
        if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
        weights_free_pointers(&ses->w);
        mapped_file_close(&ses->mf);
        return 1;
    }

    /* Phase 17.7: MLA k_rope_cache (allocated only when mla.enabled=1) */
    if (ses->mc.mla.enabled) {
        if (mla_run_state_alloc(ses->s, &ses->p, &ses->mc, ses->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate MLA k_rope_cache.\n");
            run_state_free(ses->s);
            free(ses->s);
            if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
            weights_free_pointers(&ses->w);
            mapped_file_close(&ses->mf);
            return 1;
        }
    }

    /* Qwen3.5/3.6 hybrid attention state (allocated only when has_linear_attn=1) */
    if (ses->mc.has_linear_attn) {
        if (q35_run_state_alloc(ses->s, &ses->p, &ses->mc, ses->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate Qwen3.5/3.6 hybrid attention state.\n");
            run_state_free(ses->s);
            free(ses->s);
            if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
            weights_free_pointers(&ses->w);
            mapped_file_close(&ses->mf);
            return 1;
        }
    }

    /* Qwen3-MoE attention state (allocated only when has_qk_norm=1) */
    if (ses->mc.has_qk_norm) {
        if (qwen3moe_run_state_alloc(ses->s, &ses->p, &ses->mc, ses->p.seq_len) != TN_OK) {
            TN_LOG_ERR("session_model", "Failed to allocate Qwen3-MoE attention state.\n");
            run_state_free(ses->s);
            free(ses->s);
            if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
            weights_free_pointers(&ses->w);
            mapped_file_close(&ses->mf);
            return 1;
        }
    }

    return 0;
}

/* Opens the model file and builds weights and run state. Returns 1 when loading failed (the
 * reason is printed and nothing needs releasing except the thread pool). */
int session_load_model(PzSession *ses) {
    tn_progress_stage(1, 4, "Opening model file...", ses->stdout_is_tty);

    tn_progress_stage(2, 4, "Loading weights...", ses->stdout_is_tty);

    memset(&ses->w, 0, sizeof(ses->w)); /* required before weights_alloc_pointers */
    moe_config_init_dense(&ses->mc);    /* initialise to dense; overridden below for MoE models */
    memset(&ses->gguf_hdr, 0, sizeof(ses->gguf_hdr));
    return ses->is_gguf ? load_gguf(ses) : load_native(ses);
}

/* Frees whatever the run built, in reverse order. Safe at any stage: a part that was never set up
 * is zeroed (the session starts zeroed) or flagged off. Needs ses->s set, i.e. a loaded model. */
void session_release(PzSession *ses) {
    if (ses->draft_active) draft_model_free(&ses->draft);
    prm_free(&ses->prm);
    if (ses->lora_active) lora_free(&ses->lora);
    if (ses->rag_ok) {
        embedder_free(&ses->rag.emb);
        vector_db_close(&ses->rag.db);
    }
    /* tokenizer_free is safe unconditionally: t was memset to zero above,
     * and both load paths (--tokenizer file, or GGUF auto-load) populate
     * the same fields it frees. Previously gated on args.tokenizer_path,
     * which skipped cleanup entirely for the GGUF-auto-load path — the
     * common case when no external --tokenizer is passed — leaking the
     * whole vocab (~49k strings) every run. */
    tokenizer_free(&ses->t);
    /* mla_run_state_free/q35_run_state_free/qwen3moe_run_state_free must run
     * before run_state_free() (they free k_rope_cache and the q35_ / qwen3moe_
     * pointer arrays that run_state_free() doesn't know about).
     * mla_run_state_free was previously never called anywhere in the
     * shutdown path — a real leak of k_rope_cache (n_layers malloc'd
     * buffers) + mla_rope_freq on every DeepSeek/MLA run, only its own
     * OOM-cleanup path called it. Fixed here alongside the new
     * q35_run_state_free wiring (see docs/ai/mistakes.md). */
    if (ses->mc.mla.enabled) mla_run_state_free(ses->s, ses->p.n_layers);
    if (ses->mc.has_linear_attn) q35_run_state_free(ses->s, &ses->p, &ses->mc);
    if (ses->mc.has_qk_norm) qwen3moe_run_state_free(ses->s, &ses->p);
    run_state_free(ses->s);
    free(ses->s);
    if (ses->mc.is_moe) moe_weights_free(&ses->w, &ses->mc);
    weights_free_pointers(&ses->w);
    if (ses->gguf_store) weights_free_gguf(ses->gguf_store);
    if (ses->is_gguf) gguf_header_free(&ses->gguf_hdr);
    mapped_file_close(&ses->mf);
    threadpool_destroy(ses->tp);
}
