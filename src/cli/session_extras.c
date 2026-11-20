/* Optional parts of a run: draft model, process reward model, LoRA adapter, RAG memory, and the
 * image / attention settings that depend on the loaded model. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/session.h"
#include "core/log.h"
#include "core/lora.h"
#include "math/rope.h"
#include "multimodal/vision_pipeline.h"
#include "rag/embedder.h"
#include "rag/vector_db.h"
#include "reasoning/mcts.h"
#include "speculative/draft_model.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"

#define VIS_PROMPT_BUF 8192 /* prompt with the image part removed */

/* ── Phase 19: LoRA adapter (optional) ───────────────────────────────────
 * --lora is the only way this ever activates -- see cli/args.h. Applies
 * to the *primary* model's RunState only (s->active_lora), never the
 * draft model's -- see docs/ai/decision-log.md for the scope rationale.
 * v1 covers the generic dense/GQA attention/FFN path only; lora_load()
 * itself doesn't know or care about the model's architecture family, so
 * loading succeeds even for MLA/Qwen3-MoE/hybrid models, but
 * attention_forward()/ffn_forward() only read active_lora on the
 * generic path -- the adapter would silently have no effect on those
 * architectures. */
static int load_lora(PzSession *ses) {
    /* Qwen3 (QK-norm path) has head_dim from GGUF key_length, so its Q
     * is n_heads*head_dim wide, not dim (Phase 19-B, 2026-10-19). */
    int lhd      = (ses->mc.has_qk_norm && ses->mc.attn_head_dim > 0) ? ses->mc.attn_head_dim : 0;
    int l_q_dim  = lhd ? ses->p.n_heads * lhd : ses->p.dim;
    int l_kv_dim = lhd ? ses->p.n_kv_heads * lhd : config_kv_dim(&ses->p);
    if (lora_load_ex(&ses->lora, ses->args.input.lora_path, ses->p.dim, l_q_dim, l_kv_dim,
                     ses->p.hidden_dim, ses->p.n_layers) != TN_OK) {
        TN_LOG_ERR("session_extras", "Error: failed to load --lora '%s'.\n",
                   ses->args.input.lora_path);
        return 1;
    }
    ses->lora_active    = true;
    ses->s->active_lora = &ses->lora;
    TN_LOG_INF("lora", "[lora] '%s' active (rank %d, alpha %.1f, q_dim %d, kv_dim %d)\n",
               ses->args.input.lora_path, ses->lora.rank, (double)ses->lora.alpha, l_q_dim,
               l_kv_dim);
    return 0;
}

/* Loads --draft-model, --prm, --lora and --memory-db. Returns 1 when --draft-model or --lora was
 * given and could not be loaded (the caller releases the session). */
int session_load_extras(PzSession *ses) {
    /* ── Phase 18: speculative decoding draft model (optional) ──────────────
     * --draft-model is the only way this ever activates -- see cli/args.h.
     * Loaded once here so both call sites below (--prompt one-shot and the
     * REPL) can share it. */
    KVStrategyOptions kv_opt;
    args_fill_kv_options(&ses->args, &kv_opt);
    if (ses->args.input.draft_model_path) {
        if (draft_model_load(&ses->draft, ses->args.input.draft_model_path, ses->p.vocab_size,
                             ses->args.sampling.max_tokens + ses->args.search.spec_length + 64,
                             ses->tp, &kv_opt) != TN_OK) {
            TN_LOG_ERR("session_extras", "Error: failed to load --draft-model '%s'.\n",
                       ses->args.input.draft_model_path);
            return 1;
        }
        ses->draft_active = true;
    }

    /* ── Phase 30: process reward model for --mcts (optional) ───────────── */
    mcts_options_defaults(&ses->args.search.mcts);
    if (ses->args.search.prm_path &&
        prm_load(&ses->prm, ses->args.search.prm_path, ses->args.search.prm_format,
                 ses->args.search.prm_good, ses->args.search.prm_bad, ses->s->max_seq_len, ses->tp,
                 &kv_opt) != TN_OK) {
        TN_LOG_INF("prm", "[prm] continuing without a PRM: steps are scored by the model itself\n");
    }

    if (ses->args.input.lora_path && load_lora(ses)) return 1;

    /* ── Phase 15: RAG initialisation ────────────────────────────────────── */

    if (ses->args.input.memory_db_path) {
        TernaryError err = vector_db_open(&ses->rag.db, ses->args.input.memory_db_path, ses->p.dim);
        if (err != TN_OK) {
            TN_LOG_WRN("rag",
                       "[RAG] Warning: could not open vector DB '%s' (err=%d). "
                       "Memory features disabled.\n",
                       ses->args.input.memory_db_path, (int)err);
        } else {
            err = embedder_init(&ses->rag.emb, &ses->p);
            if (err != TN_OK) {
                TN_LOG_WRN("rag",
                           "[RAG] Warning: could not initialise embedder (err=%d). "
                           "Memory features disabled.\n",
                           (int)err);
                vector_db_close(&ses->rag.db);
            } else {
                ses->rag.enabled = 1;
                ses->rag_ok      = 1;
                TN_OUT_INF("rag", "[RAG] Memory enabled — %d entries in '%s'\n",
                           ses->rag.db.num_entries, ses->args.input.memory_db_path);
            }
        }
    }
    /* ─────────────────────────────────────────────────────────────────────── */

    return 0;
}

/* --image (vision pipeline, shared with the HTTP API) and the attention kernel settings. */
void session_apply_runtime_options(PzSession *ses) {
    static char vis_prompt_buf[VIS_PROMPT_BUF];
    /* ── Phase 34: Vision pipeline (extracted to multimodal/vision_pipeline.c
     * in Phase 22.2 so the HTTP API can reuse it for image_url uploads) ──── */
    if (ses->args.input.image_path) {
        const char *stripped_prompt = ses->args.input.prompt;
        vision_pipeline_run(ses->args.input.image_path, ses->args.input.vision_path,
                            ses->args.input.proj_path, &ses->p, &ses->w, ses->s, &ses->mc, &ses->t,
                            ses->tp, ses->args.input.prompt, vis_prompt_buf, sizeof(vis_prompt_buf),
                            &stripped_prompt);
        ses->args.input.prompt = (char *)stripped_prompt;
    }

    /* Phase 27: attention kernel + head-parallel threshold (tile size from L1d). */
    attn_kernel_configure(ses->args.perf.attn_kernel
                              ? (AttnKernel)attn_kernel_parse(ses->args.perf.attn_kernel)
                              : ATTN_KERNEL_STANDARD,
                          ses->args.perf.attn_tile, config_head_dim(&ses->p));
    attn_parallel_min_ctx_set(ses->args.perf.attn_parallel_min_ctx);
    if (ses->args.output.verbose || ses->args.perf.attn_kernel) {
        const CacheInfo *ci = tn_cache_info();
        tn_log_out(
            TN_LOG_INFO, "session_extras",
            "Attention: %s kernel (tile %d), heads parallel from %d ctx; L1d %d KiB, L2 %d KiB "
            "(%s)\n",
            attn_kernel_name(attn_kernel_get()), attn_tile_get(), attn_parallel_min_ctx_get(),
            ci->l1d_bytes / 1024, ci->l2_bytes / 1024, tn_cache_source_name(ci->source));
    }


    /* Phase 27: attention kernel + head-parallel threshold (tile size from L1d). */
    attn_kernel_configure(ses->args.perf.attn_kernel
                              ? (AttnKernel)attn_kernel_parse(ses->args.perf.attn_kernel)
                              : ATTN_KERNEL_STANDARD,
                          ses->args.perf.attn_tile, config_head_dim(&ses->p));
    attn_parallel_min_ctx_set(ses->args.perf.attn_parallel_min_ctx);
    if (ses->args.output.verbose || ses->args.perf.attn_kernel) {
        const CacheInfo *ci = tn_cache_info();
        tn_log_out(
            TN_LOG_INFO, "session_extras",
            "Attention: %s kernel (tile %d), heads parallel from %d ctx; L1d %d KiB, L2 %d KiB "
            "(%s)\n",
            attn_kernel_name(attn_kernel_get()), attn_tile_get(), attn_parallel_min_ctx_get(),
            ci->l1d_bytes / 1024, ci->l2_bytes / 1024, tn_cache_source_name(ci->source));
    }
}
