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

/* Loads --draft-model, --prm, --lora and --memory-db. Returns 1 when --draft-model or --lora was
 * given and could not be loaded (the caller releases the session). */
int session_load_extras(PzSession *S) {
    /* ── Phase 18: speculative decoding draft model (optional) ──────────────
     * --draft-model is the only way this ever activates -- see cli/args.h.
     * Loaded once here so both call sites below (--prompt one-shot and the
     * REPL) can share it. */
    if (S->args.draft_model_path) {
        if (draft_model_load(&S->draft, S->args.draft_model_path, S->p.vocab_size,
                             S->args.max_tokens + S->args.spec_length + 64, S->tp) != TN_OK) {
            TN_LOG_ERR("session_extras", "Error: failed to load --draft-model '%s'.\n",
                       S->args.draft_model_path);
            return 1;
        }
        S->draft_active = true;
    }

    /* ── Phase 30: process reward model for --mcts (optional) ───────────── */
    mcts_options_defaults(&S->args.mcts);
    if (S->args.prm_path &&
        prm_load(&S->prm, S->args.prm_path, S->args.prm_format, S->args.prm_good, S->args.prm_bad,
                 S->s->max_seq_len, S->tp) != TN_OK) {
        TN_LOG_INF("prm", "[prm] continuing without a PRM: steps are scored by the model itself\n");
    }

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
    if (S->args.lora_path) {
        /* Qwen3 (QK-norm path) has head_dim from GGUF key_length, so its Q
         * is n_heads*head_dim wide, not dim (Phase 19-B, 2026-10-19). */
        int lhd      = (S->mc.has_qk_norm && S->mc.attn_head_dim > 0) ? S->mc.attn_head_dim : 0;
        int l_q_dim  = lhd ? S->p.n_heads * lhd : S->p.dim;
        int l_kv_dim = lhd ? S->p.n_kv_heads * lhd : config_kv_dim(&S->p);
        if (lora_load_ex(&S->lora, S->args.lora_path, S->p.dim, l_q_dim, l_kv_dim, S->p.hidden_dim,
                         S->p.n_layers) != TN_OK) {
            TN_LOG_ERR("session_extras", "Error: failed to load --lora '%s'.\n", S->args.lora_path);
            return 1;
        }
        S->lora_active    = true;
        S->s->active_lora = &S->lora;
        TN_LOG_INF("lora", "[lora] '%s' active (rank %d, alpha %.1f, q_dim %d, kv_dim %d)\n",
                   S->args.lora_path, S->lora.rank, (double)S->lora.alpha, l_q_dim, l_kv_dim);
    }

    /* ── Phase 15: RAG initialisation ────────────────────────────────────── */

    if (S->args.memory_db_path) {
        TernaryError err = vector_db_open(&S->rag.db, S->args.memory_db_path, S->p.dim);
        if (err != TN_OK) {
            TN_LOG_WRN("rag",
                       "[RAG] Warning: could not open vector DB '%s' (err=%d). "
                       "Memory features disabled.\n",
                       S->args.memory_db_path, (int)err);
        } else {
            err = embedder_init(&S->rag.emb, &S->p);
            if (err != TN_OK) {
                TN_LOG_WRN("rag",
                           "[RAG] Warning: could not initialise embedder (err=%d). "
                           "Memory features disabled.\n",
                           (int)err);
                vector_db_close(&S->rag.db);
            } else {
                S->rag.enabled = 1;
                S->rag_ok      = 1;
                TN_OUT_INF("rag", "[RAG] Memory enabled — %d entries in '%s'\n",
                           S->rag.db.num_entries, S->args.memory_db_path);
            }
        }
    }
    /* ─────────────────────────────────────────────────────────────────────── */

    return 0;
}

/* --image (vision pipeline, shared with the HTTP API) and the attention kernel settings. */
void session_apply_runtime_options(PzSession *S) {
    static char vis_prompt_buf[VIS_PROMPT_BUF];
    /* ── Phase 34: Vision pipeline (extracted to multimodal/vision_pipeline.c
     * in Phase 22.2 so the HTTP API can reuse it for image_url uploads) ──── */
    if (S->args.image_path) {
        const char *stripped_prompt = S->args.prompt;
        vision_pipeline_run(S->args.image_path, S->args.vision_path, S->args.proj_path, &S->p,
                            &S->w, S->s, &S->mc, &S->t, S->tp, S->args.prompt, vis_prompt_buf,
                            sizeof(vis_prompt_buf), &stripped_prompt);
        S->args.prompt = (char *)stripped_prompt;
    }

    /* Phase 27: attention kernel + head-parallel threshold (tile size from L1d). */
    attn_kernel_configure(S->args.attn_kernel ? (AttnKernel)attn_kernel_parse(S->args.attn_kernel)
                                              : ATTN_KERNEL_STANDARD,
                          S->args.attn_tile, config_head_dim(&S->p));
    attn_parallel_min_ctx_set(S->args.attn_parallel_min_ctx);
    if (S->args.verbose || S->args.attn_kernel) {
        const CacheInfo *ci = tn_cache_info();
        tn_log_out(
            TN_LOG_INFO, "session_extras",
            "Attention: %s kernel (tile %d), heads parallel from %d ctx; L1d %d KiB, L2 %d KiB "
            "(%s)\n",
            attn_kernel_name(attn_kernel_get()), attn_tile_get(), attn_parallel_min_ctx_get(),
            ci->l1d_bytes / 1024, ci->l2_bytes / 1024, tn_cache_source_name(ci->source));
    }


    /* Phase 27: attention kernel + head-parallel threshold (tile size from L1d). */
    attn_kernel_configure(S->args.attn_kernel ? (AttnKernel)attn_kernel_parse(S->args.attn_kernel)
                                              : ATTN_KERNEL_STANDARD,
                          S->args.attn_tile, config_head_dim(&S->p));
    attn_parallel_min_ctx_set(S->args.attn_parallel_min_ctx);
    if (S->args.verbose || S->args.attn_kernel) {
        const CacheInfo *ci = tn_cache_info();
        tn_log_out(
            TN_LOG_INFO, "session_extras",
            "Attention: %s kernel (tile %d), heads parallel from %d ctx; L1d %d KiB, L2 %d KiB "
            "(%s)\n",
            attn_kernel_name(attn_kernel_get()), attn_tile_get(), attn_parallel_min_ctx_get(),
            ci->l1d_bytes / 1024, ci->l2_bytes / 1024, tn_cache_source_name(ci->source));
    }
}
