/* Tokenizer and chat-template selection. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/session.h"
#include "core/gguf_reader.h"
#include "core/log.h"
#include "core/text_file.h"
#include "math/rope.h"
#include "tokenizer/template_guard.h"
#include "tokenizer/tokenizer_gguf.h"

/* RoPE layout override, then the tokenizer (external file, or the GGUF's own) and the chat
 * template (--chat-template-file > built-in fix for a known-broken template > the model's).
 * Returns 1 when --tokenizer cannot be loaded. */
int session_setup_tokenizer(PzSession *S) {
    /* RoPE pair layout: from the GGUF architecture unless --rope-layout says
     * otherwise (e.g. a GGUF converted with a non-standard Q/K order). */
    if (S->args.rope_layout >= 0) S->p.rope_layout = S->args.rope_layout;
    if (S->args.verbose || S->args.rope_layout >= 0)
        printf("RoPE layout: %s%s\n", rope_layout_name(S->p.rope_layout),
               S->args.rope_layout >= 0 ? " (--rope-layout)" : " (from model architecture)");

    /* Tokenizer */
    if (S->args.tokenizer_path) {
        if (tokenizer_load(&S->t, S->args.tokenizer_path) != TN_OK) {
            TN_LOG_ERR("session_tokenizer", "Failed to load tokenizer: %s\n",
                       S->args.tokenizer_path);
            return 1;
        }
        /* When the model is GGUF, pull chat_template and special token IDs
         * from the GGUF metadata if the .bin tokenizer did not supply them.
         * This keeps the tokenizer modular: external .bin files carry vocab/BPE
         * while the model file is the authoritative source for chat format. */
        if (S->is_gguf) {
            if (!S->t.chat_template) {
                const GGUFMeta *m = gguf_meta_find(&S->gguf_hdr, "tokenizer.chat_template");
                if (m && m->val_type == GGUF_VAL_STRING && m->val.string.len > 0) {
                    S->t.chat_template = (char *)malloc(m->val.string.len + 1);
                    if (S->t.chat_template) {
                        memcpy(S->t.chat_template, m->val.string.str, m->val.string.len);
                        S->t.chat_template[m->val.string.len] = '\0';
                        TN_LOG_INF("tokenizer",
                                   "[tokenizer] chat_template patched from GGUF metadata\n");
                    }
                }
            }
            /* GGUF metadata is authoritative for BOS/EOS when the .bin vocab
             * scan did not find a match (eos_token_id < 0 means no candidate
             * from the BOS_CANDIDATES / EOS_PRIMARY lists was present in this
             * model's vocabulary, e.g. DeepSeek uses <｜begin▁of▁sentence｜>
             * which is not in the generic candidate lists). */
            if (S->t.bos_token_id < 0)
                S->t.bos_token_id =
                    (int)gguf_meta_u32(&S->gguf_hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
            if (S->t.eos_token_id < 0) {
                S->t.eos_token_id =
                    (int)gguf_meta_u32(&S->gguf_hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
                /* Ensure eos_list is populated so agent_loop.c EOS detection
                 * also fires — agent_loop only iterates eos_list, not eos_token_id */
                if (S->t.eos_token_id >= 0 && S->t.n_eos < TN_TOKENIZER_MAX_EOG) {
                    int already = 0;
                    for (int _i = 0; _i < S->t.n_eos; _i++)
                        if (S->t.eos_list[_i] == S->t.eos_token_id) {
                            already = 1;
                            break;
                        }
                    if (!already) S->t.eos_list[S->t.n_eos++] = S->t.eos_token_id;
                }
            }
        }
    } else if (S->is_gguf) {
        /* Phase 34.5: Auto-load tokenizer from GGUF when no external file given */
        TernaryError terr = tokenizer_load_from_gguf(&S->t, &S->gguf_hdr);
        if (terr != TN_OK) {
            TN_LOG_WRN("warn",
                       "[WARN] GGUF tokenizer load failed (err=%d). "
                       "Run with --tokenizer <path> to specify one.\n",
                       (int)terr);
            /* Non-fatal: engine can still run if tokens passed externally */
        }
    }

    /* --chat-template-file replaces the model's template for the CLI and
     * --server alike (both render from t.chat_template). Some GGUFs ship a
     * broken one — the official BitNet b1.58 2B-4T file appends eos_token
     * after the generation prompt (2026-10-19). */
    S->chat_template_from_file = 0;
    if (S->args.chat_template_file) {
        size_t tl  = 0;
        char *tmpl = tn_read_text_file(S->args.chat_template_file, (size_t)1 << 20, &tl);
        if (!tmpl) {
            TN_LOG_WRN("warn",
                       "[WARN] cannot read --chat-template-file '%s' (1 B..1 MiB); "
                       "keeping the model's template\n",
                       S->args.chat_template_file);
        } else {
            free(S->t.chat_template);
            S->t.chat_template         = tmpl;
            S->chat_template_from_file = 1;
            TN_LOG_INF("tokenizer", "[tokenizer] chat template from %s (%zu bytes)\n",
                       S->args.chat_template_file, tl);
        }
    }

    /* Known-broken embedded templates are replaced by a corrected built-in one
     * (2026-11-16: the official BitNet b1.58 2B-4T template ends the prompt in
     * EOS, so the model generated nothing). Order: --chat-template-file > this
     * > the GGUF's template; a request's own template still wins per request. */
    S->chat_template_builtin = 0;
    if (S->is_gguf && !S->chat_template_from_file && S->t.chat_template) {
        const char *fix = tn_template_known_fix(S->gguf_hdr.arch, S->t.chat_template);
        char *copy      = fix ? strdup(fix) : NULL;
        if (copy) {
            free(S->t.chat_template);
            S->t.chat_template       = copy;
            S->chat_template_builtin = 1;
            TN_LOG_INF("tokenizer",
                       "[tokenizer] the model's chat template ends the prompt with the "
                       "end-of-sequence token (known problem); using the built-in %s "
                       "template. --chat-template-file overrides it.\n",
                       S->gguf_hdr.arch);
        }
    }
    return 0;
}
