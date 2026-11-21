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

/* RoPE pair layout: from the GGUF architecture unless --rope-layout says
 * otherwise (e.g. a GGUF converted with a non-standard Q/K order). */
static void session_rope_layout(PzSession *ses) {
    if (ses->args.perf.rope_layout >= 0) ses->p.rope_layout = ses->args.perf.rope_layout;
    if (ses->args.output.verbose || ses->args.perf.rope_layout >= 0)
        printf("RoPE layout: %s%s\n", rope_layout_name(ses->p.rope_layout),
               ses->args.perf.rope_layout >= 0 ? " (--rope-layout)" : " (from model architecture)");
}

/* Copies the GGUF's chat template into the tokenizer (silently skipped on OOM). */
static void patch_template_from_gguf(PzSession *ses) {
    const GGUFMeta *m = gguf_meta_find(&ses->gguf_hdr, "tokenizer.chat_template");
    if (!m || m->val_type != GGUF_VAL_STRING || m->val.string.len == 0) return;
    ses->t.chat_template = (char *)malloc(m->val.string.len + 1);
    if (!ses->t.chat_template) return;
    memcpy(ses->t.chat_template, m->val.string.str, m->val.string.len);
    ses->t.chat_template[m->val.string.len] = '\0';
    TN_LOG_INF("tokenizer", "[tokenizer] chat_template patched from GGUF metadata\n");
}

/* Appends the EOS id to eos_list unless it is already there. agent_loop.c EOS
 * detection only iterates eos_list, not eos_token_id. */
static void eos_list_add(Tokenizer *t) {
    if (t->eos_token_id < 0 || t->n_eos >= TN_TOKENIZER_MAX_EOG) return;
    for (int ei = 0; ei < t->n_eos; ei++)
        if (t->eos_list[ei] == t->eos_token_id) return;
    t->eos_list[t->n_eos++] = t->eos_token_id;
}

/* GGUF metadata is authoritative for BOS/EOS when the .bin vocab scan did not
 * find a match (id < 0 means no candidate from the BOS_CANDIDATES / EOS_PRIMARY
 * lists was in this model's vocabulary, e.g. DeepSeek's <｜begin▁of▁sentence｜>). */
static void patch_special_ids_from_gguf(PzSession *ses) {
    if (ses->t.bos_token_id < 0)
        ses->t.bos_token_id =
            (int)gguf_meta_u32(&ses->gguf_hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
    if (ses->t.eos_token_id < 0) {
        ses->t.eos_token_id =
            (int)gguf_meta_u32(&ses->gguf_hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
        eos_list_add(&ses->t);
    }
}

/* External tokenizer file. When the model is GGUF, pull chat_template and special
 * token IDs from its metadata if the .bin did not supply them: the .bin carries
 * vocab/BPE while the model file is authoritative for chat format. Returns 1 on a
 * load failure. */
static int load_external_tokenizer(PzSession *ses) {
    if (tokenizer_load(&ses->t, ses->args.input.tokenizer_path) != TN_OK) {
        TN_LOG_ERR("session_tokenizer", "Failed to load tokenizer: %s\n",
                   ses->args.input.tokenizer_path);
        return 1;
    }
    if (!ses->is_gguf) return 0;
    if (!ses->t.chat_template) patch_template_from_gguf(ses);
    patch_special_ids_from_gguf(ses);
    return 0;
}

/* Phase 34.5: auto-load the tokenizer from the GGUF when no external file is given. */
static void load_gguf_tokenizer(PzSession *ses) {
    TernaryError terr = tokenizer_load_from_gguf(&ses->t, &ses->gguf_hdr);
    if (terr == TN_OK) return;
    /* Non-fatal: engine can still run if tokens passed externally */
    TN_LOG_WRN("warn",
               "[WARN] GGUF tokenizer load failed (err=%d). "
               "Run with --tokenizer <path> to specify one.\n",
               (int)terr);
}

/* --chat-template-file replaces the model's template for the CLI and
 * --server alike (both render from t.chat_template). Some GGUFs ship a
 * broken one — the official BitNet b1.58 2B-4T file appends eos_token
 * after the generation prompt (2026-10-19). */
static void apply_template_file(PzSession *ses) {
    ses->chat_template_from_file = 0;
    if (!ses->args.input.chat_template_file) return;
    size_t tl  = 0;
    char *tmpl = tn_read_text_file(ses->args.input.chat_template_file, (size_t)1 << 20, &tl);
    if (!tmpl) {
        TN_LOG_WRN("warn",
                   "[WARN] cannot read --chat-template-file '%s' (1 B..1 MiB); "
                   "keeping the model's template\n",
                   ses->args.input.chat_template_file);
        return;
    }
    free(ses->t.chat_template);
    ses->t.chat_template         = tmpl;
    ses->chat_template_from_file = 1;
    TN_LOG_INF("tokenizer", "[tokenizer] chat template from %s (%zu bytes)\n",
               ses->args.input.chat_template_file, tl);
}

/* Known-broken embedded templates are replaced by a corrected built-in one
 * (2026-11-16: the official BitNet b1.58 2B-4T template ends the prompt in
 * EOS, so the model generated nothing). Order: --chat-template-file > this
 * > the GGUF's template; a request's own template still wins per request. */
static void apply_builtin_template_fix(PzSession *ses) {
    ses->chat_template_builtin = 0;
    if (!ses->is_gguf || ses->chat_template_from_file || !ses->t.chat_template) return;
    const char *fix = tn_template_known_fix(ses->gguf_hdr.arch, ses->t.chat_template);
    char *copy      = fix ? strdup(fix) : NULL;
    if (!copy) return;
    free(ses->t.chat_template);
    ses->t.chat_template       = copy;
    ses->chat_template_builtin = 1;
    TN_LOG_INF("tokenizer",
               "[tokenizer] the model's chat template ends the prompt with the "
               "end-of-sequence token (known problem); using the built-in %s "
               "template. --chat-template-file overrides it.\n",
               ses->gguf_hdr.arch);
}

/* RoPE layout override, then the tokenizer (external file, or the GGUF's own) and the chat
 * template (--chat-template-file > built-in fix for a known-broken template > the model's).
 * Returns 1 when --tokenizer cannot be loaded. */
int session_setup_tokenizer(PzSession *ses) {
    session_rope_layout(ses);
    if (ses->args.input.tokenizer_path) {
        if (load_external_tokenizer(ses)) return 1;
    } else if (ses->is_gguf) {
        load_gguf_tokenizer(ses);
    }
    apply_template_file(ses);
    apply_builtin_template_fix(ses);
    return 0;
}
