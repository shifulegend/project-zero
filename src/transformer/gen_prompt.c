/*
 * src/transformer/gen_prompt.c
 *
 * Renders and tokenizes a generation's prompt (chat template or raw), split out of gen_session.c.
 * Logic unchanged.
 */

#include "transformer/gen_prompt.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/debug.h"
#include "core/log.h"
#include "core/step_timing.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/template_guard.h"

/* Max messages rendered through the chat template in one call (system
 * prompt + history). Bounded so the role/content pointer arrays live on the
 * stack; 64 matches the API's CHAT_MAX_MESSAGES. */
#define GEN_MAX_TEMPLATE_MESSAGES 65

/* 2026-10-22: since untrusted text is neutralised instead of rejected, the
 * tokenizer only fails when the prompt does not fit in the context. */
static void report_encode_failure(const char *what, int seq_len) {
    TN_LOG_ERR("error",
               "[error] the %s does not fit in the model's context of %d tokens. "
               "Nothing was generated; shorten it or raise --ctx-size.\n",
               what, seq_len);
}

static int history_opens_with_system(const GenerationParams *gp) {
    return gp->n_messages > 0 && gp->roles && gp->roles[0] && strcmp(gp->roles[0], "system") == 0;
}

/* Appends the caller's history (null roles read "user", null contents ""); returns the new count. */
static int append_history(const GenerationParams *gp, int n_msg, const char **roles,
                          const char **contents) {
    for (int i = 0; i < gp->n_messages && n_msg < GEN_MAX_TEMPLATE_MESSAGES; i++) {
        roles[n_msg]      = gp->roles[i] ? gp->roles[i] : "user";
        contents[n_msg++] = gp->contents[i] ? gp->contents[i] : "";
    }
    return n_msg;
}

/* The conversation to render: the caller's history when supplied, else the single prompt as one
 * user turn; an optional system prompt goes first unless the history already opens with one.
 * Returns the number of messages. */
static int template_messages(const GenerationParams *gp, const char *prompt, const char **roles,
                             const char **contents) {
    int n_msg = 0;
    if (gp->system_prompt && gp->system_prompt[0] && !history_opens_with_system(gp)) {
        roles[n_msg]      = "system";
        contents[n_msg++] = gp->system_prompt;
    }
    if (gp->n_messages > 0 && gp->roles && gp->contents)
        return append_history(gp, n_msg, roles, contents);
    roles[n_msg]      = "user";
    contents[n_msg++] = prompt;
    return n_msg;
}

/* Renders the chat template over the messages (heap string, or NULL on failure). Message text is
 * content, never chat markup: a "<|im_end|>" typed by a user (or pasted from a document) must not
 * close their turn (2026-10-22, prompt-injection defence; see tokenizer.h). */
static char *render_chat_template(const Tokenizer *tok, const char *chat_tmpl,
                                  const GenerationParams *gp, const char *prompt,
                                  const char *bos_str, const char *eos_str) {
    const char *roles[GEN_MAX_TEMPLATE_MESSAGES];
    const char *contents[GEN_MAX_TEMPLATE_MESSAGES];
    int n_msg = template_messages(gp, prompt, roles, contents);

    char *safe[GEN_MAX_TEMPLATE_MESSAGES];
    int n_safe = 0;
    for (; n_safe < n_msg; n_safe++) {
        safe[n_safe] = tokenizer_neutralize_control(tok, contents[n_safe]);
        if (!safe[n_safe]) break;
        contents[n_safe] = safe[n_safe];
    }
    int64_t t_step  = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    char *formatted = n_safe < n_msg
                          ? NULL
                          : chat_template_apply(chat_tmpl, roles, contents, n_msg, bos_str, eos_str,
                                                /*add_generation_prompt=*/1);
    for (int i = 0; i < n_safe; i++) free(safe[i]);
    if (t_step) tn_step_timing_add(TN_STEP_2_BOS_INJECTION, tn_step_timing_now_ns() - t_step);
    return formatted;
}

/* Appends the assistant prefix (text the reply must start with) to the rendered prompt; frees
 * `formatted` and returns NULL when out of memory. */
static char *append_assistant_prefix(char *formatted, const char *prefix) {
    size_t fl = strlen(formatted), al = strlen(prefix);
    char *joined = (char *)realloc(formatted, fl + al + 1);
    if (!joined) {
        free(formatted);
        return NULL;
    }
    memcpy(joined + fl, prefix, al + 1);
    return joined;
}

/* The text of a special token for the template's bos_token / eos_token, "" when it has none. */
static const char *special_token_text(const Tokenizer *tok, int id) {
    return (id >= 0 && id < tok->vocab_size) ? tok->vocab[id] : "";
}

/* GGUF path: apply the model's own Jinja2 chat template. The template itself renders BOS
 * ({{ bos_token }}), so BOS must NOT be prepended manually: that would produce a double-BOS.
 * Returns the number of prompt tokens, or -1. */
static int build_chat_prompt(GenSession *g, const char *chat_tmpl, const char *prompt) {
    const Config *cfg          = g->model.cfg;
    const GenerationParams *gp = g->req.gp;
    Tokenizer *tok             = g->model.tok;
    const char *bos_str        = special_token_text(tok, tok->bos_token_id);
    const char *eos_str        = special_token_text(tok, tok->eos_token_id);

    char *formatted = render_chat_template(tok, chat_tmpl, gp, prompt, bos_str, eos_str);
    if (!formatted) return -1;

    if (g_tn_verbose) TN_LOG_DBG("dbg", "[DBG] formatted prompt: %s\n", formatted);

    if (gp->assistant_prefix && gp->assistant_prefix[0]) {
        formatted = append_assistant_prefix(formatted, gp->assistant_prefix);
        if (!formatted) return -1;
    }

    /* A generation prompt that ends in the end-of-sequence token makes the
     * model start a new document: generation stops at 0 tokens. */
    if (tn_prompt_ends_with_token(formatted, eos_str)) tn_template_warn_eos_ending_once();

    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int n_prompt =
        tokenizer_encode(tok, formatted, strlen(formatted), g->prompt.tokens, cfg->seq_len);
    if (t_step) tn_step_timing_add(TN_STEP_1_TOKENIZATION, tn_step_timing_now_ns() - t_step);
    free(formatted);

    if (n_prompt < 0) {
        report_encode_failure("chat-formatted prompt", cfg->seq_len);
        return -1;
    }
    return n_prompt;
}

/* The BOS id a raw prompt starts with, -1 for none. 2026-10-19: honours tokenizer.ggml.add_bos_token:
 * Qwen/SmolLM GGUFs say false, and llama.cpp feeds them no BOS. */
static int raw_bos_id(const Config *cfg, const Tokenizer *tok) {
    if (!tok->add_bos_token) return -1;
    if (cfg->bos_token_id > 0) return cfg->bos_token_id;
    return tok->bos_token_id >= 0 ? tok->bos_token_id : -1;
}

/* Legacy path / --raw-prompt: raw prompt + manual BOS. 2026-10-19: honours
 * tokenizer.ggml.add_bos_token: Qwen/SmolLM GGUFs say false, and llama.cpp feeds them no BOS.
 * Returns the number of prompt tokens, or -1. */
static int build_raw_prompt(GenSession *g, const char *prompt) {
    const Config *cfg          = g->model.cfg;
    const GenerationParams *gp = g->req.gp;
    Tokenizer *tok             = g->model.tok;
    int n_prompt               = 0;
    int bos                    = raw_bos_id(cfg, tok);
    if (bos >= 0) {
        int64_t t_step               = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        g->prompt.tokens[n_prompt++] = bos;
        if (t_step) tn_step_timing_add(TN_STEP_2_BOS_INJECTION, tn_step_timing_now_ns() - t_step);
    }

    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int n_encoded  = tokenizer_encode(tok, prompt, strlen(prompt), &g->prompt.tokens[n_prompt],
                                      cfg->seq_len - n_prompt);
    if (t_step) tn_step_timing_add(TN_STEP_1_TOKENIZATION, tn_step_timing_now_ns() - t_step);
    if (n_encoded < 0) {
        report_encode_failure("prompt", cfg->seq_len);
        return -1;
    }
    n_prompt += n_encoded;
    if (gp->assistant_prefix && gp->assistant_prefix[0]) {
        int n_more = tokenizer_encode(tok, gp->assistant_prefix, strlen(gp->assistant_prefix),
                                      &g->prompt.tokens[n_prompt], cfg->seq_len - n_prompt);
        if (n_more < 0) {
            report_encode_failure("assistant prefix", cfg->seq_len);
            return -1;
        }
        n_prompt += n_more;
    }
    return n_prompt;
}

int gen_session_build_prompt(GenSession *g, const char *prompt) {
    const GenerationParams *gp = g->req.gp;
    Tokenizer *tok             = g->model.tok;

    /* A request's own template (API "chat_template") wins over the model's. */
    const char *chat_tmpl =
        (gp->chat_template && gp->chat_template[0]) ? gp->chat_template : tok->chat_template;
    /* Chat mode ends a turn at any CONTROL token the template never writes
     * (Falcon3's ">>UNUSED_1<<"); raw prompts keep the declared EOG set. The
     * set belongs to this session, so concurrent sessions never share it. */
    g->req.chat_stops =
        tokenizer_chat_stops_new(tok, (chat_tmpl && !gp->raw_prompt) ? chat_tmpl : NULL);
    g->sampling.json_fsm.chat_stops = g->req.chat_stops;

    int n_prompt = (chat_tmpl && !gp->raw_prompt) ? build_chat_prompt(g, chat_tmpl, prompt)
                                                  : build_raw_prompt(g, prompt);
    if (n_prompt < 0) return -1;
    g->prompt.n_tokens = n_prompt;
    return 0;
}
