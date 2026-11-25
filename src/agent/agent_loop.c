#include "agent/agent_loop.h"

#include "agent/cmd_exec.h"
#include "agent/tool_interceptor.h"
#include "core/log.h"
#include "core/timer.h"
#include "math/simd_core_ops.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "tokenizer/tokenizer.h"
#include "transformer/forward.h"
#include "transformer/output_inject.h"

/* Phase 15 RAG */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rag/auto_save.h"
#include "rag/memory_search.h"

/* Minimal agent loop: replicates the core of generate() but intercepts
 * decoded pieces and reacts to tags such as <exec>...</exec>.
 */

/* Helper: split a command string into argv[] using strtok_r (re-entrant).
 * Caller must free returned array and its elements. */
static char **build_argv_from_command(const char *cmd, int *out_argc) {
    if (!cmd) return NULL;
    char *copy = strdup(cmd);
    if (!copy) return NULL;
    char **argv = (char **)calloc(64, sizeof(char *));
    if (!argv) {
        free(copy);
        return NULL;
    }
    int argc      = 0;
    char *saveptr = NULL;
    char *tok_s   = strtok_r(copy, " \t\n", &saveptr);
    while (tok_s && argc < 63) {
        argv[argc++] = strdup(tok_s);
        tok_s        = strtok_r(NULL, " \t\n", &saveptr);
    }
    argv[argc] = NULL;
    free(copy);
    *out_argc = argc;
    return argv;
}

static void free_argv(char **argv, int argc) {
    if (!argv) return;
    for (int i = 0; i < argc; i++) free(argv[i]);
    free(argv);
}

/* Handle <search_memory> tag: search RAG and inject results into KV cache */
static void handle_search_memory(const char *query, RagContext *rag, const Config *cfg,
                                 const TransformerWeights *w, RunState *s, Tokenizer *tok,
                                 ThreadPool *tp) {
    MemoryResult results[3];
    int n = memory_search(&rag->db, query, cfg, w, &rag->emb, tok, tp, results, 3);
    if (n <= 0) {
        printf("[AGENT] <search_memory>: no relevant memories found.\n");
        inject_text_into_kv("<result>No relevant memories found.</result>", cfg, w, s, tok, tp);
        return;
    }

    /* Format results and inject */
    char buf[4096];
    int off = 0;
    off += snprintf(buf + off, (int)sizeof(buf) - off,
                    "<result>\nMemory search results for \"%s\":\n", query);
    for (int i = 0; i < n && off < (int)sizeof(buf) - 128; i++) {
        off += snprintf(buf + off, (int)sizeof(buf) - off, "  [%.2f] %s\n", results[i].score,
                        results[i].text ? results[i].text : "");
    }
    off += snprintf(buf + off, (int)sizeof(buf) - off, "</result>");
    (void)off;

    printf("[AGENT] <search_memory> results (%d found):\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%.3f] %s\n", results[i].score, results[i].text ? results[i].text : "");
    }

    inject_text_into_kv(buf, cfg, w, s, tok, tp);
}

/* Handle <save_memory> tag: embed + deduplicate + store */
static void handle_save_memory(const char *text, RagContext *rag, const Config *cfg,
                               const TransformerWeights *w, RunState *s, Tokenizer *tok,
                               ThreadPool *tp) {
    int result = auto_save_memory(text, &rag->db, cfg, w, &rag->emb, tok, tp);
    const char *status_msg;
    if (result == 0) {
        printf("[AGENT] <save_memory>: saved.\n");
        status_msg = "<result>Memory saved.</result>";
    } else if (result == 1) {
        printf("[AGENT] <save_memory>: duplicate detected, skipped.\n");
        status_msg = "<result>Memory already known (duplicate skipped).</result>";
    } else {
        printf("[AGENT] <save_memory>: error saving memory.\n");
        status_msg = "<result>Memory save failed.</result>";
    }
    inject_text_into_kv(status_msg, cfg, w, s, tok, tp);
}

int agent_sample_next(float *logits, int vocab_size, float temperature, float top_p,
                      unsigned long long *rng_state) {
    if (temperature <= 0.0f || temperature < 1e-6f) return sample_argmax(logits, vocab_size);
    apply_temperature(logits, vocab_size, temperature);
    if (top_p < 1.0f && top_p > 0.0f) return sample_top_p(logits, vocab_size, top_p, rng_state);
    tn_softmax(logits, vocab_size);
    float r   = rng_float(rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}

/* What the tag handlers need. */
typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    RunState *s;
    Tokenizer *tok;
    ThreadPool *tp;
    RagContext *rag;
} AgentCtx;

/* <exec>: ask the user, run the command, show and inject its output. */
static void handle_exec_tag(const AgentCtx *c, const char *cmd) {
    printf("\n[AGENT] <exec> detected: %s\n", cmd);
    if (!user_approval_prompt(cmd)) {
        printf("[AGENT] Execution denied by user.\n");
        return;
    }
    int argc    = 0;
    char **argv = build_argv_from_command(cmd, &argc);
    if (!argv) {
        printf("[AGENT] Failed to parse command\n");
        return;
    }
    char exec_out[8192] = {0};
    ExecResult er       = execute_command(argv, 5, exec_out, sizeof(exec_out));
    printf("[AGENT] Command exit=%d timed_out=%d\n", er.exit_code, er.timed_out);
    if (er.stdout_len > 0) {
        printf("[AGENT] Output:\n%s\n", exec_out);
        inject_text_into_kv(exec_out, c->cfg, c->w, c->s, c->tok, c->tp);
    }
    free_argv(argv, argc);
}

/* Reacts to one detected tag (anything but TI_TAG_NONE). */
static void handle_tag(const AgentCtx *c, TI_Tag tag, const char *text) {
    int rag_on = c->rag && c->rag->enabled;
    switch (tag) {
        case TI_TAG_EXEC:
            handle_exec_tag(c, text);
            break;
        case TI_TAG_THINK:
            break; /* Hidden reasoning — logged but not printed to user */
        case TI_TAG_SAVE_MEMORY:
            if (rag_on)
                handle_save_memory(text, c->rag, c->cfg, c->w, c->s, c->tok, c->tp);
            else
                printf("[AGENT] <save_memory> (RAG disabled): %s\n", text);
            break;
        case TI_TAG_SEARCH_MEMORY:
            if (rag_on)
                handle_search_memory(text, c->rag, c->cfg, c->w, c->s, c->tok, c->tp);
            else
                printf("[AGENT] <search_memory> (RAG disabled): %s\n", text);
            break;
        default:
            printf("[AGENT] Tag %d: %s\n", tag, text);
            break;
    }
}

/* Feeds one decoded piece through the tag parser: plain text is printed, a tag is acted on. */
static void emit_piece(const AgentCtx *c, TiParser *ti, const char *piece) {
    char tag_buf[4096];
    TI_Tag tag = ti_process_piece(ti, piece, tag_buf, sizeof(tag_buf));
    if (tag == TI_TAG_NONE) {
        printf("%s", piece);
        fflush(stdout);
    } else {
        handle_tag(c, tag, tag_buf);
    }
    fflush(stdout);
}

/* BOS (if the tokenizer or config provides one) plus the encoded prompt; the count, or -1. */
static int build_prompt_tokens(const Config *cfg, Tokenizer *tok, const char *prompt, int *out) {
    int n_prompt = 0;
    int bos      = (tok->bos_token_id >= 0)  ? tok->bos_token_id
                   : (cfg->bos_token_id > 0) ? cfg->bos_token_id
                                             : -1;
    if (bos >= 0) out[n_prompt++] = bos;
    int n_encoded =
        tokenizer_encode(tok, prompt, strlen(prompt), &out[n_prompt], cfg->seq_len - n_prompt);
    return n_encoded < 0 ? -1 : n_prompt + n_encoded;
}

static int agent_args_ok(const Config *cfg, const TransformerWeights *w, const RunState *s,
                         const Tokenizer *tok) {
    return cfg && w && s && tok;
}

/* State of one run of the loop. */
typedef struct {
    AgentCtx ctx;
    TiParser ti;
    const int *prompt_tokens;
    int n_prompt;
    float temperature, top_p;
    unsigned long long rng_state;
    int prev_token;
    int next;
} AgentRun;

/* One position: feed the token, pick the next one and, once past the prompt, show it. Returns 1
 * when generation should stop (an end-of-generation token). */
static int agent_step(AgentRun *r, int pos) {
    const AgentCtx *c = &r->ctx;
    int token         = (pos < r->n_prompt) ? r->prompt_tokens[pos] : r->next;
    float *logits     = transformer_forward(token, pos, c->cfg, c->w, c->s, NULL, c->tp);
    int generating    = pos >= r->n_prompt - 1;

    if (!generating)
        r->next = r->prompt_tokens[pos + 1];
    else
        r->next =
            agent_sample_next(logits, c->cfg->vocab_size, r->temperature, r->top_p, &r->rng_state);

    if (generating) {
        const char *piece = tokenizer_decode(c->tok, r->prev_token, r->next);
        if (piece) emit_piece(c, &r->ti, piece);
    }

    r->prev_token = r->next;
    if (!generating) r->next = r->prompt_tokens[pos + 1];

    /* EOS check: stop on any token in the tokenizer's eos_list.
     * No hardcoded token IDs — all EOS come from vocab scan / GGUF metadata. */
    return generating && tokenizer_is_eog(c->tok, r->next, NULL); /* raw prompt: declared EOG set */
}

int run_agent_loop(const char *prompt, const Config *cfg, const TransformerWeights *w, RunState *s,
                   Tokenizer *tok, ThreadPool *tp, int max_tokens, float temperature, float top_p,
                   RagContext *rag) {
    if (!agent_args_ok(cfg, w, s, tok)) return -1;
    if (!prompt) prompt = "";

    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return -1;
    int n_prompt = build_prompt_tokens(cfg, tok, prompt, prompt_tokens);
    if (n_prompt < 0) {
        free(prompt_tokens);
        return -1;
    }

    AgentRun run = {.ctx           = {cfg, w, s, tok, tp, rag},
                    .prompt_tokens = prompt_tokens,
                    .n_prompt      = n_prompt,
                    .temperature   = temperature,
                    .top_p         = top_p,
                    .prev_token    = -1,
                    .next          = 0};
    rng_seed(&run.rng_state, (unsigned long long)time(NULL));

    /* Initialize sliding window with prompt pinned */
    sw_init(&s->sw, s->max_seq_len, n_prompt);
    ti_init(&run.ti);

    for (int pos = 0; pos < max_tokens && pos < cfg->seq_len; pos++) {
        if (agent_step(&run, pos)) break;
    }

    free(prompt_tokens);
    printf("\n");
    fflush(stdout);
    return 0;
}
