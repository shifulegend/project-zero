#include "agent/agent_loop.h"
#include "agent/tool_interceptor.h"
#include "agent/cmd_exec.h"
#include "agent/output_inject.h"

#include "transformer/forward.h"
#include "math/simd_dispatch.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "tokenizer/tokenizer.h"
#include "cli/timer.h"

/* Phase 15 RAG */
#include "rag/auto_save.h"
#include "rag/memory_search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    if (!argv) { free(copy); return NULL; }
    int argc = 0;
    char *saveptr = NULL;
    char *tok_s = strtok_r(copy, " \t\n", &saveptr);
    while (tok_s && argc < 63) {
        argv[argc++] = strdup(tok_s);
        tok_s = strtok_r(NULL, " \t\n", &saveptr);
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
static void handle_search_memory(const char *query,
                                  RagContext *rag,
                                  const Config *cfg,
                                  const TransformerWeights *w,
                                  RunState *s,
                                  Tokenizer *tok,
                                  ThreadPool *tp) {
    MemoryResult results[3];
    int n = memory_search(&rag->db, query, cfg, w, &rag->emb,
                          tok, tp, results, 3);
    if (n <= 0) {
        printf("[AGENT] <search_memory>: no relevant memories found.\n");
        inject_text_into_kv("<result>No relevant memories found.</result>",
                            cfg, w, s, tok, tp);
        return;
    }

    /* Format results and inject */
    char buf[4096];
    int off = 0;
    off += snprintf(buf + off, (int)sizeof(buf) - off,
                    "<result>\nMemory search results for \"%s\":\n", query);
    for (int i = 0; i < n && off < (int)sizeof(buf) - 128; i++) {
        off += snprintf(buf + off, (int)sizeof(buf) - off,
                        "  [%.2f] %s\n", results[i].score,
                        results[i].text ? results[i].text : "");
    }
    off += snprintf(buf + off, (int)sizeof(buf) - off, "</result>");
    (void)off;

    printf("[AGENT] <search_memory> results (%d found):\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%.3f] %s\n", results[i].score,
               results[i].text ? results[i].text : "");
    }

    inject_text_into_kv(buf, cfg, w, s, tok, tp);
}

/* Handle <save_memory> tag: embed + deduplicate + store */
static void handle_save_memory(const char *text,
                                RagContext *rag,
                                const Config *cfg,
                                const TransformerWeights *w,
                                RunState *s,
                                Tokenizer *tok,
                                ThreadPool *tp) {
    int result = auto_save_memory(text, &rag->db, cfg, w,
                                  &rag->emb, tok, tp);
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

int run_agent_loop(const char *prompt,
                   const Config *cfg,
                   const TransformerWeights *w,
                   RunState *s,
                   Tokenizer *tok,
                   ThreadPool *tp,
                   int max_tokens,
                   float temperature,
                   float top_p,
                   RagContext *rag) {
    if (!cfg || !w || !s || !tok) return -1;
    if (!prompt) prompt = "";

    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return -1;

    int n_prompt = 0;
    /* Inject BOS token if the tokenizer or config provides one */
    {
        int bos = (tok->bos_token_id >= 0) ? tok->bos_token_id
                : (cfg->bos_token_id > 0)  ? cfg->bos_token_id
                : -1;
        if (bos >= 0) {
            prompt_tokens[n_prompt++] = bos;
        }
    }

    int n_encoded = tokenizer_encode(tok, prompt, strlen(prompt), &prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
    if (n_encoded < 0) {
        free(prompt_tokens);
        return -1;
    }
    n_prompt += n_encoded;

    unsigned long long rng_state;
    rng_seed(&rng_state, (unsigned long long)time(NULL));

    /* Initialize sliding window with prompt pinned */
    sw_init(&s->sw, s->max_seq_len, n_prompt);

    int prev_token = -1;
    int next = 0;

    ti_init();
    char tag_buf[4096];

    for (int pos = 0; pos < max_tokens && pos < cfg->seq_len; pos++) {
        int token = (pos < n_prompt) ? prompt_tokens[pos] : next;
        float *logits = transformer_forward(token, pos, cfg, w, s, NULL, tp);

        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            if (temperature <= 0.0f || temperature < 1e-6f) {
                next = sample_argmax(logits, cfg->vocab_size);
            } else {
                apply_temperature(logits, cfg->vocab_size, temperature);
                if (top_p < 1.0f && top_p > 0.0f) {
                    next = sample_top_p(logits, cfg->vocab_size, top_p, &rng_state);
                } else {
                    tn_softmax(logits, cfg->vocab_size);
                    float r = rng_float(&rng_state);
                    float cdf = 0.0f;
                    next = cfg->vocab_size - 1;
                    for (int i = 0; i < cfg->vocab_size; i++) {
                        cdf += logits[i];
                        if (r < cdf) { next = i; break; }
                    }
                }
            }
        }

        if (pos >= n_prompt - 1) {
            const char *piece = tokenizer_decode(tok, prev_token, next);
            if (piece) {
                TI_Tag tag = ti_process_piece(piece, tag_buf, sizeof(tag_buf));

                if (tag == TI_TAG_NONE) {
                    printf("%s", piece);
                    fflush(stdout);

                } else if (tag == TI_TAG_EXEC) {
                    printf("\n[AGENT] <exec> detected: %s\n", tag_buf);
                    if (user_approval_prompt(tag_buf)) {
                        int argc = 0;
                        char **argv = build_argv_from_command(tag_buf, &argc);
                        if (!argv) {
                            printf("[AGENT] Failed to parse command\n");
                        } else {
                            char exec_out[8192] = {0};
                            ExecResult er = execute_command(argv, 5, exec_out, sizeof(exec_out));
                            printf("[AGENT] Command exit=%d timed_out=%d\n", er.exit_code, er.timed_out);
                            if (er.stdout_len > 0) {
                                printf("[AGENT] Output:\n%s\n", exec_out);
                                inject_text_into_kv(exec_out, cfg, w, s, tok, tp);
                            }
                            free_argv(argv, argc);
                        }
                    } else {
                        printf("[AGENT] Execution denied by user.\n");
                    }

                } else if (tag == TI_TAG_THINK) {
                    /* Hidden reasoning — logged but not printed to user */
                    (void)tag_buf;

                } else if (tag == TI_TAG_SAVE_MEMORY) {
                    if (rag && rag->enabled) {
                        handle_save_memory(tag_buf, rag, cfg, w, s, tok, tp);
                    } else {
                        printf("[AGENT] <save_memory> (RAG disabled): %s\n", tag_buf);
                    }

                } else if (tag == TI_TAG_SEARCH_MEMORY) {
                    if (rag && rag->enabled) {
                        handle_search_memory(tag_buf, rag, cfg, w, s, tok, tp);
                    } else {
                        printf("[AGENT] <search_memory> (RAG disabled): %s\n", tag_buf);
                    }

                } else {
                    printf("[AGENT] Tag %d: %s\n", tag, tag_buf);
                }

                fflush(stdout);
            }
        }

        prev_token = next;
        if (pos < n_prompt - 1) next = prompt_tokens[pos + 1];

        /* EOS check: stop on any token in the tokenizer's eos_list.
         * No hardcoded token IDs — all EOS come from vocab scan / GGUF metadata. */
        if (pos >= n_prompt - 1) {
            int is_eos = 0;
            for (int ei = 0; ei < tok->n_eos && !is_eos; ei++) {
                if (next == tok->eos_list[ei]) is_eos = 1;
            }
            if (is_eos) break;
        }
    }

    free(prompt_tokens);
    printf("\n");
    fflush(stdout);
    return 0;
}
