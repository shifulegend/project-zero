#include "cli/repl.h"
#include "cli/timer.h"
#include "transformer/generate.h"
#include "agent/agent_loop.h"
#include "rag/auto_retrieve.h"
#include "rag/auto_save.h"
#include "rag/memory_search.h"
#include "rag/vector_db.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_REPL_LINE 4096

static void print_kv_stats(RunState *s, Config *p) {
    printf("[Context] Used: %d / %d tokens\n", s->current_pos, p->seq_len);
}

/* Print the contents of the vector DB (Phase 15 /memory list) */
static void print_memory_list(RagContext *rag) {
    if (!rag || !rag->enabled) {
        printf("[Memory] RAG not enabled (pass --memory-db <path> to enable).\n");
        return;
    }
    VectorDB *db = &rag->db;
    if (db->num_entries == 0) {
        printf("[Memory] Vector DB is empty.\n");
        return;
    }
    printf("[Memory] %d entries stored:\n", db->num_entries);
    for (int i = 0; i < db->num_entries; i++) {
        const char *text = vector_db_text(db, i);
        printf("  [%d] %s\n", i, text ? text : "(null)");
    }
}

void run_repl(Config *p, TransformerWeights *w,
              const MoEConfig *mc,
              VisionConfig *vc, VisionWeights *vw, VisionProjector *vp,
              RunState *s, Tokenizer *t, ThreadPool *tp,
              CliArgs *args, RagContext *rag) {
    (void)vc; (void)vw; (void)vp; /* multimodal stubs — unused */

    char line[MAX_REPL_LINE];
    printf("\n--- Project Zero Interactive REPL ---\n");
    if (rag && rag->enabled) {
        printf("[Memory] RAG enabled — %d memories loaded from '%s'\n",
               rag->db.num_entries, rag->db.path);
    }
    printf("Type /quit to exit, /help for commands.\n");

    while (1) {
        printf("\n> ");
        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        if (strlen(line) == 0) continue;

        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) {
            break;
        }

        if (strcmp(line, "/help") == 0) {
            printf("Commands:\n");
            printf("  /quit               Exit the REPL\n");
            printf("  /context            Show KV cache usage\n");
            printf("  /think              Toggle reasoning mode\n");
            printf("  /agent <prompt>     Run agentic tool loop\n");
            printf("  /memory list        List all stored memories (Phase 15)\n");
            printf("  /memory search <q>  Manually search memories (Phase 15)\n");
            printf("  /memory save <text> Manually save a memory (Phase 15)\n");
            continue;
        }

        if (strcmp(line, "/context") == 0) {
            print_kv_stats(s, p);
            continue;
        }

        if (strcmp(line, "/think") == 0) {
            args->enable_reasoning = !args->enable_reasoning;
            printf("[Config] Reasoning mode %s\n", args->enable_reasoning ? "ENABLED" : "DISABLED");
            continue;
        }

        /* /memory list */
        if (strcmp(line, "/memory list") == 0) {
            print_memory_list(rag);
            continue;
        }

        /* /memory search <query> */
        if (strncmp(line, "/memory search ", 15) == 0) {
            if (!rag || !rag->enabled) {
                printf("[Memory] RAG not enabled.\n");
            } else {
                const char *query = line + 15;
                MemoryResult results[5];
                int n = memory_search(&rag->db, query, p, w, &rag->emb, t, tp,
                                      results, 5);
                if (n <= 0) {
                    printf("[Memory] No results for: %s\n", query);
                } else {
                    printf("[Memory] %d result(s) for: %s\n", n, query);
                    for (int i = 0; i < n; i++) {
                        printf("  [%.3f] %s\n", results[i].score,
                               results[i].text ? results[i].text : "");
                    }
                }
            }
            continue;
        }

        /* /memory save <text> — Phase 15 manual save */
        if (strncmp(line, "/memory save ", 13) == 0) {
            if (!rag || !rag->enabled) {
                printf("[Memory] RAG not enabled (pass --memory-db <path> to enable).\n");
            } else {
                const char *save_text = line + 13;
                int ret = auto_save_memory(save_text, &rag->db, p, w,
                                           &rag->emb, t, tp);
                if (ret == 0)
                    printf("[Memory] Saved: \"%s\"\n", save_text);
                else if (ret == 1)
                    printf("[Memory] Duplicate detected — entry not saved.\n");
                else
                    printf("[Memory] Error saving memory.\n");
            }
            continue;
        }

        /* /agent <prompt> */
        if (strncmp(line, "/agent ", 7) == 0 || strcmp(line, "/agent") == 0) {
            const char *agent_prompt = (strncmp(line, "/agent ", 7) == 0) ? (line + 7) : "";
            run_agent_loop(agent_prompt, p, w, s, t, tp,
                           args->max_tokens, args->temperature, args->top_p, rag);
            continue;
        }

        /* Standard generation — optionally prepend retrieved memories */
        if (rag && rag->enabled) {
            int injected = auto_retrieve_and_inject(line, &rag->db, p, w, s,
                                                    &rag->emb, t, tp);
            if (injected > 0 && args->verbose) {
                printf("[Memory] Injected %d tokens of relevant context.\n", injected);
            }
        }

        int64_t start_time = timer_now_us();
        generate(p, w, s, mc, t, tp, line, args->max_tokens, args->temperature, args->top_p);
        int64_t end_time = timer_now_us();

        if (args->verbose) {
            printf("\n[Generation took %.2fs]\n", (end_time - start_time) / 1000000.0);
        }

    }
}
