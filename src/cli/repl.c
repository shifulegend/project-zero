#include "cli/repl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "agent/agent_loop.h"
#include "cli/color.h"
#include "cli/live_stats.h"
#include "cli/md_render.h"
#include "core/log.h"
#include "core/timer.h"
#include "rag/auto_retrieve.h"
#include "rag/auto_save.h"
#include "rag/memory_search.h"
#include "rag/vector_db.h"
#include "speculative/spec_decode.h"
#include "transformer/generate.h"

#define MAX_REPL_LINE 4096

/* Composite generate_with_callback callback: renders markdown/color and
 * tracks a live tok/s indicator. REPL never requests early cancellation. */
typedef struct {
    MdRenderState md;
    TnLiveStats live;
    int is_tty;
    int color_enabled;
} ReplGenContext;

static int repl_token_callback(const char *piece, void *userdata) {
    ReplGenContext *rc = (ReplGenContext *)userdata;
    md_render_feed(&rc->md, piece);
    tn_live_stats_tick(&rc->live);
    tn_live_stats_render(&rc->live, rc->is_tty, rc->color_enabled);
    return 0;
}

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

/* /think and /json: flips a setting and says what it is now. */
static void toggle_mode(bool *flag, const char *label) {
    *flag = !*flag;
    printf("[Config] %s %s\n", label, *flag ? "ENABLED" : "DISABLED");
}

/* Everything a REPL command may need. */
typedef struct {
    Config *p;
    TransformerWeights *w;
    const MoEConfig *mc;
    RunState *s;
    Tokenizer *t;
    ThreadPool *tp;
    CliArgs *args;
    RagContext *rag;
    DraftModel *draft;
} ReplCtx;

static void cmd_help(ReplCtx *c, const char *arg) {
    (void)c;
    (void)arg;
    printf("Commands:\n");
    printf("  /quit               Exit the REPL\n");
    printf("  /context            Show KV cache usage\n");
    printf("  /think              Toggle reasoning mode\n");
    printf("  /json               Toggle JSON-constrained output mode\n");
    printf("  /agent <prompt>     Run agentic tool loop\n");
    printf("  /memory list        List all stored memories (Phase 15)\n");
    printf("  /memory search <q>  Manually search memories (Phase 15)\n");
    printf("  /memory save <text> Manually save a memory (Phase 15)\n");
}

static void cmd_context(ReplCtx *c, const char *arg) {
    (void)arg;
    print_kv_stats(c->s, c->p);
}

static void cmd_think(ReplCtx *c, const char *arg) {
    (void)arg;
    toggle_mode(&c->args->search.enable_reasoning, "Reasoning mode");
}

static void cmd_json(ReplCtx *c, const char *arg) {
    (void)arg;
    toggle_mode(&c->args->sampling.json_mode, "JSON-constrained mode");
}

static void cmd_memory_list(ReplCtx *c, const char *arg) {
    (void)arg;
    print_memory_list(c->rag);
}

/* /memory search <query> */
static void cmd_memory_search(ReplCtx *c, const char *query) {
    if (!c->rag || !c->rag->enabled) {
        printf("[Memory] RAG not enabled.\n");
        return;
    }
    MemoryResult results[5];
    int n = memory_search(&c->rag->db, query, c->p, c->w, &c->rag->emb, c->t, c->tp, results, 5);
    if (n <= 0) {
        printf("[Memory] No results for: %s\n", query);
        return;
    }
    printf("[Memory] %d result(s) for: %s\n", n, query);
    for (int i = 0; i < n; i++) {
        printf("  [%.3f] %s\n", results[i].score, results[i].text ? results[i].text : "");
    }
}

/* /memory save <text> — Phase 15 manual save */
static void cmd_memory_save(ReplCtx *c, const char *save_text) {
    if (!c->rag || !c->rag->enabled) {
        printf("[Memory] RAG not enabled (pass --memory-db <path> to enable).\n");
        return;
    }
    int ret = auto_save_memory(save_text, &c->rag->db, c->p, c->w, &c->rag->emb, c->t, c->tp);
    if (ret == 0)
        printf("[Memory] Saved: \"%s\"\n", save_text);
    else if (ret == 1)
        printf("[Memory] Duplicate detected — entry not saved.\n");
    else
        printf("[Memory] Error saving memory.\n");
}

/* /agent <prompt> */
static void cmd_agent(ReplCtx *c, const char *agent_prompt) {
    run_agent_loop(agent_prompt, c->p, c->w, c->s, c->t, c->tp, c->args->sampling.max_tokens,
                   c->args->sampling.temperature, c->args->sampling.top_p, c->rag);
}

typedef struct {
    const char *text; /* the whole line, or its prefix when `prefix` */
    int prefix;
    void (*run)(ReplCtx *c, const char *arg);
} ReplCommand;

static const ReplCommand k_commands[] = {
    {"/help", 0, cmd_help},
    {"/context", 0, cmd_context},
    {"/think", 0, cmd_think},
    {"/json", 0, cmd_json},
    {"/memory list", 0, cmd_memory_list},
    {"/memory search ", 1, cmd_memory_search},
    {"/memory save ", 1, cmd_memory_save},
    {"/agent ", 1, cmd_agent},
    {"/agent", 0, cmd_agent},
};

/* Runs the command `line` names; 0 when it is not one (it is then a prompt). */
static int run_command(ReplCtx *c, const char *line) {
    for (size_t k = 0; k < sizeof(k_commands) / sizeof(k_commands[0]); k++) {
        const ReplCommand *cmd = &k_commands[k];
        size_t n               = strlen(cmd->text);
        if (cmd->prefix ? strncmp(line, cmd->text, n) != 0 : strcmp(line, cmd->text) != 0) continue;
        cmd->run(c, cmd->prefix ? line + n : "");
        return 1;
    }
    return 0;
}

/* Standard generation — optionally prepend retrieved memories. */
static void generate_reply(ReplCtx *c, const char *line) {
    CliArgs *args = c->args;
    if (c->rag && c->rag->enabled) {
        int injected = auto_retrieve_and_inject(line, &c->rag->db, c->p, c->w, c->s, &c->rag->emb,
                                                c->t, c->tp);
        if (injected > 0 && args->output.verbose) {
            printf("[Memory] Injected %d tokens of relevant context.\n", injected);
        }
    }

    int is_tty        = isatty(fileno(stdout));
    int color_enabled = tn_color_resolve(args->output.color_mode, is_tty, getenv("NO_COLOR"));

    ReplGenContext rc;
    md_render_init(&rc.md, stdout, color_enabled);
    tn_live_stats_init(&rc.live);
    rc.is_tty        = is_tty;
    rc.color_enabled = color_enabled;

    int64_t start_time = timer_now_us();
    if (c->draft) {
        speculative_generate_with_callback(
            c->p, c->w, c->s, c->mc, c->t, c->tp, line, args->sampling.max_tokens,
            args->sampling.temperature, args->sampling.top_p, args->sampling.json_mode, c->draft,
            args->search.spec_length, args->sampling.seed, repl_token_callback, &rc);
    } else {
        GenerationParams gp;
        args_fill_gen_params(args, &gp);
        generate_ex(c->p, c->w, c->s, c->mc, c->t, c->tp, line, &gp, repl_token_callback, &rc);
    }
    md_render_flush(&rc.md);
    if (rc.is_tty && rc.live.count >= 2)
        fputc('\n', stderr); /* move past the live tok/s line (terminal UI, not a log) */
    md_render_free(&rc.md);
    int64_t end_time = timer_now_us();
    printf("\n");

    if (args->output.verbose) {
        printf("\n[Generation took %.2fs]\n", (end_time - start_time) / 1000000.0);
    }
}

static void print_repl_banner(const RagContext *rag) {
    printf("\n--- Project Zero Interactive REPL ---\n");
    if (rag && rag->enabled) {
        printf("[Memory] RAG enabled — %d memories loaded from '%s'\n", rag->db.num_entries,
               rag->db.path);
    }
    printf("Type /quit to exit, /help for commands.\n");
}

/* Prompts and reads one line without its newline; 0 at end of input. */
static int read_repl_line(char *line, size_t cap) {
    printf("\n> ");
    if (!fgets(line, (int)cap, stdin)) return 0;
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
    return 1;
}

void run_repl(Config *p, TransformerWeights *w, const MoEConfig *mc, VisionConfig *vc,
              VisionWeights *vw, VisionProjector *vp, RunState *s, Tokenizer *t, ThreadPool *tp,
              CliArgs *args, RagContext *rag, DraftModel *draft) {
    (void)vc;
    (void)vw;
    (void)vp; /* multimodal stubs — unused */

    ReplCtx ctx = {p, w, mc, s, t, tp, args, rag, draft};
    char line[MAX_REPL_LINE];
    print_repl_banner(rag);

    while (read_repl_line(line, sizeof(line))) {
        if (strlen(line) == 0) continue;
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
        if (run_command(&ctx, line)) continue;
        generate_reply(&ctx, line);
    }
}
