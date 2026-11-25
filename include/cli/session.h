#ifndef TN_CLI_SESSION_H
#define TN_CLI_SESSION_H

/* One command-line run of the engine, split into phases (src/cli/session_*.c). main() creates a
 * PzSession, runs the phases in order and releases it; every phase reads and fills this struct
 * instead of sharing locals. The struct is zeroed at creation, which is what lets
 * session_release() free whatever stage was reached. */

#include <stdbool.h>

#include "cli/args.h"
#include "core/config.h"
#include "core/gguf_reader.h"
#include "core/lora.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "distributed/dist_plan.h"
#include "memory/mapped_file.h"
#include "rag/rag_context.h"
#include "reasoning/prm.h"
#include "runtime/gguf_loader.h"
#include "runtime/run_state.h"
#include "speculative/draft_model.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"
#include "update/update_check.h"

/* Version is injected unquoted by the build (-DPZ_VERSION, see Makefile) and stringified here;
 * falls back to "dev" for builds that don't pass it (e.g. plain `make debug`). */
#ifndef PZ_VERSION
#define PZ_VERSION dev
#endif
#define PZ_STRINGIFY2(x) #x
#define PZ_STRINGIFY(x) PZ_STRINGIFY2(x)
#define PZ_VERSION_STR PZ_STRINGIFY(PZ_VERSION)

/* Background release check and whether to print its result after the output. */
typedef struct {
    UpdateService svc; /* zeroed = unconfigured */
    int notice;        /* CLI only; the server shows it in the web UI */
} PzUpdateState;

typedef struct {
    CliArgs args;
    int exit_code;
    int stdout_is_tty;

    /* hardware */
    ThreadPool *tp;

    /* model (valid once session_load_model() returned 0) */
    int is_gguf;
    Config p;
    TransformerWeights w;
    MoEConfig mc;
    GGUFHeader gguf_hdr;
    GGUFWeightStore *gguf_store;
    MappedFile mf;
    RunState *s;

    /* tokenizer */
    Tokenizer t;
    int chat_template_from_file;
    int chat_template_builtin;

    /* optional parts (each freed only when its flag says it was set up) */
    DraftModel draft;
    bool draft_active;
    PrmModel prm;
    LoRAWeights lora;
    bool lora_active;
    RagContext rag;
    int rag_ok;
    DistPlan dist;
    int dist_active;
    PzUpdateState update;
} PzSession;

/* Phases, in the order main() runs them. Those returning int give 0 for success and non-zero
 * when the run must stop (the reason is already printed). */
void session_apply_log_options(PzSession *ses); /* --log-format, --log-level, --verbose */
int session_early_command(PzSession *ses); /* -1: none; else the exit status of --version etc. */
void session_open_dump(PzSession *ses);
int session_detect_format(PzSession *ses);
int session_tune_hardware(PzSession *ses);
int session_load_model(PzSession *ses);
int session_setup_tokenizer(PzSession *ses);
int session_load_extras(PzSession *ses);
void session_apply_runtime_options(PzSession *ses);
int session_distributed(PzSession *ses); /* non-zero: the run is finished, skip to release */
void session_start_update_check(PzSession *ses);
void session_accuracy_checks(PzSession *ses);
void session_execute(PzSession *ses);
void session_run_server(PzSession *ses); /* --server: configure, serve until SIGINT/SIGTERM */
void session_finish(PzSession *ses);
void session_release(PzSession *ses);

#endif
