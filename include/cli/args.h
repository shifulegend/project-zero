#ifndef TN_CLI_ARGS_H
#define TN_CLI_ARGS_H

#include <stdbool.h>

#include "api/server_config.h"
#include "cli/color.h"
#include "core/error.h"
#include "core/platform.h"
#include "reasoning/mcts.h"
#include "transformer/gen_params.h"
#include "update/update_check.h"

#define CLI_MAX_CORS_ORIGINS 16

/* Files and text that go in: model, tokenizer, prompt, images, adapters, templates. */
typedef struct {
    char *model_path;
    char *tokenizer_path;
    char *prompt;
    char *image_path;
    char *vision_path;        /* Phase 34: path to vision.bin encoder weights (NULL = no image) */
    char *proj_path;          /* Phase 34: path to projector.bin weights (NULL = no image) */
    char *memory_db_path;     /* Phase 15: path to vector DB file (NULL = disable RAG) */
    char *draft_model_path;   /* --draft-model <path>: GGUF draft model. NULL (default)
                                  = disabled -- the ONLY way to enable speculative
                                  decoding; never a default/embedded path.         */
    char *lora_path;          /* --lora <path>: .lora.bin adapter. NULL (default) =
                                  disabled -- the ONLY way to enable a LoRA adapter;
                                  never a default/embedded path.                    */
    char *system_prompt;      /* --system-prompt <text>: NULL = none          */
    char *chat_template_file; /* --chat-template-file <path>: replaces the
                                  model's tokenizer.chat_template (CLI and
                                  --server); NULL = use the GGUF template     */
} CliInput;

/* How a reply is generated: sampler chain, limits and stop sequences (also the server-side defaults). */
typedef struct {
    float temperature;
    float top_p;
    int max_tokens;
    bool json_mode;  /* Phase 20: --json forces syntactically valid JSON output */
    bool raw_prompt; /* --raw-prompt: no chat template, prompt fed verbatim */
    bool ignore_eos; /* --ignore-eos: generate exactly --max-tokens */
    int seed;
    int top_k;                /* --top-k <N>: 0 = disabled (default)          */
    float min_p;              /* --min-p <p>: 0 = disabled (default)          */
    float repeat_penalty;     /* --repeat-penalty <r>: 1 = disabled (default) */
    float presence_penalty;   /* --presence-penalty <v>: 0 = disabled         */
    float frequency_penalty;  /* --frequency-penalty <v>: 0 = disabled        */
    int repeat_last_n;        /* --repeat-last-n <N>: penalty window (64)     */
    char *stop[GEN_MAX_STOP]; /* --stop <text> (repeatable)                   */
    int n_stop;
    int max_tokens_limit; /* --max-tokens-limit <N>: per-request cap in --server mode;
                                  0 = model context length                      */
    int chat_format;      /* --chat-format: -1 auto, else ChatTemplateType */
    int best_of;          /* --best-of <N>: candidates, keep most likely (1) */
    int best_of_max;      /* --best-of-max <N>: per-request cap (server, 8) */
} CliSampling;

/* Perplexity, accuracy probe and tensor dump options. */
typedef struct {
    char *perplexity_file; /* --perplexity <file>: evaluate PPL, then exit */
    int ppl_ctx;           /* --ppl-ctx <n>: tokens per chunk (512) */
    int ppl_chunks;        /* --ppl-chunks <n>: 0 = all */
    int act_quant;         /* --act-quant: TnActQuant (0 auto, 1 int8, 2 float) */
    double
        quality_tolerance; /* --quality-tolerance: allowed perplexity increase in percent (0 = no bar) */
    int self_check_tokens;   /* --self-check-tokens: length of the accuracy probe */
    bool self_check;         /* --self-check: run the accuracy probe, print, exit */
    char *dump_tensors_path; /* --dump-tensors FILE: write intermediate tensors CSV  */
} CliEval;

/* Threads, SIMD, kernels, KV cache and prompt cache. */
typedef struct {
    int num_threads;
    int rope_layout;         /* --rope-layout auto|norm|neox: -1 = from GGUF arch */
    int classifier_override; /* -1 = auto (BF16 default), 0 = BF16, 1 = INT8, 2 = INT4,
                                  3 = auto-fast (let engine pick fastest quantized cls) */
    char *simd_override;     /* NULL = auto, or "avx2", "avx512f", "vnni", "scalar" */
    bool calibrate;          /* --calibrate: force re-calibration of optimal settings */
    bool no_calibrate; /* --no-calibrate / PZ_NO_CALIBRATE: never load, run or save a calibration */
    char *attn_kernel; /* --attn-kernel standard|tiled (standard)       */
    int attn_tile;     /* --attn-tile <N>: tiled kernel tile (0 = L1d)  */
    int attn_parallel_min_ctx;   /* --attn-parallel-min-ctx <N> (256, 0=off)  */
    int ctx_size;                /* --ctx-size <N>: context length (0 = auto)     */
    int kv_type;                 /* --kv-type f32|i8|i4 (KVCacheType, default f32) */
    char *kv_offload_dir;        /* --kv-offload <dir>: disk-backed KV (Phase 31) */
    float kv_ram_fraction;       /* --kv-ram-fraction <f>: KV RAM budget (0=0.60) */
    int prompt_cache_mb;         /* --prompt-cache-mb <N>: Phase 26 store (256)   */
    int prompt_cache_min_tokens; /* --prompt-cache-min-tokens <N> (32)       */
    bool prompt_cache;           /* --prompt-cache on|off: server KV prefix reuse
                                  default (on)                                  */
} CliPerf;

/* Update notifications (update/update_check.h). */
typedef struct {
    bool check_update;     /* --check-update: check now, print, exit (no --model needed) */
    bool no_update_check;  /* --no-update-check */
    char *update_repo;     /* --update-repo owner/name */
    char *update_api;      /* --update-api https://... (GitHub Enterprise / mirror) */
    int update_interval_h; /* --update-interval <hours>, -1 = default 24, 0 = off */
    int update_channel;    /* --update-channel auto|stable|prerelease */
} CliUpdate;

/* HTTP server: port, CORS, auth, limits, queueing, web UI. */
typedef struct {
    int server_mode;   /* --server: run as OpenAI-compatible HTTP server        */
    int server_port;   /* --port <N>: listen port (default: 8080)              */
    bool cors_enabled; /* --cors: enable CORS response headers                 */
    char *cors_origins[CLI_MAX_CORS_ORIGINS]; /* --cors-origin <origin> (repeatable)    */
    int num_cors_origins;
    char *api_key;            /* --api-key <key>: require Authorization: Bearer <key> */
    bool metrics_enabled;     /* --metrics: enable GET /metrics                       */
    char *static_dir;         /* --static-dir <path>: serve UI from disk (dev mode)   */
    WebUiMode web_ui_mode;    /* --web-ui <auto|on|off>: default auto                 */
    bool no_request_template; /* --no-request-template: ignore API "chat_template" */
    bool no_config_write;     /* --no-config-write: refuse POST /v1/config/defaults */
    char *ui_title;           /* --ui-title <name>: web UI brand name          */
    int ui_logo;              /* --ui-logo ascii|text (0 = ascii, default)     */
    int queue_size;           /* --queue-size <N>: waiting chat requests (0)   */
    int queue_timeout_s;      /* --queue-timeout <s>: max wait (120)           */
    int rate_limit;           /* --rate-limit <n>: req/s on non-generation routes (20) */
    int client_timeout_s;     /* --client-timeout <s>: socket I/O timeout (30) */
    int max_connections;      /* --max-connections <N>: open connections (64)  */
    int parallel;             /* --parallel <N>: chat requests served at once (1) */
    int parallel_batch;       /* --parallel-batch <N>: rows per batched pass (128) */
} CliServer;

/* Console output: verbosity, colour, log format, --version. */
typedef struct {
    bool verbose;
    bool show_version;      /* --version/-v: print version + detected SIMD backend, exit 0 */
    TnColorMode color_mode; /* --color <auto|always|never>: default auto            */
    int log_json;           /* --log-format json: diagnostics as JSON lines on stderr   */
    int log_level; /* --log-level <debug|info|warn|error>; -1 = default (info, debug with --verbose) */
} CliOutput;

/* Reasoning, tree search (MCTS), judge model and speculative decoding. */
typedef struct {
    bool enable_reasoning;
    int spec_length;  /* --spec-length <N>: tokens drafted per round.
                                  Default 5. Only used when draft_model_path is set. */
    MctsOptions mcts; /* --mcts N (simulations; 0 = off) --mcts-expand --mcts-depth
                                  --mcts-step-tokens --mcts-c --mcts-temperature --mcts-delim */
    int mcts_max;     /* --mcts-max <N>: per-request simulation cap (server) */
    char *prm_path;   /* --prm <gguf>: process reward model (judge) */
    char *prm_format; /* --prm-format <s>: judge input with {question} {steps} */
    char *prm_good;   /* --prm-good <s> / --prm-bad <s>: judge answer strings */
    char *prm_bad;
    char mcts_delim_buf[16]; /* unescaped --mcts-delim */
} CliSearch;

/* Output watermarking. */
typedef struct {
    bool enabled;           /* --watermark                                   */
    char *watermark_key;    /* --watermark-key <passphrase>                  */
    float watermark_gamma;  /* --watermark-gamma <g> (default 0.25)          */
    float watermark_delta;  /* --watermark-delta <d> (default 2.0)           */
    char *watermark_detect; /* --watermark-detect <text>: score text, exit   */
} CliWatermark;

/* Distributed layers (RPC worker and remote layers). */
typedef struct {
    int rpc_worker_port; /* --rpc-worker <port>: serve --layers, exit      */
    char *worker_layers; /* --layers A-B: range a worker serves           */
    bool rpc_bind_all;   /* --rpc-bind-all: worker listens on 0.0.0.0     */
    char *remote_layers; /* --remote-layers host:port@A-B[,...]           */
    char *plan_layers;   /* --plan-layers GB,GB,...: print a RAM split    */
} CliDist;

// CLI arguments, grouped by what they configure (at most 25 fields per struct).
typedef struct {
    CliInput input;
    CliSampling sampling;
    CliEval eval;
    CliPerf perf;
    CliUpdate update;
    CliServer server;
    CliOutput output;
    CliSearch search;
    CliWatermark watermark;
    CliDist dist;
} CliArgs;

// Parse CLI arguments into the CliArgs struct
TernaryError parse_args(CliArgs *args, int argc, char **argv);

// Builds the GenerationParams a CLI run (or the server defaults) starts from.
void args_fill_gen_params(const CliArgs *args, GenerationParams *gp);

/* Phase 38: update-check options from the --update-* / --no-update-check flags. */
void args_fill_update_options(const CliArgs *args, UpdateOptions *o);

// Print usage help
void print_usage(const char *prog_name);

#endif // TN_CLI_ARGS_H
