#ifndef PROJECT_ZERO_ARGS_H
#define PROJECT_ZERO_ARGS_H

#include "reasoning/mcts.h"
#include <stdbool.h>
#include "core/platform.h"
#include "core/error.h"
#include "cli/color.h"
#include "api/server_config.h"
#include "transformer/gen_params.h"
#include "update/update_check.h"

#define CLI_MAX_CORS_ORIGINS 16

// CLI Arguments Structure
typedef struct {
    char *model_path;
    char *tokenizer_path;
    char *prompt;
    char *image_path;
    char *vision_path;     /* Phase 34: path to vision.bin encoder weights (NULL = no image) */
    char *proj_path;       /* Phase 34: path to projector.bin weights (NULL = no image) */
    char *memory_db_path;  /* Phase 15: path to vector DB file (NULL = disable RAG) */
    float temperature;
    float top_p;
    int max_tokens;
    int num_threads;
    bool enable_reasoning;
    bool json_mode;           /* Phase 20: --json forces syntactically valid JSON output */
    bool raw_prompt;          /* --raw-prompt: no chat template, prompt fed verbatim */
    bool ignore_eos;          /* --ignore-eos: generate exactly --max-tokens */
    int  rope_layout;         /* --rope-layout auto|norm|neox: -1 = from GGUF arch */
    char *perplexity_file;    /* --perplexity <file>: evaluate PPL, then exit */
    int  ppl_ctx;             /* --ppl-ctx <n>: tokens per chunk (512) */
    int  ppl_chunks;          /* --ppl-chunks <n>: 0 = all */
    bool verbose;
    int seed;
    int classifier_override;  /* -1 = auto (BF16 default), 0 = BF16, 1 = INT8, 2 = INT4,
                                  3 = auto-fast (let engine pick fastest quantized cls) */
    char *simd_override;      /* NULL = auto, or "avx2", "avx512f", "vnni", "scalar" */
    bool calibrate;           /* --calibrate: force re-calibration of optimal settings */
    bool show_version;        /* --version/-v: print version + detected SIMD backend, exit 0 */
    /* Phase 38: update notifications (update/update_check.h) */
    bool check_update;        /* --check-update: check now, print, exit (no --model needed) */
    bool no_update_check;     /* --no-update-check */
    char *update_repo;        /* --update-repo owner/name */
    char *update_api;         /* --update-api https://... (GitHub Enterprise / mirror) */
    int  update_interval_h;   /* --update-interval <hours>, -1 = default 24, 0 = off */
    int  update_channel;      /* --update-channel auto|stable|prerelease */
    char *dump_tensors_path;  /* --dump-tensors FILE: write intermediate tensors CSV  */
    /* Phase 21: API server */
    int   server_mode;        /* --server: run as OpenAI-compatible HTTP server        */
    int   server_port;        /* --port <N>: listen port (default: 8080)              */
    /* Phase 22: API hardening */
    bool  cors_enabled;        /* --cors: enable CORS response headers                 */
    char *cors_origins[CLI_MAX_CORS_ORIGINS]; /* --cors-origin <origin> (repeatable)    */
    int   num_cors_origins;
    char *api_key;             /* --api-key <key>: require Authorization: Bearer <key> */
    bool  metrics_enabled;     /* --metrics: enable GET /metrics                       */
    /* Phase 22.2: web UI */
    char     *static_dir;      /* --static-dir <path>: serve UI from disk (dev mode)   */
    WebUiMode web_ui_mode;     /* --web-ui <auto|on|off>: default auto                 */
    /* Phase 22.3: CLI/REPL polish */
    TnColorMode color_mode;    /* --color <auto|always|never>: default auto            */
    /* Phase 18: speculative decoding */
    char *draft_model_path;   /* --draft-model <path>: GGUF draft model. NULL (default)
                                  = disabled -- the ONLY way to enable speculative
                                  decoding; never a default/embedded path.         */
    int   spec_length;        /* --spec-length <N>: tokens drafted per round.
                                  Default 5. Only used when draft_model_path is set. */
    /* Phase 19: LoRA adapters */
    char *lora_path;          /* --lora <path>: .lora.bin adapter. NULL (default) =
                                  disabled -- the ONLY way to enable a LoRA adapter;
                                  never a default/embedded path.                    */
    /* Sampler chain + generation knobs (sampling/sampler_chain.h). In
     * --server mode these become the server-side defaults every API
     * request (and the web UI) starts from and may override. */
    int    top_k;              /* --top-k <N>: 0 = disabled (default)          */
    float  min_p;              /* --min-p <p>: 0 = disabled (default)          */
    float  repeat_penalty;     /* --repeat-penalty <r>: 1 = disabled (default) */
    float  presence_penalty;   /* --presence-penalty <v>: 0 = disabled         */
    float  frequency_penalty;  /* --frequency-penalty <v>: 0 = disabled        */
    int    repeat_last_n;      /* --repeat-last-n <N>: penalty window (64)     */
    char  *system_prompt;      /* --system-prompt <text>: NULL = none          */
    bool   no_request_template; /* --no-request-template: ignore API "chat_template" */
    bool   no_config_write;    /* --no-config-write: refuse POST /v1/config/defaults */
    char  *chat_template_file; /* --chat-template-file <path>: replaces the
                                  model's tokenizer.chat_template (CLI and
                                  --server); NULL = use the GGUF template     */
    char  *stop[GEN_MAX_STOP]; /* --stop <text> (repeatable)                   */
    int    n_stop;
    int    max_tokens_limit;   /* --max-tokens-limit <N>: per-request cap in
                                  --server mode; 0 = model context length    */
    /* Phase 33: watermarking (off by default) */
    bool   watermark;          /* --watermark                                   */
    char  *watermark_key;      /* --watermark-key <passphrase>                  */
    float  watermark_gamma;    /* --watermark-gamma <g> (default 0.25)          */
    float  watermark_delta;    /* --watermark-delta <d> (default 2.0)           */
    char  *watermark_detect;   /* --watermark-detect <text>: score text, exit   */
    char  *ui_title;           /* --ui-title <name>: web UI brand name          */
    int    ui_logo;            /* --ui-logo ascii|text (0 = ascii, default)     */
    int    chat_format;        /* --chat-format: -1 auto, else ChatTemplateType */
    /* Phase 28: distributed layers */
    int    rpc_worker_port;    /* --rpc-worker <port>: serve --layers, exit      */
    char  *worker_layers;      /* --layers A-B: range a worker serves           */
    bool   rpc_bind_all;       /* --rpc-bind-all: worker listens on 0.0.0.0     */
    char  *remote_layers;      /* --remote-layers host:port@A-B[,...]           */
    char  *plan_layers;        /* --plan-layers GB,GB,...: print a RAM split    */
    int    queue_size;         /* --queue-size <N>: waiting chat requests (0)   */
    int    queue_timeout_s;    /* --queue-timeout <s>: max wait (120)           */
    int    best_of;            /* --best-of <N>: candidates, keep most likely (1) */
    int    best_of_max;        /* --best-of-max <N>: per-request cap (server, 8) */
    /* Phase 30: step-level MCTS (reasoning/mcts.h) and an optional PRM judge. */
    MctsOptions mcts;          /* --mcts N (simulations; 0 = off) --mcts-expand --mcts-depth
                                  --mcts-step-tokens --mcts-c --mcts-temperature --mcts-delim */
    int    mcts_max;           /* --mcts-max <N>: per-request simulation cap (server) */
    char  *prm_path;           /* --prm <gguf>: process reward model (judge) */
    char  *prm_format;         /* --prm-format <s>: judge input with {question} {steps} */
    char  *prm_good;           /* --prm-good <s> / --prm-bad <s>: judge answer strings */
    char  *prm_bad;
    char   mcts_delim_buf[16]; /* unescaped --mcts-delim */
    char  *attn_kernel;        /* --attn-kernel standard|tiled (standard)       */
    int    attn_tile;          /* --attn-tile <N>: tiled kernel tile (0 = L1d)  */
    int    attn_parallel_min_ctx; /* --attn-parallel-min-ctx <N> (256, 0=off)  */
    int    ctx_size;           /* --ctx-size <N>: context length (0 = auto)     */
    int    kv_type;            /* --kv-type f32|i8|i4 (KVCacheType, default f32) */
    char  *kv_offload_dir;     /* --kv-offload <dir>: disk-backed KV (Phase 31) */
    float  kv_ram_fraction;    /* --kv-ram-fraction <f>: KV RAM budget (0=0.60) */
    int    prompt_cache_mb;    /* --prompt-cache-mb <N>: Phase 26 store (256)   */
    int    prompt_cache_min_tokens; /* --prompt-cache-min-tokens <N> (32)       */
    bool   prompt_cache;       /* --prompt-cache on|off: server KV prefix reuse
                                  default (on)                                  */
} CliArgs;

// Parse CLI arguments into the CliArgs struct
TernaryError parse_args(CliArgs *args, int argc, char **argv);

// Builds the GenerationParams a CLI run (or the server defaults) starts from.
void args_fill_gen_params(const CliArgs *args, GenerationParams *gp);

/* Phase 38: update-check options from the --update-* / --no-update-check flags. */
void args_fill_update_options(const CliArgs *args, UpdateOptions *o);

// Print usage help
void print_usage(const char *prog_name);

#endif // PROJECT_ZERO_ARGS_H
