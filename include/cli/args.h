#ifndef PROJECT_ZERO_ARGS_H
#define PROJECT_ZERO_ARGS_H

#include <stdbool.h>
#include "core/platform.h"
#include "core/error.h"
#include "cli/color.h"
#include "api/server_config.h"
#include "transformer/gen_params.h"

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
    bool verbose;
    int seed;
    int classifier_override;  /* -1 = auto (BF16 default), 0 = BF16, 1 = INT8, 2 = INT4,
                                  3 = auto-fast (let engine pick fastest quantized cls) */
    char *simd_override;      /* NULL = auto, or "avx2", "avx512f", "vnni", "scalar" */
    bool calibrate;           /* --calibrate: force re-calibration of optimal settings */
    bool show_version;        /* --version/-v: print version + detected SIMD backend, exit 0 */
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
    int    ctx_size;           /* --ctx-size <N>: context length (0 = auto)     */
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

// Print usage help
void print_usage(const char *prog_name);

#endif // PROJECT_ZERO_ARGS_H
