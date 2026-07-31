#ifndef PROJECT_ZERO_ARGS_H
#define PROJECT_ZERO_ARGS_H

#include <stdbool.h>
#include "core/platform.h"
#include "core/error.h"
#include "cli/color.h"
#include "api/server_config.h"

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
} CliArgs;

// Parse CLI arguments into the CliArgs struct
TernaryError parse_args(CliArgs *args, int argc, char **argv);

// Print usage help
void print_usage(const char *prog_name);

#endif // PROJECT_ZERO_ARGS_H
