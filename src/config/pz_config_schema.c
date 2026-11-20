/*
 * config/pz_config_schema — the table of persistable settings behind `pz config`, PZ_<KEY>
 * environment variables and the config file (split out of pz_config.c, which reads and writes
 * the file).
 */
#include "config/pz_config_schema.h"

#include <stddef.h>

#include "config/pz_config.h"
#include "core/limits.h" /* PZ_PARALLEL_*, BEST_OF_N_MAX: bounds shared with parse_args */

/* ── schema ───────────────────────────────────────────────────────────────
 * Settings worth persisting; each key is the CLI flag without "--", so the
 * value is validated again by parse_args at startup. Per-run inputs
 * (--prompt, --image, --perplexity ...) are deliberately absent. Keep this
 * table in sync with args.c (engineering-rules DoD #6). */
const PzCfgKey g_pz_cfg_schema[] = {
    /* model */
    {"model", PZ_CFG_STRING, "model", "Default model file (.gguf)", NULL, 0, 0, 0},
    {"tokenizer", PZ_CFG_STRING, "model", "Tokenizer file for non-GGUF models", NULL, 0, 0, 0},
    {"chat-template-file", PZ_CFG_STRING, "model",
     "Jinja chat template that replaces the model's own", NULL, 0, 0, 0},
    {"chat-format", PZ_CFG_STRING, "model", "Prompt format for models without a chat template",
     NULL, 0, 0, 0},
    {"lora", PZ_CFG_STRING, "model", "LoRA adapter to apply", NULL, 0, 0, 0},
    {"draft-model", PZ_CFG_STRING, "model", "Draft model for speculative decoding", NULL, 0, 0, 0},
    {"spec-length", PZ_CFG_INT, "model", "Draft tokens per speculative step", NULL, 1, 64, 0},
    /* hardware */
    {"threads", PZ_CFG_INT, "hardware", "Worker threads (0 = auto)", NULL, 0, 1024, 0},
    {"simd", PZ_CFG_STRING, "hardware", "SIMD backend", "auto|avx2|avx512f|vnni|scalar", 0, 0, 0},
    {"classifier", PZ_CFG_STRING, "hardware", "LM-head precision", "auto|bf16|int8|int4|auto-fast",
     0, 0, 0},
    {"no-calibrate", PZ_CFG_BOOL, "hardware",
     "Never run or use the hardware calibration (reproducible benchmarks)", NULL, 0, 0, 0},
    {"act-quant", PZ_CFG_STRING, "hardware", "Activation numerics of ternary matmuls",
     "auto|int8|float", 0, 0, 0},
    {"quality-tolerance", PZ_CFG_FLOAT, "hardware",
     "Largest perplexity increase (%) a fast setting may cost; 0 = no bar", NULL, 0, 100, 0},
    {"self-check-tokens", PZ_CFG_INT, "hardware", "Length of the accuracy probe in tokens", NULL,
     16, 4096, 0},
    /* memory / context */
    {"ctx-size", PZ_CFG_INT, "memory", "Context length in tokens (0 = sized from RAM)", NULL, 0,
     16777216, 0},
    {"kv-type", PZ_CFG_STRING, "memory", "KV cache storage", "f32|i8|i4", 0, 0, 0},
    {"kv-ram-fraction", PZ_CFG_FLOAT, "memory",
     "Share of free RAM the KV cache may use (0 = default)", NULL, 0, 1, 0},
    {"kv-offload", PZ_CFG_STRING, "memory", "Directory for a disk-backed KV cache", NULL, 0, 0, 0},
    {"prompt-cache-mb", PZ_CFG_INT, "memory", "Prompt snapshot cache budget in MiB", NULL, 0,
     1048576, 0},
    /* generation defaults */
    {"max-tokens", PZ_CFG_INT, "generation", "Default tokens to generate", NULL, 1, 16777216, 0},
    {"temperature", PZ_CFG_FLOAT, "generation", "Sampling temperature", NULL, 0, 10, 0},
    {"top-p", PZ_CFG_FLOAT, "generation", "Nucleus sampling threshold", NULL, 0, 1, 0},
    {"top-k", PZ_CFG_INT, "generation", "Top-k sampling (0 = off)", NULL, 0, 1000000, 0},
    {"min-p", PZ_CFG_FLOAT, "generation", "Min-p sampling threshold", NULL, 0, 1, 0},
    {"repeat-penalty", PZ_CFG_FLOAT, "generation", "Repetition penalty (1 = off)", NULL, 0, 10, 0},
    {"presence-penalty", PZ_CFG_FLOAT, "generation", "Presence penalty", NULL, -10, 10, 0},
    {"frequency-penalty", PZ_CFG_FLOAT, "generation", "Frequency penalty", NULL, -10, 10, 0},
    {"repeat-last-n", PZ_CFG_INT, "generation", "Tokens considered by the penalties (-1 = all)",
     NULL, -1, 1048576, 0},
    {"seed", PZ_CFG_INT, "generation", "Sampling seed (-1 = random)", NULL, -1, 9.2e18, 0},
    {"system-prompt", PZ_CFG_STRING, "generation", "Default system prompt", NULL, 0, 0, 0},
    {"stop", PZ_CFG_LIST, "generation", "Stop sequences", NULL, 0, 0, 0},
    {"color", PZ_CFG_STRING, "generation", "Terminal colour", "auto|always|never", 0, 0, 0},
    /* logging */
    {"log-format", PZ_CFG_STRING, "logging", "Diagnostics on stderr: text or JSON lines",
     "text|json", 0, 0, 0},
    {"log-level", PZ_CFG_STRING, "logging", "Lowest level written", "debug|info|warn|error", 0, 0,
     0},
    /* server */
    {"port", PZ_CFG_INT, "server", "HTTP port for --server", NULL, 1, 65535, 0},
    {"web-ui", PZ_CFG_STRING, "server", "Web chat UI", "auto|on|off", 0, 0, 0},
    {"ui-title", PZ_CFG_STRING, "server", "Web UI title (also renders the wordmark)", NULL, 0, 0,
     0},
    {"cors", PZ_CFG_BOOL, "server", "Allow cross-origin requests", NULL, 0, 0, 0},
    {"cors-origin", PZ_CFG_LIST, "server", "Allowed CORS origins", NULL, 0, 0, 0},
    {"api-key", PZ_CFG_STRING, "server", "Require this bearer token", NULL, 0, 0, 1},
    {"metrics", PZ_CFG_BOOL, "server", "Expose /metrics", NULL, 0, 0, 0},
    {"queue-size", PZ_CFG_INT, "server", "Requests that may wait for the generation slot", NULL, 0,
     1048576, 0},
    {"queue-timeout", PZ_CFG_INT, "server", "Seconds a queued request waits", NULL, 0, 86400, 0},
    {"rate-limit", PZ_CFG_INT, "server", "Requests/second on the non-generation routes (0 = off)",
     NULL, 0, 100000, 0},
    {"client-timeout", PZ_CFG_INT, "server",
     "Seconds a silent client may hold a connection (0 = never)", NULL, 0, 86400, 0},
    {"max-prompt-bytes", PZ_CFG_INT, "server",
     "Message text one chat request may carry; more gets 400 (0 = no cap)", NULL, 0, 1073741824, 0},
    {"max-connections", PZ_CFG_INT, "server",
     "Open client connections at once; more get 503 (0 = unlimited)", NULL, 0, 1048576, 0},
    {"parallel", PZ_CFG_INT, "server", "Chat requests served at once (1 = one at a time)", NULL, 1,
     PZ_PARALLEL_MAX, 0},
    {"parallel-batch", PZ_CFG_INT, "server", "Tokens per batched forward pass with --parallel",
     NULL, 1, PZ_PARALLEL_BATCH_MAX, 0},
    {"max-tokens-limit", PZ_CFG_INT, "server", "Per-request max_tokens cap (0 = context)", NULL, 0,
     16777216, 0},
    {"best-of-max", PZ_CFG_INT, "server", "Largest best_of a request may ask for", NULL, 1,
     BEST_OF_N_MAX, 0},
    {"no-config-write", PZ_CFG_BOOL, "server", "Web UI may not save server defaults to this file",
     NULL, 0, 0, 0},
    {"mcts-max", PZ_CFG_INT, "server", "Largest tree-search budget a request may ask for", NULL, 0,
     1048576, 0},
    /* updates */
    {"no-update-check", PZ_CFG_BOOL, "updates", "Never check for new releases", NULL, 0, 0, 0},
    {"update-channel", PZ_CFG_STRING, "updates", "Release channel", "auto|stable|prerelease", 0, 0,
     0},
    {"update-interval", PZ_CFG_INT, "updates", "Hours between update checks (0 = off)", NULL, 0,
     8760, 0},
};

/* Compile-time check that the table fits the arrays sized by PZ_CFG_SCHEMA_MAX. */
typedef char pz_cfg_schema_fits_the_bound
    [(sizeof(g_pz_cfg_schema) / sizeof(g_pz_cfg_schema[0]) <= PZ_CFG_SCHEMA_MAX) ? 1 : -1];

const int g_pz_cfg_schema_len = (int)(sizeof(g_pz_cfg_schema) / sizeof(g_pz_cfg_schema[0]));
