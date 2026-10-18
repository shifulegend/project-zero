#include "cli/args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);

    printf("\nModel & Generation:\n");
    printf("  --model <path>      Path to the .bin/.gguf model file (required)\n");
    printf("  --tokenizer <path>  Path to the tokenizer.bin file (optional for GGUF)\n");
    printf("  --prompt <string>   Initial prompt string (optional, starts REPL if omitted)\n");
    printf("  --temperature <val> Temperature for sampling (default: 0.7)\n");
    printf("  --top-p <val>       Top-p nucleus sampling cutoff (default: 0.9)\n");
    printf("  --max-tokens <int>  Maximum tokens to generate (default: 512)\n");
    printf("  --seed <int>        RNG seed for reproducible sampling (default: random)\n");
    printf("  --top-k <int>       Keep only the k most likely tokens (default: 0 = off)\n");
    printf("  --min-p <val>       Drop tokens below min_p * p(max) (default: 0 = off)\n");
    printf("  --repeat-penalty <r> Repetition penalty (default: 1.0 = off)\n");
    printf("  --presence-penalty <v>  OpenAI presence penalty, -2..2 (default: 0)\n");
    printf("  --frequency-penalty <v> OpenAI frequency penalty, -2..2 (default: 0)\n");
    printf("  --repeat-last-n <N> Penalty window in generated tokens (default: 64,\n");
    printf("                      -1 = all generated tokens, 0 = penalties off)\n");
    printf("  --system-prompt <s> System prompt for chat-template models (default: none)\n");
    printf("  --stop <text>       Stop generation at this string; repeatable (max %d)\n", GEN_MAX_STOP);
    printf("  --reasoning         Enable hidden reasoning mode via <think> tags\n");
    printf("  --json              Constrain output to syntactically valid JSON\n");
    printf("                      (grammar-masked sampling; physically cannot emit\n");
    printf("                      invalid JSON syntax, not just prompted to try)\n");

    printf("  --ctx-size <int>    Context length to allocate (default: auto from free RAM;\n");
    printf("                      capped at the model's own context and the RAM budget)\n");
    printf("  --kv-ram-fraction <f> Share of free RAM the KV cache may use (default 0.60)\n");

    printf("\nWatermarking (off by default):\n");
    printf("  --watermark         Bias sampling toward a keyed green list so output can\n");
    printf("                      later be identified (statistical, not visible).\n");
    printf("  --watermark-key <s> Secret passphrase for the green list.\n");
    printf("  --watermark-gamma <g> Green-list fraction (default: 0.25).\n");
    printf("  --watermark-delta <d> Logit bias for green tokens (default: 2.0).\n");
    printf("  --watermark-detect <text> Score <text> for the watermark (needs --model for\n");
    printf("                      the tokenizer and the same key/gamma), then exit.\n");

    printf("\nHardware & Performance:\n");
    printf("  --threads <int>     Number of threads (default: auto)\n");
    printf("  --classifier <fmt>  Classifier quantization: auto, bf16, int8, int4, auto-fast\n");
    printf("                      Default: auto (BF16, full intelligence, zero loss).\n");
    printf("                      int8/int4: faster but quantizes classifier weights.\n");
    printf("                      auto-fast: let engine pick fastest quantized mode.\n");
    printf("  --simd <backend>    Force SIMD backend: auto, avx2, avx512f, vnni, scalar\n");
    printf("                      Default: auto (highest tier). Use avx2 on Tiger Lake\n");
    printf("                      to avoid AVX-512 frequency throttle.\n");
    printf("  --calibrate         Run micro-benchmarks to find optimal settings for\n");
    printf("                      this hardware. Results cached in ~/.project-zero/\n");
    printf("                      Runs automatically on first use or hardware change.\n");

    printf("\nServer (OpenAI-compatible HTTP API):\n");
    printf("  --server            Run as an HTTP API server (--prompt is ignored).\n");
    printf("  --port <int>        Listen port when --server is used (default: 8080).\n");
    printf("  --cors              Enable CORS response headers (off by default).\n");
    printf("  --cors-origin <o>   Allowed origin for CORS; repeatable, or \"*\" for any.\n");
    printf("                      Requires --cors.\n");
    printf("  --api-key <key>     Require 'Authorization: Bearer <key>' on API requests\n");
    printf("                      (off by default — loopback experimentation is open).\n");
    printf("  --metrics           Enable GET /metrics (Prometheus text exposition).\n");
    printf("  --web-ui <mode>     Web chat UI: auto (default, on with --server), on, off.\n");
    printf("  --max-tokens-limit <N> Per-request max_tokens cap (default: 0 = model\n");
    printf("                      context length). Generation flags above become the\n");
    printf("                      server defaults; requests/the web UI may override them.\n");
    printf("  --ui-title <name>   Brand name shown in the web UI (default: project-zero)\n");
    printf("  --prompt-cache <m>  on (default) | off: reuse the KV cache for the prompt\n");
    printf("                      prefix shared with the previous request (multi-turn\n");
    printf("                      chat skips re-reading the history). Per request:\n");
    printf("                      \"cache_prompt\": false.\n");
    printf("  --prompt-cache-mb <N> RAM for snapshots of several conversations' prompt\n");
    printf("                      KV so switching between them skips re-reading\n");
    printf("                      (default: 256, 0 = only the live conversation).\n");
    printf("  --prompt-cache-min-tokens <N> Shortest sequence worth storing (default 32)\n");
    printf("  --static-dir <path> Serve the web UI from disk instead of the embedded\n");
    printf("                      bundle (dev mode, for iterating on webui/dist/).\n");

    printf("\nMultimodal (vision):\n");
    printf("  --image <path>      Path to image file for multimodal queries (optional)\n");
    printf("  --vision <path>     Path to vision.bin encoder weights\n");
    printf("  --proj <path>       Path to projector.bin weights\n");
    printf("                      Both --vision and --proj required to use --image.\n");
    printf("                      Extract with: python tools/extract_multimodal.py\n");

    printf("\nMemory & RAG:\n");
    printf("  --memory-db <path>  Path to vector DB file for RAG memory.\n");
    printf("                      File is created if it does not exist.\n");

    printf("\nSpeculative decoding:\n");
    printf("  --draft-model <path> Path to a small GGUF draft model. Disabled by default --\n");
    printf("                      this is the ONLY way to enable speculative decoding;\n");
    printf("                      there is no default/embedded draft model. The draft\n");
    printf("                      model must share the verifier's tokenizer (same\n");
    printf("                      vocab_size) and be GGUF (native .bin models cannot be\n");
    printf("                      used as a draft). Not supported for linear-attention\n");
    printf("                      models (Qwen3.5/3.6 hybrid) -- refused at startup.\n");
    printf("  --spec-length <int> Tokens drafted per round when --draft-model is set\n");
    printf("                      (default: 5). Ignored otherwise.\n");

    printf("\nLoRA adapters:\n");
    printf("  --lora <path>       Path to a .lora.bin adapter. Disabled by default --\n");
    printf("                      this is the ONLY way to enable a LoRA adapter; there\n");
    printf("                      is no default/embedded adapter. The adapter must have\n");
    printf("                      been built for this exact base model (n_layers, dim,\n");
    printf("                      hidden_dim all checked at load time). v1 scope: the\n");
    printf("                      generic dense/GQA attention/FFN path only -- not MLA,\n");
    printf("                      Qwen3-MoE, Qwen3.5/3.6 hybrid, or MoE-FFN layers.\n");

    printf("\nOutput:\n");
    printf("  --color <mode>      Color output: auto (default), always, never.\n");
    printf("                      auto respects the NO_COLOR env var and disables\n");
    printf("                      color when stdout isn't a terminal.\n");
    printf("  --verbose           Enable verbose logging\n");
    printf("  --dump-tensors <f>  Write intermediate tensor stats to a CSV file\n");
    printf("  --version, -v       Print version and the SIMD backend for this CPU, then exit.\n");
    printf("  --help, -h          Print this help and exit.\n");

    printf("\nExamples:\n");
    printf("  %s --model models/smollm2.gguf --prompt \"What is the capital of France?\"\n", prog_name);
    printf("  %s --model models/smollm2.gguf --threads 4          # interactive REPL\n", prog_name);
    printf("  %s --model models/smollm2.gguf --server --port 8080 --cors --cors-origin \"*\"\n", prog_name);
}

TernaryError parse_args(CliArgs *args, int argc, char **argv) {
    if (!args) return TN_ERR_INVALID_CONFIG;

    // Set defaults
    args->model_path = NULL;
    args->tokenizer_path = NULL;
    args->prompt = NULL;
    args->image_path = NULL;
    args->vision_path = NULL;
    args->proj_path = NULL;
    args->memory_db_path = NULL;
    args->temperature = 0.7f;
    args->top_p = 0.9f;
    args->max_tokens = 512;
    args->num_threads = 0; // auto
    args->enable_reasoning = false;
    args->json_mode = false;
    args->verbose = false;
    args->seed = -1;
    args->classifier_override = -1; // auto (BF16)
    args->simd_override = NULL;     // auto
    args->calibrate = false;
    args->show_version = false;
    args->dump_tensors_path = NULL;
    args->server_mode = 0;
    args->server_port = 8080;
    args->cors_enabled = false;
    args->num_cors_origins = 0;
    args->api_key = NULL;
    args->metrics_enabled = false;
    args->static_dir = NULL;
    args->web_ui_mode = WEBUI_MODE_AUTO;
    args->color_mode = TN_COLOR_AUTO;
    args->draft_model_path = NULL; /* disabled by default -- the ONLY way to enable
                                       speculative decoding is --draft-model */
    args->spec_length = 5;
    args->lora_path = NULL; /* disabled by default -- the ONLY way to enable a
                                LoRA adapter is --lora */
    {
        SamplerSettings ss;
        sampler_settings_default(&ss);
        args->top_k = ss.top_k;
        args->min_p = ss.min_p;
        args->repeat_penalty = ss.repetition_penalty;
        args->presence_penalty = ss.presence_penalty;
        args->frequency_penalty = ss.frequency_penalty;
        args->repeat_last_n = ss.penalty_last_n;
    }
    args->system_prompt = NULL;
    args->n_stop = 0;
    args->max_tokens_limit = 0;
    args->watermark = false;
    args->watermark_key = NULL;
    args->watermark_gamma = WATERMARK_DEFAULT_GAMMA;
    args->watermark_delta = WATERMARK_DEFAULT_DELTA;
    args->watermark_detect = NULL;
    args->ui_title = NULL;
    args->prompt_cache = true;
    args->prompt_cache_mb = 256;
    args->prompt_cache_min_tokens = 32;
    args->ctx_size = 0;
    args->kv_ram_fraction = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args->model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            args->tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args->prompt = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            args->image_path = argv[++i];
        } else if (strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
            args->vision_path = argv[++i];
        } else if (strcmp(argv[i], "--proj") == 0 && i + 1 < argc) {
            args->proj_path = argv[++i];
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            args->temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args->top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args->num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args->seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args->top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            args->min_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args->repeat_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
            args->presence_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
            args->frequency_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
            args->repeat_last_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--system-prompt") == 0 && i + 1 < argc) {
            args->system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
            if (args->n_stop >= GEN_MAX_STOP) {
                fprintf(stderr, "Error: too many --stop values (max %d)\n", GEN_MAX_STOP);
                return TN_ERR_INVALID_CONFIG;
            }
            args->stop[args->n_stop++] = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens-limit") == 0 && i + 1 < argc) {
            args->max_tokens_limit = atoi(argv[++i]);
            if (args->max_tokens_limit < 0) {
                fprintf(stderr, "Error: --max-tokens-limit must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--watermark") == 0) {
            args->watermark = true;
        } else if (strcmp(argv[i], "--watermark-key") == 0 && i + 1 < argc) {
            args->watermark_key = argv[++i];
        } else if (strcmp(argv[i], "--watermark-gamma") == 0 && i + 1 < argc) {
            args->watermark_gamma = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--watermark-delta") == 0 && i + 1 < argc) {
            args->watermark_delta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            args->ctx_size = atoi(argv[++i]);
            if (args->ctx_size < 0) {
                fprintf(stderr, "Error: --ctx-size must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--kv-ram-fraction") == 0 && i + 1 < argc) {
            args->kv_ram_fraction = (float)atof(argv[++i]);
            if (!(args->kv_ram_fraction > 0.0f && args->kv_ram_fraction <= 0.95f)) {
                fprintf(stderr, "Error: --kv-ram-fraction must be in (0, 0.95]\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--prompt-cache-mb") == 0 && i + 1 < argc) {
            args->prompt_cache_mb = atoi(argv[++i]);
            if (args->prompt_cache_mb < 0) {
                fprintf(stderr, "Error: --prompt-cache-mb must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--prompt-cache-min-tokens") == 0 && i + 1 < argc) {
            args->prompt_cache_min_tokens = atoi(argv[++i]);
            if (args->prompt_cache_min_tokens < 1) {
                fprintf(stderr, "Error: --prompt-cache-min-tokens must be >= 1\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--prompt-cache") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "on") == 0) args->prompt_cache = true;
            else if (strcmp(mode, "off") == 0) args->prompt_cache = false;
            else {
                fprintf(stderr, "Error: --prompt-cache must be on or off\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--ui-title") == 0 && i + 1 < argc) {
            args->ui_title = argv[++i];
        } else if (strcmp(argv[i], "--watermark-detect") == 0 && i + 1 < argc) {
            args->watermark_detect = argv[++i];
        } else if (strcmp(argv[i], "--memory-db") == 0 && i + 1 < argc) {
            args->memory_db_path = argv[++i];
        } else if (strcmp(argv[i], "--classifier") == 0 && i + 1 < argc) {
            const char *fmt = argv[++i];
            if (strcmp(fmt, "auto") == 0)           args->classifier_override = -1;
            else if (strcmp(fmt, "bf16") == 0)       args->classifier_override = 0;
            else if (strcmp(fmt, "int8") == 0)       args->classifier_override = 1;
            else if (strcmp(fmt, "int4") == 0)       args->classifier_override = 2;
            else if (strcmp(fmt, "auto-fast") == 0)  args->classifier_override = 3;
            else {
                fprintf(stderr, "Error: --classifier must be one of: auto, bf16, int8, int4, auto-fast\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--simd") == 0 && i + 1 < argc) {
            const char *backend = argv[++i];
            if (strcmp(backend, "auto") == 0) {
                args->simd_override = NULL;
            } else if (strcmp(backend, "avx2") == 0 || strcmp(backend, "avx512f") == 0 ||
                       strcmp(backend, "vnni") == 0 || strcmp(backend, "scalar") == 0) {
                args->simd_override = (char *)backend;
            } else {
                fprintf(stderr, "Error: --simd must be one of: auto, avx2, avx512f, vnni, scalar\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--calibrate") == 0) {
            args->calibrate = true;
        } else if (strcmp(argv[i], "--reasoning") == 0) {
            args->enable_reasoning = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            args->json_mode = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            args->verbose = true;
        } else if (strcmp(argv[i], "--dump-tensors") == 0 && i + 1 < argc) {
            args->dump_tensors_path = argv[++i];
        } else if (strcmp(argv[i], "--server") == 0) {
            args->server_mode = 1;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args->server_port = atoi(argv[++i]);
            if (args->server_port <= 0 || args->server_port > 65535) {
                fprintf(stderr, "Error: --port must be 1–65535\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--cors") == 0) {
            args->cors_enabled = true;
        } else if (strcmp(argv[i], "--cors-origin") == 0 && i + 1 < argc) {
            if (args->num_cors_origins >= CLI_MAX_CORS_ORIGINS) {
                fprintf(stderr, "Error: too many --cors-origin values (max %d)\n", CLI_MAX_CORS_ORIGINS);
                return TN_ERR_INVALID_CONFIG;
            }
            args->cors_origins[args->num_cors_origins++] = argv[++i];
        } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            args->api_key = argv[++i];
        } else if (strcmp(argv[i], "--metrics") == 0) {
            args->metrics_enabled = true;
        } else if (strcmp(argv[i], "--static-dir") == 0 && i + 1 < argc) {
            args->static_dir = argv[++i];
        } else if (strcmp(argv[i], "--web-ui") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "auto") == 0)      args->web_ui_mode = WEBUI_MODE_AUTO;
            else if (strcmp(mode, "on") == 0)   args->web_ui_mode = WEBUI_MODE_ON;
            else if (strcmp(mode, "off") == 0)  args->web_ui_mode = WEBUI_MODE_OFF;
            else {
                fprintf(stderr, "Error: --web-ui must be one of: auto, on, off\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            args->draft_model_path = argv[++i];
        } else if (strcmp(argv[i], "--spec-length") == 0 && i + 1 < argc) {
            args->spec_length = atoi(argv[++i]);
            if (args->spec_length <= 0) {
                fprintf(stderr, "Error: --spec-length must be a positive integer\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            args->lora_path = argv[++i];
        } else if (strcmp(argv[i], "--color") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "auto") != 0 && strcmp(mode, "always") != 0 && strcmp(mode, "never") != 0) {
                fprintf(stderr, "Error: --color must be one of: auto, always, never\n");
                return TN_ERR_INVALID_CONFIG;
            }
            args->color_mode = tn_color_mode_parse(mode);
        } else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            /* Short-circuit: --version must work without --model so the release
             * binary is smoke-testable on a clean machine with no model file.
             * main() prints the version and exits 0 before any model load. */
            args->show_version = true;
            return TN_OK;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        }
    }

    if (!args->model_path) {
        fprintf(stderr, "Error: --model <path> is required.\n");
        print_usage(argv[0]);
        return TN_ERR_INVALID_CONFIG;
    }

    /* Only warn about missing tokenizer for non-GGUF models.
     * GGUF files embed tokenizer metadata (vocab, BPE merges, chat template,
     * BOS/EOS token IDs) and are loaded automatically via tokenizer_load_from_gguf(). */
    if (!args->tokenizer_path) {
        const char *mp = args->model_path ? args->model_path : "";
        int is_gguf = (strstr(mp, ".gguf") != NULL);
        if (!is_gguf) {
            fprintf(stderr, "Warning: --tokenizer <path> not provided. "
                    "Text generation will fail without a vocabulary file.\n");
        }
    }

    return TN_OK;
}

void args_fill_gen_params(const CliArgs *args, GenerationParams *gp) {
    gen_params_default(gp);
    if (!args) return;
    gp->max_tokens                 = args->max_tokens;
    gp->sampler.temperature        = args->temperature;
    gp->sampler.top_p              = args->top_p;
    gp->sampler.top_k              = args->top_k;
    gp->sampler.min_p              = args->min_p;
    gp->sampler.repetition_penalty = args->repeat_penalty;
    gp->sampler.presence_penalty   = args->presence_penalty;
    gp->sampler.frequency_penalty  = args->frequency_penalty;
    gp->sampler.penalty_last_n     = args->repeat_last_n;
    sampler_settings_sanitize(&gp->sampler);
    gp->seed      = args->seed;
    gp->json_mode = args->json_mode;
    gp->system_prompt = args->system_prompt;
    for (int i = 0; i < args->n_stop && i < GEN_MAX_STOP; i++) gp->stop[i] = args->stop[i];
    gp->n_stop = args->n_stop < GEN_MAX_STOP ? args->n_stop : GEN_MAX_STOP;
    gp->watermark.enabled = args->watermark;
    gp->watermark.key     = watermark_key_from_string(args->watermark_key);
    gp->watermark.gamma   = args->watermark_gamma;
    gp->watermark.delta   = args->watermark_delta;
    watermark_config_sanitize(&gp->watermark);
}
