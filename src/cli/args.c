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
    printf("  --seed <int>        RNG seed (default: random)\n");
    printf("  --reasoning         Enable hidden reasoning mode via <think> tags\n");

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
    args->moe_threading_override = NULL;

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
        } else if (strcmp(argv[i], "--moe-threading") == 0 && i + 1 < argc) {
            args->moe_threading_override = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args->seed = atoi(argv[++i]);
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
