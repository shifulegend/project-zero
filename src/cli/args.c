#include "cli/args.h"
#include "transformer/attn_tiled.h"
#include "kv_cache/kv_quant_ops.h"
#include "api/chat_compile.h"
#include "reasoning/best_of.h"
#include "update/update_check.h"
#include "math/rope.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("       %s config        set up defaults interactively (pz config --help)\n", prog_name);
    printf("Saved settings (pz config) and PZ_<KEY> environment variables act as defaults;\n"
           "flags given here override them. --no-config ignores both.\n");

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
    printf("  --chat-template-file <f>  Jinja chat template replacing the model's own\n");
    printf("                      (e.g. docs/templates/bitnet-b1.58.jinja); default: GGUF\n");
    printf("  --no-request-template  Server: ignore a request's \"chat_template\" field\n");
    printf("  --no-config-write   Server: the web UI may not save defaults to the config file\n");
    printf("  --stop <text>       Stop generation at this string; repeatable (max %d)\n", GEN_MAX_STOP);
    printf("  --best-of <N>       Generate N candidates, keep the one the model finds\n");
    printf("                      most likely (mean log-prob). Default 1 (off).\n");
    printf("  --best-of-max <N>   Server: largest best_of a request may ask for (default %d)\n", BEST_OF_MAX_DEFAULT);
    printf("  --mcts <N>          Step-level tree search with N simulations (0 = off): sample\n");
    printf("                      --mcts-expand children per step, score each step, answer\n");
    printf("                      along the most visited path\n");
    printf("  --mcts-expand <k> --mcts-depth <d> --mcts-step-tokens <n> --mcts-c <c>\n");
    printf("  --mcts-temperature <t> --mcts-delim <s>   search shape (defaults %d %d %d %.1f %.1f \"\\n\")\n",
           MCTS_DEFAULT_EXPAND, MCTS_DEFAULT_DEPTH, MCTS_DEFAULT_STEP_TOKENS,
           (double)MCTS_DEFAULT_C, (double)MCTS_DEFAULT_TEMPERATURE);
    printf("  --mcts-max <N>      Server: largest \"mcts\" simulations a request may ask for\n");
    printf("  --prm <gguf>        Process reward model scoring each step (default: the policy's\n");
    printf("                      own token probability). --prm-format <s> with {question}\n");
    printf("                      {steps}; --prm-good <s> / --prm-bad <s> answer strings\n");
    printf("  --reasoning         Enable hidden reasoning mode via <think> tags\n");
    printf("  --json              Constrain output to syntactically valid JSON\n");
    printf("  --perplexity <file> Report perplexity on a text file, then exit (llama.cpp's\n");
    printf("                      llama-perplexity method: chunks of --ppl-ctx tokens\n");
    printf("                      (default 512), second half of each chunk scored;\n");
    printf("                      --ppl-chunks N limits the chunk count).\n");
    printf("  --ignore-eos        Never sample end-of-sequence: generate exactly --max-tokens\n");
    printf("                      (fixed-length benchmarks; server default, per-request too).\n");
    printf("  --raw-prompt        Feed --prompt verbatim (BOS + text) instead of wrapping it in\n");
    printf("                      the model's chat template -- plain completion, as\n");
    printf("                      llama-completion does (benchmarks, base models).\n");
    printf("                      (grammar-masked sampling; physically cannot emit\n");
    printf("                      invalid JSON syntax, not just prompted to try)\n");

    printf("  --ctx-size <int>    Context length to allocate (default: auto from free RAM;\n");
    printf("                      capped at the model's own context and the RAM budget)\n");
    printf("  --kv-ram-fraction <f> Share of free RAM the KV cache may use (default 0.60)\n");
    printf("  --kv-type <t>       KV cache storage: f32 (default) | i8 (4x smaller, per-row\n");
    printf("                      absmax scale) | i4 (~7x smaller, 32-value groups; needs\n");
    printf("                      head_dim %% 32 == 0, else i8). Dense/GQA attention path\n");
    printf("  --kv-offload <dir>  Keep the KV cache in a disk-backed file in <dir> (NVMe/SSD):\n");
    printf("                      the OS keeps hot pages in RAM, so context is bounded by\n");
    printf("                      free disk (x --kv-ram-fraction), not RAM. Dense/GQA path.\n");
    printf("  --rope-layout <l>   auto (default: from the GGUF architecture, as llama.cpp) |\n");
    printf("                      norm (adjacent pairs) | neox (rotate-half pairs)\n");
    printf("  --attn-kernel <k>   standard (default) | tiled (single-pass online softmax)\n");
    printf("  --attn-tile <int>   Positions per tile for --attn-kernel tiled (0 = from L1d)\n");
    printf("  --attn-parallel-min-ctx <N> Spread attention heads over threads once the\n");
    printf("                      context reaches N tokens (default %d, 0 = never)\n", ATTN_PARALLEL_MIN_CTX_DEFAULT);

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
    printf("  --queue-size <N>    Let up to N chat requests wait (in order) while one is\n");
    printf("                      generating, instead of answering 429 (default: 0)\n");
    printf("  --queue-timeout <s> Seconds a queued request waits before giving up (120)\n");
    printf("  --ui-title <name>   Brand name shown in the web UI (default: project-zero)\n");
    printf("  --ui-logo <m>       ascii (default: the block wordmark of --ui-title, as in the\n");
    printf("                      terminal banner; falls back to text for other letters) | text\n");
    printf("  --chat-format <f>   Prompt format for models without an embedded chat\n");
    printf("                      template: auto (default: from the vocabulary), chatml,\n");
    printf("                      llama3, deepseek, mistral, plain\n");
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

    printf("\nDistributed layers (pipeline across processes/machines):\n");
    printf("  --rpc-worker <port> Serve a layer range for a coordinator, then exit on error.\n");
    printf("  --layers <A-B>      Layers [A, B) this worker runs (with --rpc-worker).\n");
    printf("  --rpc-bind-all      Worker listens on all interfaces (default: 127.0.0.1 only;\n");
    printf("                      the protocol has no authentication -- trusted LANs only).\n");
    printf("  --remote-layers <s> Coordinator: host:port@A-B[,host:port@C-D...] — run those\n");
    printf("                      layers on workers started with the same model file.\n");
    printf("  --plan-layers <s>   Print a layer split proportional to each node's RAM in GB\n");
    printf("                      (e.g. 16,8,8 — first entry is this coordinator), then exit.\n");

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

    printf("\nUpdates (notify only -- nothing is downloaded or installed):\n");
    printf("  --check-update      Check GitHub Releases now, print the result, exit\n");
    printf("                      (0 up to date/unknown, 3 update available, 1 error).\n");
    printf("  --no-update-check   Never check. Also: PZ_NO_UPDATE_CHECK=1, DO_NOT_TRACK=1;\n");
    printf("                      always off when CI is set. Otherwise interactive runs and\n");
    printf("                      --server check in the background at most once per\n");
    printf("                      --update-interval (sends only a User-Agent with the version).\n");
    printf("  --update-interval <h> Hours between checks (default %d, 0 = off).\n", UPDATE_DEFAULT_INTERVAL_H);
    printf("  --update-channel <c> auto (default: pre-releases while on 0.x) | stable | prerelease\n");
    printf("  --update-repo <o/n> Releases source (default: upstream; forks set their own).\n");
    printf("  --update-api <url>  API base, e.g. GitHub Enterprise (default https://api.github.com).\n");

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
    args->raw_prompt = false;
    args->ignore_eos = false;
    args->rope_layout = -1;
    args->perplexity_file = NULL;
    args->ppl_ctx = 512;
    args->ppl_chunks = 0;
    args->verbose = false;
    args->seed = -1;
    args->classifier_override = -1; // auto (BF16)
    args->simd_override = NULL;     // auto
    args->calibrate = false;
    args->show_version = false;
    args->check_update = false;
    args->no_update_check = false;
    args->update_repo = NULL;
    args->update_api = NULL;
    args->update_interval_h = -1;
    args->update_channel = UPDATE_CHANNEL_AUTO;
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
    args->chat_template_file = NULL;
    args->no_request_template = false;
    args->n_stop = 0;
    args->max_tokens_limit = 0;
    args->watermark = false;
    args->watermark_key = NULL;
    args->watermark_gamma = WATERMARK_DEFAULT_GAMMA;
    args->watermark_delta = WATERMARK_DEFAULT_DELTA;
    args->watermark_detect = NULL;
    args->ui_title = NULL;
    args->ui_logo = 0;
    args->chat_format = -1;
    args->best_of = 1;
    args->queue_size = 0;
    args->rpc_worker_port = 0;
    args->worker_layers = NULL;
    args->rpc_bind_all = false;
    args->remote_layers = NULL;
    args->plan_layers = NULL;
    args->queue_timeout_s = 120;
    args->best_of_max = BEST_OF_MAX_DEFAULT;
    memset(&args->mcts, 0, sizeof(args->mcts));
    args->mcts_max = 64;
    args->prm_path = args->prm_format = args->prm_good = args->prm_bad = NULL;
    args->prompt_cache = true;
    args->prompt_cache_mb = 256;
    args->prompt_cache_min_tokens = 32;
    args->ctx_size = 0;
    args->kv_type = 0;
    args->kv_offload_dir = NULL;
    args->attn_kernel = NULL;
    args->attn_tile = 0;
    args->attn_parallel_min_ctx = ATTN_PARALLEL_MIN_CTX_DEFAULT;
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
        } else if (strcmp(argv[i], "--no-request-template") == 0) {
            args->no_request_template = true;
        } else if (strcmp(argv[i], "--no-config-write") == 0) {
            args->no_config_write = true;
        } else if (strcmp(argv[i], "--chat-template-file") == 0 && i + 1 < argc) {
            args->chat_template_file = argv[++i];
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
        } else if (strcmp(argv[i], "--perplexity") == 0 && i + 1 < argc) {
            args->perplexity_file = argv[++i];
        } else if (strcmp(argv[i], "--ppl-ctx") == 0 && i + 1 < argc) {
            args->ppl_ctx = atoi(argv[++i]);
            if (args->ppl_ctx < 4) {
                fprintf(stderr, "Error: --ppl-ctx must be >= 4\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--ppl-chunks") == 0 && i + 1 < argc) {
            args->ppl_chunks = atoi(argv[++i]);
            if (args->ppl_chunks < 0) {
                fprintf(stderr, "Error: --ppl-chunks must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--rope-layout") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            args->rope_layout = strcmp(m, "auto") == 0 ? -1 : rope_layout_parse(m);
            if (strcmp(m, "auto") != 0 && args->rope_layout < 0) {
                fprintf(stderr, "Error: --rope-layout must be auto, norm or neox\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--attn-kernel") == 0 && i + 1 < argc) {
            args->attn_kernel = argv[++i];
            if (attn_kernel_parse(args->attn_kernel) < 0) {
                fprintf(stderr, "Error: --attn-kernel must be standard or tiled\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--attn-tile") == 0 && i + 1 < argc) {
            args->attn_tile = atoi(argv[++i]);
            if (args->attn_tile < 0) {
                fprintf(stderr, "Error: --attn-tile must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--attn-parallel-min-ctx") == 0 && i + 1 < argc) {
            args->attn_parallel_min_ctx = atoi(argv[++i]);
            if (args->attn_parallel_min_ctx < 0) {
                fprintf(stderr, "Error: --attn-parallel-min-ctx must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
            args->kv_type = kv_type_parse(argv[++i]);
            if (args->kv_type < 0) {
                fprintf(stderr, "Error: --kv-type must be f32, i8 or i4\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--kv-offload") == 0 && i + 1 < argc) {
            args->kv_offload_dir = argv[++i];
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
        } else if (strcmp(argv[i], "--rpc-worker") == 0 && i + 1 < argc) {
            args->rpc_worker_port = atoi(argv[++i]);
            if (args->rpc_worker_port <= 0 || args->rpc_worker_port > 65535) {
                fprintf(stderr, "Error: --rpc-worker must be a port 1-65535\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            args->worker_layers = argv[++i];
        } else if (strcmp(argv[i], "--rpc-bind-all") == 0) {
            args->rpc_bind_all = true;
        } else if (strcmp(argv[i], "--remote-layers") == 0 && i + 1 < argc) {
            args->remote_layers = argv[++i];
        } else if (strcmp(argv[i], "--plan-layers") == 0 && i + 1 < argc) {
            args->plan_layers = argv[++i];
        } else if (strcmp(argv[i], "--queue-size") == 0 && i + 1 < argc) {
            args->queue_size = atoi(argv[++i]);
            if (args->queue_size < 0) {
                fprintf(stderr, "Error: --queue-size must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--queue-timeout") == 0 && i + 1 < argc) {
            args->queue_timeout_s = atoi(argv[++i]);
            if (args->queue_timeout_s < 1) {
                fprintf(stderr, "Error: --queue-timeout must be >= 1 second\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--best-of") == 0 && i + 1 < argc) {
            args->best_of = atoi(argv[++i]);
            if (args->best_of < 1 || args->best_of > 64) {
                fprintf(stderr, "Error: --best-of must be 1..64\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts") == 0 && i + 1 < argc) {
            args->mcts.simulations = atoi(argv[++i]);
            if (args->mcts.simulations < 0 || args->mcts.simulations > 4096) {
                fprintf(stderr, "Error: --mcts must be 0..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-max") == 0 && i + 1 < argc) {
            args->mcts_max = atoi(argv[++i]);
            if (args->mcts_max < 0 || args->mcts_max > 4096) {
                fprintf(stderr, "Error: --mcts-max must be 0..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-expand") == 0 && i + 1 < argc) {
            args->mcts.expand = atoi(argv[++i]);
            if (args->mcts.expand < 1 || args->mcts.expand > 16) {
                fprintf(stderr, "Error: --mcts-expand must be 1..16\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-depth") == 0 && i + 1 < argc) {
            args->mcts.max_depth = atoi(argv[++i]);
            if (args->mcts.max_depth < 1 || args->mcts.max_depth > 256) {
                fprintf(stderr, "Error: --mcts-depth must be 1..256\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-step-tokens") == 0 && i + 1 < argc) {
            args->mcts.step_tokens = atoi(argv[++i]);
            if (args->mcts.step_tokens < 1 || args->mcts.step_tokens > 4096) {
                fprintf(stderr, "Error: --mcts-step-tokens must be 1..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-c") == 0 && i + 1 < argc) {
            args->mcts.c_ucb = (float)atof(argv[++i]);
            if (!(args->mcts.c_ucb > 0.0f && args->mcts.c_ucb <= 10.0f)) {
                fprintf(stderr, "Error: --mcts-c must be in (0, 10]\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-temperature") == 0 && i + 1 < argc) {
            args->mcts.temperature = (float)atof(argv[++i]);
            if (!(args->mcts.temperature > 0.0f && args->mcts.temperature <= 2.0f)) {
                fprintf(stderr, "Error: --mcts-temperature must be in (0, 2]\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-delim") == 0 && i + 1 < argc) {
            /* "\n" in the argument means a newline */
            const char *d = argv[++i];
            size_t k = 0;
            for (; *d && k + 1 < sizeof(args->mcts_delim_buf); d++) {
                if (d[0] == '\\' && d[1] == 'n') { args->mcts_delim_buf[k++] = '\n'; d++; }
                else args->mcts_delim_buf[k++] = *d;
            }
            args->mcts_delim_buf[k] = '\0';
            if (k == 0) {
                fprintf(stderr, "Error: --mcts-delim must not be empty\n");
                return TN_ERR_INVALID_CONFIG;
            }
            args->mcts.step_delim = args->mcts_delim_buf;
        } else if (strcmp(argv[i], "--prm") == 0 && i + 1 < argc) {
            args->prm_path = argv[++i];
        } else if (strcmp(argv[i], "--prm-format") == 0 && i + 1 < argc) {
            args->prm_format = argv[++i];
        } else if (strcmp(argv[i], "--prm-good") == 0 && i + 1 < argc) {
            args->prm_good = argv[++i];
        } else if (strcmp(argv[i], "--prm-bad") == 0 && i + 1 < argc) {
            args->prm_bad = argv[++i];
        } else if (strcmp(argv[i], "--best-of-max") == 0 && i + 1 < argc) {
            args->best_of_max = atoi(argv[++i]);
            if (args->best_of_max < 1 || args->best_of_max > 64) {
                fprintf(stderr, "Error: --best-of-max must be 1..64\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--chat-format") == 0 && i + 1 < argc) {
            args->chat_format = chat_template_parse(argv[++i]);
            if (args->chat_format < -1) {
                fprintf(stderr, "Error: --chat-format must be auto, chatml, llama3, deepseek, mistral or plain\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--ui-title") == 0 && i + 1 < argc) {
            args->ui_title = argv[++i];
        } else if (strcmp(argv[i], "--ui-logo") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "ascii") == 0) args->ui_logo = 0;
            else if (strcmp(m, "text") == 0) args->ui_logo = 1;
            else {
                fprintf(stderr, "Error: --ui-logo must be ascii or text\n");
                return TN_ERR_INVALID_CONFIG;
            }
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
        } else if (strcmp(argv[i], "--raw-prompt") == 0) {
            args->raw_prompt = true;
        } else if (strcmp(argv[i], "--ignore-eos") == 0) {
            args->ignore_eos = true;
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
        } else if (strcmp(argv[i], "--check-update") == 0) {
            args->check_update = true;
        } else if (strcmp(argv[i], "--no-update-check") == 0) {
            args->no_update_check = true;
        } else if (strcmp(argv[i], "--update-repo") == 0 && i + 1 < argc) {
            args->update_repo = argv[++i];
            if (!update_repo_valid(args->update_repo)) {
                fprintf(stderr, "Error: --update-repo must be owner/name\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-api") == 0 && i + 1 < argc) {
            args->update_api = argv[++i];
            if (!update_api_base_valid(args->update_api)) {
                fprintf(stderr, "Error: --update-api must be an https:// URL\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-interval") == 0 && i + 1 < argc) {
            args->update_interval_h = atoi(argv[++i]);
            if (args->update_interval_h < 0 || args->update_interval_h > 24 * 365) {
                fprintf(stderr, "Error: --update-interval must be 0-8760 hours\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-channel") == 0 && i + 1 < argc) {
            args->update_channel = update_channel_parse(argv[++i]);
            if (args->update_channel < 0) {
                fprintf(stderr, "Error: --update-channel must be auto, stable or prerelease\n");
                return TN_ERR_INVALID_CONFIG;
            }
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

    if (args->check_update) return TN_OK; /* needs no model */

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
    gp->raw_prompt = args->raw_prompt;
    gp->ignore_eos = args->ignore_eos;
    gp->system_prompt = args->system_prompt;
    for (int i = 0; i < args->n_stop && i < GEN_MAX_STOP; i++) gp->stop[i] = args->stop[i];
    gp->n_stop = args->n_stop < GEN_MAX_STOP ? args->n_stop : GEN_MAX_STOP;
    gp->watermark.enabled = args->watermark;
    gp->watermark.key     = watermark_key_from_string(args->watermark_key);
    gp->watermark.gamma   = args->watermark_gamma;
    gp->watermark.delta   = args->watermark_delta;
    watermark_config_sanitize(&gp->watermark);
}

void args_fill_update_options(const CliArgs *args, UpdateOptions *o) {
    update_options_default(o);
    if (args->no_update_check) o->enabled = 0;
    if (args->update_interval_h >= 0) o->interval_hours = args->update_interval_h;
    o->channel = args->update_channel;
    if (args->update_repo) snprintf(o->repo, sizeof(o->repo), "%s", args->update_repo);
    if (args->update_api) snprintf(o->api_base, sizeof(o->api_base), "%s", args->update_api);
}
