/*
 * src/cli/args_usage.c
 *
 * The --help text. Split out of args.c; the text is unchanged (`pz --help` output is identical).
 */

#include <stdio.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/args.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

static void usage_intro(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("       %s config        set up defaults interactively (pz config --help)\n", prog_name);
    printf("Saved settings (pz config) and PZ_<KEY> environment variables act as defaults;\n"
           "flags given here override them. --no-config ignores both.\n");
}

static void usage_generation(void) {
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
    printf("                      (e.g. docs/templates/bitnet-b1.58.jinja); default: the GGUF's "
           "own\n");
    printf(
        "                      (BitNet's, which generates nothing, is replaced by a built-in)\n");
    printf("  --no-request-template  Server: ignore a request's \"chat_template\" field\n");
    printf("  --no-config-write   Server: the web UI may not save defaults to the config file\n");
    printf("  --stop <text>       Stop generation at this string; repeatable (max %d)\n",
           GEN_MAX_STOP);
    printf("  --best-of <N>       Generate N candidates, keep the one the model finds\n");
    printf("                      most likely (mean log-prob). Default 1 (off).\n");
    printf("  --best-of-max <N>   Server: largest best_of a request may ask for (default %d)\n",
           BEST_OF_MAX_DEFAULT);
    printf("  --mcts <N>          Step-level tree search with N simulations (0 = off): sample\n");
    printf("                      --mcts-expand children per step, score each step, answer\n");
    printf("                      along the most visited path\n");
    printf("  --mcts-expand <k> --mcts-depth <d> --mcts-step-tokens <n> --mcts-c <c>\n");
    printf("  --mcts-temperature <t> --mcts-delim <s>   search shape (defaults %d %d %d %.1f %.1f "
           "\"\\n\")\n",
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
    printf("                      context reaches N tokens (default %d, 0 = never)\n",
           ATTN_PARALLEL_MIN_CTX_DEFAULT);
}

static void usage_watermark(void) {
    printf("\nWatermarking (off by default):\n");
    printf("  --watermark         Bias sampling toward a keyed green list so output can\n");
    printf("                      later be identified (statistical, not visible).\n");
    printf("  --watermark-key <s> Secret passphrase for the green list.\n");
    printf("  --watermark-gamma <g> Green-list fraction (default: 0.25).\n");
    printf("  --watermark-delta <d> Logit bias for green tokens (default: 2.0).\n");
    printf("  --watermark-detect <text> Score <text> for the watermark (needs --model for\n");
    printf("                      the tokenizer and the same key/gamma), then exit.\n");
}

static void usage_hardware(void) {
    printf("\nHardware & Performance:\n");
    printf("  --threads <int>     Number of threads (default: auto)\n");
    printf("  --classifier <fmt>  Classifier quantization: auto, bf16, int8, int4, auto-fast\n");
    printf("                      Default: auto (BF16, full intelligence, zero loss).\n");
    printf("                      int8/int4: faster but quantizes classifier weights.\n");
    printf("                      auto-fast: let engine pick fastest quantized mode.\n");
    printf("  --simd <backend>    Force SIMD backend: auto, avx2, avx512f, vnni, scalar\n");
    printf("                      Default: auto = the highest tier the CPU supports, changed\n");
    printf("                      only when calibration measures another tier clearly faster\n");
    printf("                      (--no-calibrate: the highest tier, always). Use avx2 on Tiger\n");
    printf("                      Lake to avoid AVX-512 frequency throttle.\n");
    printf("  --act-quant <m>     Activation numerics of ternary matmuls: auto (default), int8\n");
    printf("                      (the BitNet W1.58A8 definition; fastest with VNNI/dotprod) or\n");
    printf("                      float (float32 activations, slower). Independent of --simd.\n");
    printf("  --quality-tolerance <pct>  Largest perplexity increase (percent, default 1.0) a\n");
    printf("                      fast setting may cost against float activations and a BF16\n");
    printf("                      head. An explicit --classifier int8/int4/auto-fast that\n");
    printf("                      exceeds it falls back to bf16 at startup; 0 = no bar.\n");
    printf("  --self-check        Run the accuracy probe on this model and these settings,\n");
    printf("                      print the result and exit (status 1 when over the tolerance).\n");
    printf("  --self-check-tokens <n>  Probe length in tokens (default 96).\n");
    printf("  --calibrate         Run micro-benchmarks to find optimal settings for\n");
    printf("                      this hardware. Results cached in ~/.project-zero/\n");
    printf("                      Runs automatically on first use or hardware change.\n");
    printf("  --no-calibrate      Never load, run or save a calibration (also "
           "PZ_NO_CALIBRATE=true):\n");
    printf("                      the highest SIMD tier and the detected thread count, for\n");
    printf("                      reproducible benchmarks and throwaway containers.\n");
}

static void usage_server(void) {
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
    printf("  --rate-limit <n>    Requests per second allowed on the non-generation routes (20; 0 "
           "= off)\n");
    printf("  --client-timeout <s> Drop a client that sends or reads nothing for this many seconds "
           "(30; 0 = never)\n");
    printf("  --max-prompt-bytes <N> Server: message text one chat request may carry; more gets "
           "400 (1048576; 0 = no cap)\n");
    printf("  --max-connections <N> Open client connections at once; more get 503 (64; 0 = "
           "unlimited)\n");
    printf("  --parallel <N>      Serve up to N chat requests at once, batched into one\n");
    printf("                      forward pass per token (default: 1 = one at a time;\n");
    printf("                      max %d). Dense models only; others stay sequential.\n",
           PZ_PARALLEL_MAX);
    printf("  --parallel-batch <N> Tokens per batched pass: every active reply's next\n");
    printf("                      token plus a chunk of waiting prompts (default: %d)\n",
           PZ_PARALLEL_BATCH_DEFAULT);
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
}

static void usage_distributed(void) {
    printf("\nDistributed layers (pipeline across processes/machines):\n");
    printf("  --rpc-worker <port> Serve a layer range for a coordinator, then exit on error.\n");
    printf("  --layers <A-B>      Layers [A, B) this worker runs (with --rpc-worker).\n");
    printf("  --rpc-bind-all      Worker listens on all interfaces (default: 127.0.0.1 only;\n");
    printf("                      the protocol has no authentication -- trusted LANs only).\n");
    printf("  --remote-layers <s> Coordinator: host:port@A-B[,host:port@C-D...] — run those\n");
    printf("                      layers on workers started with the same model file.\n");
    printf("  --plan-layers <s>   Print a layer split proportional to each node's RAM in GB\n");
    printf("                      (e.g. 16,8,8 — first entry is this coordinator), then exit.\n");
}

static void usage_vision(void) {
    printf("\nMultimodal (vision):\n");
    printf("  --image <path>      Path to image file for multimodal queries (optional)\n");
    printf("  --vision <path>     Path to vision.bin encoder weights\n");
    printf("  --proj <path>       Path to projector.bin weights\n");
    printf("                      Both --vision and --proj required to use --image.\n");
    printf("                      Extract with: python tools/extract_multimodal.py\n");
}

static void usage_memory(void) {
    printf("\nMemory & RAG:\n");
    printf("  --memory-db <path>  Path to vector DB file for RAG memory.\n");
    printf("                      File is created if it does not exist.\n");
}

static void usage_speculative(void) {
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
}

static void usage_lora(void) {
    printf("\nLoRA adapters:\n");
    printf("  --lora <path>       Path to a .lora.bin adapter. Disabled by default --\n");
    printf("                      this is the ONLY way to enable a LoRA adapter; there\n");
    printf("                      is no default/embedded adapter. The adapter must have\n");
    printf("                      been built for this exact base model (n_layers, dim,\n");
    printf("                      hidden_dim all checked at load time). v1 scope: the\n");
    printf("                      generic dense/GQA attention/FFN path only -- not MLA,\n");
    printf("                      Qwen3-MoE, Qwen3.5/3.6 hybrid, or MoE-FFN layers.\n");
}

static void usage_output(void) {
    printf("\nOutput:\n");
    printf("  --color <mode>      Color output: auto (default), always, never.\n");
    printf("                      auto respects the NO_COLOR env var and disables\n");
    printf("                      color when stdout isn't a terminal.\n");
    printf("  --verbose           Enable verbose logging (same as --log-level debug)\n");
    printf("  --log-format <fmt>  Diagnostics on stderr: text (default) or json (one object per\n");
    printf("                      line: ts, level, component, msg)\n");
    printf("  --log-level <lvl>   Lowest level written: debug, info (default), warn, error\n");
    printf("  --dump-tensors <f>  Write intermediate tensor stats to a CSV file\n");
    printf("  --version, -v       Print version and the SIMD backend for this CPU, then exit.\n");
    printf("  --help, -h          Print this help and exit.\n");
}

static void usage_updates(void) {
    printf("\nUpdates (notify only -- nothing is downloaded or installed):\n");
    printf("  --check-update      Check GitHub Releases now, print the result, exit\n");
    printf("                      (0 up to date/unknown, 3 update available, 1 error).\n");
    printf("  --no-update-check   Never check. Also: PZ_NO_UPDATE_CHECK=1, DO_NOT_TRACK=1;\n");
    printf("                      always off when CI is set. Otherwise interactive runs and\n");
    printf("                      --server check in the background at most once per\n");
    printf("                      --update-interval (sends only a User-Agent with the version).\n");
    printf("  --update-interval <h> Hours between checks (default %d, 0 = off).\n",
           UPDATE_DEFAULT_INTERVAL_H);
    printf(
        "  --update-channel <c> auto (default: pre-releases while on 0.x) | stable | prerelease\n");
    printf("  --update-repo <o/n> Releases source (default: upstream; forks set their own).\n");
    printf("  --update-api <url>  API base, e.g. GitHub Enterprise (default "
           "https://api.github.com).\n");
}

static void usage_examples(const char *prog_name) {
    printf("\nExamples:\n");
    printf("  %s --model models/smollm2.gguf --prompt \"What is the capital of France?\"\n",
           prog_name);
    printf("  %s --model models/smollm2.gguf --threads 4          # interactive REPL\n", prog_name);
    printf("  %s --model models/smollm2.gguf --server --port 8080 --cors --cors-origin \"*\"\n",
           prog_name);
}

void print_usage(const char *prog_name) {
    usage_intro(prog_name);
    usage_generation();
    usage_watermark();
    usage_hardware();
    usage_server();
    usage_distributed();
    usage_vision();
    usage_memory();
    usage_speculative();
    usage_lora();
    usage_output();
    usage_updates();
    usage_examples(prog_name);
}
