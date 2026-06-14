# Project Overview — project-zero

> Canonical source of truth. Tool adapters (CLAUDE.md, AGENTS.md, gemini/GEMINI.md,
> .github/copilot-instructions.md) summarize and link here. Keep this current.
> Last updated: 2026-06-07.

## Purpose
`project-zero` is a from-scratch, CPU-optimized LLM inference engine in C/C++ targeting
**BitNet b1.58 ternary weights** and **DeepSeek-V2** architecture (MoE + MLA attention),
plus a **partial** OpenAI-compatible HTTP API layer (Phase 21 — see below). Goal: high
single-machine CPU throughput with SIMD-tuned kernels, no GPU.

## Stack & key dependencies
- **Languages:** C99 (engine). One **temporary** C++17 translation unit
  (`src/tokenizer/chat_template.cpp`) is slated for a C port — target is 100% C
  (tracked in `.github/ROADMAP.md` → "Language & Dependency Goals").
- **Build systems:** `Makefile` (primary, per-file SIMD flag control) and `CMakeLists.txt`
  (used by the security-audit workflow). Both are first-class — keep them in sync.
- **Runtime deps:** POSIX threads (`-pthread`), libm, libstdc++ (C tests link C++ objects).
- **SIMD:** AVX2 / AVX-512F / AVX-512VNNI / AVX-VNNI(256) / ARM NEON+dotprod, runtime-dispatched.
- **Model format:** GGUF (llama.cpp-compatible) + a native packed `.bin` for BitNet.
- **No third-party ML libs** — kernels, GGUF reader, tokenizer, sampler are all in-tree.
- **Python (tools only, temporary):** conversion/dev/test scripts in `tools/` use
  `huggingface_hub`, `torch`, `transformers`, `safetensors` (not needed to build/run the
  engine). The final product targets **zero Python** — see `.github/ROADMAP.md`.

## Architecture overview
Load (GGUF/`.bin` → weights, mmap) → tokenize (GGUF tokenizer + Jinja-style chat template) →
forward pass (embedding → per-layer {RMSNorm, attention (MHA/GQA or MLA), FFN/MoE} → final
RMSNorm → classifier) → sample → decode. SIMD backend and classifier quantization are selected
at runtime (`--simd`, `--classifier`, or auto/calibration).

## Important directories
| Path | Purpose |
|------|---------|
| `src/core/` | Weight/GGUF loading, quantization, MoE config, run-state, hardware profile |
| `src/math/` | SIMD matmul/elementwise kernels (F16, Q4K, Q5_0/1, Q5K, Q2K, Q8_0, ternary VNNI) |
| `src/transformer/` | Attention (MHA/GQA), MLA attention, MoE routing/FFN, FFN, generation loop |
| `src/tokenizer/` | GGUF tokenizer (encode/decode/load), chat template (C++) |
| `src/api/` | HTTP server, SSE streaming, JSON parse, chat compilation |
| `src/kv_cache/` | KV cache strategy/compression, sliding window |
| `src/sampling/` | RNG, temperature, top-k, top-p, argmax |
| `src/memory/` | Aligned alloc, mmap |
| `src/multimodal/` | Vision encoder/projector/bridge, image load (optional/experimental) |
| `src/cli/` | Arg parsing, REPL, `main.c` (builds `adaptive_ai_engine`) |
| `include/` | Public headers mirroring `src/` |
| `tests/` | Unit + audit + red/blackbox tests (every `tests/*.c` is auto-built & run) |
| `tools/` | Model conversion + benchmark scripts |
| `docs/` | Reports + this `docs/ai/` AI-dev system |

## Build / test / run (verified)
```bash
make release CC=gcc            # or CC=clang; optimized engine + libs
make test    CC=gcc            # builds & runs EVERY tests/*.c (ASan/UBSan), aborts on first fail
make debug   CC=gcc            # -O0 -g -march=native -fsanitize=address,undefined
cmake -B build && cmake --build build -j$(nproc)   # alternative build
./adaptive_ai_engine --model models/<m>.gguf --prompt "..." \
  --max-tokens 16 --temperature 0.0 --threads 4 [--simd auto] [--classifier auto]
```
CI: `.github/workflows/ci.yml` (release/test/debug × gcc,clang × ubuntu-latest,ubuntu-22.04 + macOS
release/test) and `.github/workflows/security_audit.yml` (cmake+ASan/UBSan + `tools/fuzz_config.py`).

## Domain terminology
- **Ternary / b1.58** — weights in {-1,0,+1}, packed 4/byte.
- **MoE** — mixture-of-experts; router picks top-k experts per token.
- **MLA** — multi-head latent attention (DeepSeek-V2), compressed KV.
- **Classifier** — the LM head; quantizable (bf16/int8/int4/auto-fast).
- **Backend** — selected SIMD kernel family (scalar/avx2/avx512f/vnni).
- **tok/s** — generation throughput, the primary perf metric.

## Major integration boundaries
- **GGUF metadata** drives config, tokenizer, and quant types — *not* hardcoded constants.
- **Runtime SIMD dispatch** (`src/math/simd_dispatch.c`, `cpu_features.c`) selects kernels.
- **HTTP API** (`src/api/`) exposes the engine; mirrors OpenAI chat schema. **Partial
  (Phase 21, experimental):** `--server`/`--port` serve `POST /v1/chat/completions`
  (streaming + non-streaming SSE), `GET /v1/models`, `GET /health` with real inference,
  but the listener handles connections serially, binds loopback-only, and the socket
  layer is untested/not in CI. Logic-level tests in `tests/test_api_server.c`.
- **Conversion tools** (`tools/convert_*`, `import_model.py`) bridge HuggingFace → engine formats.

## Reference reports
`docs/REGRESSION_VERIFICATION_2026-06-07.md`, `BENCHMARK_REPORT.md`,
`docs/PERFORMANCE_CEILING_REPORT.md`, `GOLDEN_RULES.md`.

## UNKNOWN / TODO
- Vision/multimodal subsystem is experimental; depth of support is UNVERIFIED.
- GitNexus knowledge graph index (`.gitnexus/`) is not built in CI.
