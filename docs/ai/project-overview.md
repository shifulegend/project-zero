# Project Overview — project-zero

> Canonical source of truth. Tool adapters (CLAUDE.md, AGENTS.md, gemini/GEMINI.md,
> .github/copilot-instructions.md) summarize and link here. Keep this current.
> Last updated: 2026-07-15.

## Purpose
`project-zero` is a from-scratch, CPU-optimized LLM inference engine in C/C++ targeting
**BitNet b1.58 ternary weights** and **DeepSeek-V2** architecture (MoE + MLA attention).
The GGUF loader (`config_from_gguf()` in `src/core/gguf_loader.c`) is **architecture-agnostic**
— it keys metadata off the GGUF `general.architecture` string — so **dense GGUF transformers**
(Llama-family) also run through a generic path; only DeepSeek-V2 is special-cased for MoE + MLA.
**SmolLM2-135M-Instruct F16 is the verified dense model** (up to 83.79 tok/s); other standard
architectures (`llama`/`qwen`/`mistral`/`gemma`/`phi`) load but are untested. There is also a
**partial** OpenAI-compatible HTTP API layer (Phase 21 — see below). Goal: high single-machine
CPU throughput with SIMD-tuned kernels, no GPU.

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
| `src/cli/` | Arg parsing, REPL, `main.c` (builds `adaptive_ai_engine`); color/progress/live-stats/markdown-render (Phase 22.3, REPL-only) |
| `include/` | Public headers mirroring `src/` |
| `tests/` | Unit + audit + red/blackbox tests (every `tests/*.c` is auto-built & run) |
| `tools/` | Model conversion + benchmark scripts |
| `webui/` | Web chat UI frontend source (Vite+Svelte, Phase 22); build output is embedded into the binary, not shipped as a directory |
| `docs/` | Reports + this `docs/ai/` AI-dev system |
| `docs/design/` | UI/UX design-principles reference used to grade screenshots (Phase 22) |

## Build / test / run (verified)
```bash
make release CC=gcc            # or CC=clang; optimized engine + libs
make test    CC=gcc            # builds & runs EVERY tests/*.c (ASan/UBSan), aborts on first fail
make debug   CC=gcc            # -O0 -g -march=native -fsanitize=address,undefined
make dist    CC=gcc            # PORTABLE binary: -march=x86-64-v2 + per-TU SIMD ISA +
                               # static libstdc++/libgcc; runtime AVX2/AVX-512/VNNI dispatch
cmake -B build && cmake --build build -j$(nproc)   # alternative build (cmake -DPZ_DIST=ON = dist)
./adaptive_ai_engine --version  # version + detected SIMD backend (no model needed)
./adaptive_ai_engine --model models/<m>.gguf --prompt "..." \
  --max-tokens 16 --temperature 0.0 --threads 4 [--simd auto] [--classifier auto]
```
CI: `.github/workflows/ci.yml` (release/test/debug/dist × gcc,clang × ubuntu-latest,ubuntu-22.04 +
macOS release/test), `.github/workflows/security_audit.yml` (cmake+ASan/UBSan + `tools/fuzz_config.py`),
and `.github/workflows/release.yml` (tag `v*` → portable `make dist` binary on a GitHub Release;
see `docs/RELEASING.md`).

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
- **HTTP API** (`src/api/`) exposes the engine; mirrors OpenAI chat schema. **Phase 21** shipped
  `POST /v1/chat/completions` (streaming + non-streaming SSE), `GET /v1/models`, `GET /health`.
  **Phase 22.1** hardened this into a real server: CORS, optional API-key auth, `/metrics`
  (Prometheus), `/docs`+`/openapi.json`, a cancel endpoint, and a concurrency rearchitecture
  (per-connection threads + a `generation_mutex`, replacing the old serial-inline-accept-loop
  model) so static asset/UI serving isn't blocked behind an in-flight generation. Still binds
  loopback-only by default. Logic-level tests in `tests/test_api_server.c` plus
  `tests/test_cors.c`/`test_auth.c`/`test_metrics.c`/`test_cancel.c`/`test_openapi.c`.
- **Web chat UI** (Phase 22.2, `webui/`): a Vite+Svelte SPA served by the HTTP API at `GET /`
  (`GET /health` remains the separate JSON health check). Built bundle is embedded into the C
  binary as a committed generated TU (`src/api/webui_bundle_generated.c`) — default builds need
  no Node; `make webui-bundle` regenerates it when `webui/src` changes. Supports streaming chat,
  adjustable sampling params, stop/cancel, theme toggle, and image upload via the OpenAI
  "content parts" form (`src/api/data_url.c` decodes the base64 payload; the vision pipeline
  itself lives in `src/multimodal/vision_pipeline.c`, shared with the CLI's `--image`, active
  when the server is started with `--vision`/`--proj`).
- **Conversion tools** (`tools/convert_*`, `import_model.py`) bridge HuggingFace → engine formats.

## Reference reports
`docs/REGRESSION_VERIFICATION_2026-06-07.md`, `BENCHMARK_REPORT.md`,
`docs/PERFORMANCE_CEILING_REPORT.md`, `GOLDEN_RULES.md`.

## UNKNOWN / TODO
- Vision/multimodal subsystem is experimental; depth of support is UNVERIFIED.
- GitNexus knowledge graph index (`.gitnexus/`) is not built in CI.
