# Changelog

All notable changes to Project Zero are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major
version is 0, a minor bump (0.x.0) may carry breaking changes; they are always listed under
**Changed**, **Deprecated** or **Removed**.

Every user-facing change adds a line under **[Unreleased]** in the same commit. At release time the
section is renamed to the version and date, and the release workflow publishes that section as the
GitHub release notes (`tools/changelog.py notes <version>`; see `docs/RELEASING.md`).

## [Unreleased]

### Changed

- **The binary is now `pz`** (was `adaptive_ai_engine`). `pz --version` prints
  `Project Zero <version>`. Release downloads are named
  `project-zero-<version>-x86_64-unknown-linux-gnu.tar.gz` (was
  `adaptive_ai_engine-<version>-x86_64-linux.tar.gz`) and unpack into a directory of the same name.
- Startup takes under 1–8 s instead of 20–70 s: the DRAM bandwidth probe is cached per host
  (`--calibrate` or `PZ_BW_REPROBE=1` re-measures), large buffers are zeroed lazily and tokenizer
  lookups use the sorted index.
- Lower memory use: converted tensors release their file pages, and KV cache rows are committed only
  as far as a generation can reach.
- The INT8/INT4 classifier (LM head) quantizes block-wise instead of with one scale per row.

### Deprecated

- The `adaptive_ai_engine` name. `make` and the release archive still provide it as a symlink to
  `pz`, which prints a one-line notice; it will be removed in the release after the next one.

### Added

- **Models and formats:** Qwen 3.5/3.6 hybrid attention (e.g. Ternary-Bonsai-27B) and the Q2_0
  quant; Qwen3-MoE GGUF; Mamba-1 state-space models; every GGUF quant type in mainline llama.cpp
  (Q4_1, Q8_1, Q8_K, BF16, IQ1–IQ4 families); BitNet I2_S, NEOX RoPE and BPE pre-tokenizer parity
  with llama.cpp; `--perplexity` for accuracy checks; `--raw-prompt`.
- **Generation:** speculative decoding (`--draft-model`, `--spec-length`); LoRA adapters (`--lora`);
  grammar-constrained JSON mode; a configurable sampler chain (`--top-k --min-p --repeat-penalty
  --presence-penalty --frequency-penalty --repeat-last-n`) with stop sequences (`--stop`);
  toggleable cryptographic watermarking (`--watermark*`); best-of-N; step-level tree search with an
  optional process reward model (`--mcts*`, `--prm*`); chat templates replaceable from a file
  (`--chat-template-file`) or per request.
- **Context and KV cache:** `--ctx-size`, `--kv-ram-fraction`; int8 and int4 KV storage
  (`--kv-type f32|i8|i4`); disk-backed KV (`--kv-offload <dir>`); KV prefix reuse across requests
  and a multi-entry prompt snapshot cache (`--prompt-cache`); tiled, head-parallel attention.
- **Serving:** OpenAI-compatible API hardening (CORS and API key, both opt-in; metrics; OpenAPI docs;
  cancel); `GET /v1/config` publishing defaults, ranges and features; `POST /v1/watermark/detect`;
  a FIFO generation queue (`--queue-size`, `--queue-timeout`); graceful Ctrl+C/SIGTERM shutdown;
  distributed pipeline inference over TCP (`--remote-layers`, `--rpc-worker`); notify-only update
  checks (`--check-update`, `GET /v1/update`, `--no-update-check`).
- **Web UI:** chat with image upload; settings rendered from `/v1/config` (every generation knob,
  chat template editor, tree search); watermark panel; usage footer; update banner; brand wordmark.
- **CLI:** colour, progress, live token statistics, markdown rendering and the startup banner.
- **Version history:** this changelog follows Keep a Changelog; GitHub release notes are generated
  from it. `GET /v1/update` returns `release_notes_url` and `changelog_url` for the running build
  (derived from `--update-repo`/`--update-api`), shown in the web UI under Settings → Updates.
- **Kernels:** K-6 5-trit LUT ternary kernel (AVX-512BW); plain-NEON ternary kernel; AVX-512 VNNI
  Q2_0; llama.cpp-compatible Q4_K x8 repack with multi-row GEMV; fused zero-copy Q6_K; batched
  verify kernels for speculative decoding.

### Fixed

- Crash (`SIGILL`) on CPUs that advertise AVX-512 VBMI in CPUID but cannot execute it.
- Generation not stopping at a GGUF model's end-of-turn tokens (all declared EOS/EOT/EOM tokens and
  end-of-turn control tokens now stop it).
- `--lora` was ignored when combined with `--draft-model`.
- The INT8/INT4/BF16 classifier option had no effect on Q2_0-native models.
- Wrong output from IQ4_NL weights (nibble packing) and mojibake from byte-level BPE detokenizing.
- The JSON-mode grammar accepted numbers with leading zeros.
- A data race in thread pool shutdown.
- Saved calibration overriding the automatically chosen SIMD tier; the DRAM bandwidth probe
  under-reporting about 3×; hardware ceilings computed from fixed BitNet-2B sizes for every model.
- Release builds silently linking objects compiled for a different build variant.

### Security

- Static web assets: requests could escape `--static-dir` via path traversal.
- Agent tool sandbox: argument-level command policy, and fixes for an exit-code race, a symlink
  escape and `PATH` hijacking.

## [0.1.0] - 2026-06-20

First release with a prebuilt, portable x86-64 Linux binary
(`adaptive_ai_engine-0.1.0-x86_64-linux.tar.gz`, runtime CPU dispatch, static libstdc++/libgcc).

### Added

- BitNet b1.58 ternary inference with packed 2-bit weights and runtime SIMD dispatch: scalar, AVX2,
  AVX-512F, AVX-512 VNNI, AVX-VNNI, ARM NEON and dotprod; int8 activation quantization.
- GGUF loading (Q8_0, Q4_K, I2_S and tokenizer extraction) with an architecture router; DeepSeek-V2
  mixture-of-experts routing and multi-head latent attention.
- Sampling and text generation, KV cache strategies (int8, sliding window), hidden reasoning,
  agentic tool execution, local RAG memory and a vision encoder bridge.
- An OpenAI-compatible HTTP API, CLI and REPL.
- Conversion tools: HuggingFace BitNet and dense models, tokenizers, one-click model import.

### Fixed

- Memory-safety bugs found by the Phase 10 and QA audits: a stack overflow in top-k sampling, a
  buffer overflow in the thought filter, file-handle and descriptor leaks in model and tokenizer
  loading, a NULL dereference in token decoding, wrong value-cache indexing in attention, and an RNG
  re-seeded on every call.

[Unreleased]: https://github.com/shifulegend/project-zero/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shifulegend/project-zero/releases/tag/v0.1.0
