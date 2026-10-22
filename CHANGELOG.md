# Changelog

All notable changes to Project Zero are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major
version is 0, a minor bump (0.x.0) may carry breaking changes; they are always listed under
**Changed**, **Deprecated** or **Removed**.

Every user-facing change adds a line under **[Unreleased]** in the same commit; at release time the
section becomes the version's entry and is published as its GitHub release notes. Writing rules:
[`docs/ai/changelog-guidance.md`](docs/ai/changelog-guidance.md); release steps:
[`docs/RELEASING.md`](docs/RELEASING.md).

## [Unreleased]

### Changed

- **Breaking:** Rename the binary to `pz` (was `adaptive_ai_engine`); `pz --version` prints `Project Zero <version>`.
- **Breaking:** Rename release downloads to `project-zero-<version>-x86_64-unknown-linux-gnu.tar.gz`, unpacking into a directory of the same name (was `adaptive_ai_engine-<version>-x86_64-linux.tar.gz`).
- Publish each GitHub release's notes from this changelog instead of an auto-generated commit list.
- Cut startup from 20–70 s to under 1–8 s by caching the DRAM bandwidth probe per host (`--calibrate` or `PZ_BW_REPROBE=1` re-measures), zeroing large buffers lazily and indexing tokenizer lookups.
- Reduce resident memory by releasing the file pages of converted tensors and committing KV cache rows only as far as a generation can reach.
- Quantize the INT8/INT4 classifier (LM head) block-wise instead of with one scale per row.
- Speed up long-context decode with `--kv-type i4` by about 27% with AVX2/AVX-512 int4 KV kernels: SmolLM2-1.7B Q4_K_M, about 2,200-token context, 4 threads: 8.32 → 10.58 tok/s on a 4-vCPU AVX-512 Xeon.
- Speed up mixture-of-experts decode about 1.7–1.8× with a fused expert FFN, 8-bit-activation integer kernels and dynamic expert scheduling: DeepSeek-V2-Lite-Chat Q4_K_S at 4 threads goes from 6.3–7.3 to 11.7–12.6 tok/s (llama.cpp: 9.4–10.5) on a 4-vCPU AVX-512 VNNI Xeon.
- Keep Q4_K/Q6_K output layers and Q5_0/Q5_1/Q8_0 projections in their quantized format instead of widening them, reducing DeepSeek-V2-Lite resident memory from 12.0 to 10.6 GB.

### Deprecated

- Deprecate the `adaptive_ai_engine` name: `make` and the release archive still provide it as a symlink to `pz` that prints a notice; it is removed in the release after the next one.

### Added

- Add Qwen 3.5/3.6 hybrid-attention models (e.g. Ternary-Bonsai-27B) and the Q2_0 quant.
- Add Qwen3-MoE GGUF models.
- Add Mamba-1 state-space models.
- Add every remaining mainline GGUF quant type: Q4_1, Q8_1, Q8_K, BF16 and the IQ1–IQ4 families.
- Add `--perplexity <file>` (with `--ppl-ctx`, `--ppl-chunks`) and `--raw-prompt` for accuracy checks against llama.cpp.
- Add speculative decoding with `--draft-model` and `--spec-length`.
- Add hot-swappable LoRA adapters (`--lora`).
- Add grammar-constrained JSON mode.
- Add a configurable sampler chain (`--top-k`, `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`, `--repeat-last-n`) and stop sequences (`--stop`).
- Add toggleable cryptographic watermarking (`--watermark*`, `POST /v1/watermark/detect`).
- Add best-of-N selection and step-level tree search with an optional process reward model (`--mcts*`, `--prm*`, request field `mcts`).
- Add chat-template overrides from a file (`--chat-template-file`) or per request (`chat_template`; disable with `--no-request-template`).
- Add context sizing flags `--ctx-size` and `--kv-ram-fraction`.
- Add int8 and int4 KV cache storage (`--kv-type f32|i8|i4`).
- Add disk-backed KV cache (`--kv-offload <dir>`).
- Add KV prefix reuse across API requests and a multi-entry prompt snapshot cache (`--prompt-cache`, request field `cache_prompt`).
- Add opt-in CORS (`--cors`, `--cors-origin`) and API-key auth (`--api-key`), `/metrics`, OpenAPI docs at `/docs` and request cancellation.
- Add `GET /v1/config`, publishing server defaults, valid ranges and feature flags.
- Add a FIFO generation queue (`--queue-size`, `--queue-timeout`).
- Add graceful shutdown on Ctrl+C/SIGTERM in `--server` mode.
- Add distributed pipeline inference over TCP (`--remote-layers`, `--rpc-worker`).
- Add notify-only update checks (`--check-update`, `GET /v1/update`, `--no-update-check`, `--update-*`).
- Add `CHANGELOG.md` to the release archive.
- Add release-notes and changelog links for the running build to `GET /v1/update` (`release_notes_url`, `changelog_url`) and to the web UI under Settings → Updates.
- Add the web chat UI with image upload, settings rendered from `/v1/config` (including the chat-template editor and tree search), a watermark panel, a usage footer and an update banner.
- Add colour, progress, live token statistics, markdown rendering and a startup banner to the CLI.
- Add persistent settings: `pz config` (interactive setup), `pz config list|get|set|add|unset|path`, a config file at `~/.config/project-zero/config.toml` (`PZ_CONFIG` overrides the location) and `PZ_<KEY>` environment variables. Precedence is command-line flag > environment > file > default; `--no-config` ignores the file and environment.
- Add Settings → Server defaults to the web UI: see whether each server value comes from the config file, an environment variable or a command-line flag, and save changed settings (and the web UI title) as server defaults for the next start (`GET /v1/config` `config`, `POST /v1/config/defaults`; disable with `--no-config-write`).
- Add `install.sh`: a one-line, checksum-verified installer for Linux x86-64 that puts `pz` in `~/.local/bin` without root, finds the newest release, offers `pz config`, and uninstalls with `--uninstall`; it ships with each release and is listed in `SHA256SUMS`.
- Add faster kernels: K-6 5-trit LUT ternary (AVX-512BW), plain NEON ternary, AVX-512 VNNI Q2_0, llama.cpp-compatible Q4_K x8 repack, fused zero-copy Q6_K and batched speculative-verify kernels.

### Fixed

- Fix `pz config set repeat-last-n -1` being rejected although `--repeat-last-n -1` is valid.
- Fix JSON mode (`--json`, `response_format: json_object`) sometimes starting with a bare string or padding with whitespace at high temperature: output now always starts with `{` or `[` and ends at the matching bracket.
- Fix a nonexistent or unreadable `--model` path being reported only after the first-run hardware calibration, which can take minutes.
- Fix a `SIGILL` crash on CPUs that advertise AVX-512 VBMI in CPUID but cannot execute it.
- Fix the macOS build, broken by a Linux-only call in the disk-backed KV cache (`--kv-offload`).
- Fix oversized buffer requests on macOS and Windows getting the process killed instead of failing with an out-of-memory error.
- Fix AVX-512 VNNI kernels being disabled on every CPU by a startup self-test that expected the wrong result.
- Fix mixture-of-experts routing ignoring the GGUF gating function, weight normalization and weight scale (affected Qwen3-MoE and DeepSeek-V2).
- Fix generation not stopping at a GGUF model's end-of-turn tokens.
- Fix the CLI silently generating nothing when a prompt cannot be tokenized; it now prints an error (the API already returned 400).
- Fix `--lora` being ignored when combined with `--draft-model`.
- Fix the INT8/INT4/BF16 classifier option having no effect on Q2_0-native models.
- Fix wrong output from IQ4_NL weights and mojibake from byte-level BPE detokenizing.
- Fix JSON mode accepting numbers with leading zeros.
- Fix a data race in thread pool shutdown.
- Fix a saved calibration overriding the automatically chosen SIMD tier.
- Fix the DRAM bandwidth probe under-reporting about 3× and hardware ceilings using fixed BitNet-2B sizes for every model.
- Fix release builds silently linking objects compiled for a different build variant.

### Security

- Fix path traversal out of `--static-dir` when serving web assets.
- Harden the agent tool sandbox with an argument-level command policy and fix an exit-code race, a symlink escape and `PATH` hijacking.

## [0.1.0] - 2026-06-20

_First release with a prebuilt, portable x86-64 Linux binary (`adaptive_ai_engine-0.1.0-x86_64-linux.tar.gz`)._

### Added

- Add BitNet b1.58 ternary inference with packed 2-bit weights and int8 activation quantization.
- Add runtime SIMD dispatch across scalar, AVX2, AVX-512F, AVX-512 VNNI, AVX-VNNI, ARM NEON and dotprod.
- Add GGUF loading (Q8_0, Q4_K, I2_S, embedded tokenizer) with an architecture router.
- Add DeepSeek-V2 mixture-of-experts routing and multi-head latent attention.
- Add KV cache strategies (int8, sliding window) chosen from available RAM.
- Add hidden reasoning, agentic tool execution, local RAG memory and a vision encoder bridge.
- Add an OpenAI-compatible HTTP API, the CLI and the REPL.
- Add conversion tools for HuggingFace BitNet and dense models, tokenizers and one-click model import.

### Fixed

- Fix a stack overflow in top-k sampling and a buffer overflow in the thought filter.
- Fix file-handle and descriptor leaks in model and tokenizer loading.
- Fix a NULL dereference in token decoding.
- Fix wrong value-cache indexing in attention.
- Fix the sampling RNG being re-seeded on every call.

[Unreleased]: https://github.com/shifulegend/project-zero/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shifulegend/project-zero/releases/tag/v0.1.0
