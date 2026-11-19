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

- Log a connecting client as its network with the last octet masked (`192.168.1.x`) in `--verbose` mode, instead of its full address.
- Speed up the quantised LM heads: `--classifier int8` decodes BitNet 2B-4T 44% faster (18.0 to 25.9 tok/s) and `--classifier int4` 30% faster (20.4 to 26.5 tok/s), 4 threads, 2.10 GHz Xeon, the default head unchanged at about 21.5; both now beat the default, and the portable release binary uses the same AVX-512 kernels as a native build.
- Redo the first-run hardware calibration: SIMD tiers are compared in interleaved rounds and one replaces the default only when it wins a sign test and clears a 5% margin, and the thread count stays at the physical-core count unless another count is clearly faster. Existing calibration caches are discarded and rebuilt once on the next run (about a minute).
- Label the web UI's per-reply usage line in tokens ("7 reply tokens · 37 prompt tokens" instead of "7 tokens · 37 prompt"); hovering it explains that the prompt count includes the chat template and any system message.

### Added

- Publish a CycloneDX software bill of materials (`sbom.cdx.json`) with each release and attach a signed build-provenance attestation to the archive, `install.sh` and the SBOM; check one with `gh attestation verify <file> --repo shifulegend/project-zero`.
- Add `--rate-limit <n>` (default 20 requests per second, 0 = off; also `pz config set rate-limit`): the API's cheap non-generation routes (`/v1/models`, `/v1/config`, `/v1/update*`, `/v1/prompt_cache*`, `/v1/selfcheck`, `/v1/watermark/detect`) answer `429` with a `Retry-After` header once the limit is exceeded. Chat generation, `/health`, `/metrics` and the web UI files are not limited.
- Add `--client-timeout <s>` (default 30; 0 = never) and `--max-connections <N>` (default 64; 0 = unlimited), also available as `pz config set client-timeout` / `max-connections` and reported by `GET /v1/config`-style server info: the server drops a client that sends or reads nothing for the timeout, and answers `503` with `Retry-After` when more connections than the cap are open. Before, one stalled client held a server thread forever and a flood could create unbounded threads.
- Add `--act-quant auto|int8|float` (also `PZ_ACT_QUANT` and `pz config set act-quant`): choose the activation numerics of ternary (BitNet) matmuls independently of `--simd` (`auto` takes the numerics of the calibrated tier, prints them at startup with the measured int8 and float tier speeds and, on a ternary model, checks int8 against the accuracy bar once per model and setting, caching the verdict in `~/.project-zero/selfcheck.txt`, falling back to float activations if it fails); `float` skips the int8 tiers (VNNI, AVX-VNNI, dotprod) and runs float32 activations.
- Add an accuracy probe: `pz --self-check` scores a built-in passage with the configured numerics and LM head against float activations and a BF16 head and exits 1 when perplexity rises by more than `--quality-tolerance` percent (default 1.0, `pz config set quality-tolerance`); `--self-check-tokens` sets its length. An explicit `--classifier int8`, `int4` or `auto-fast` that fails it falls back to a BF16 head at startup with a message (`--quality-tolerance 0` keeps the request as given; BitNet 2B-4T with `--classifier int4` reads about +9% on the probe, so it needs `--quality-tolerance 10`): Mamba-130M with `--classifier int4` answers correctly again instead of producing gibberish.
- Add `POST /v1/selfcheck` and a Settings → Server defaults panel in the web UI that shows the active numerics, runs the probe and saves `act_quant` and `quality_tolerance` for the next start; `GET /v1/config` reports them under `runtime`.
- Add `--no-calibrate` (also `PZ_NO_CALIBRATE=true` or `pz config set no-calibrate true`): never load, run or save the hardware calibration, so a run uses the highest SIMD tier and the detected thread count.
- Warn on stderr when a chat template renders a prompt that ends with the end-of-sequence token, and when a chat generation ends with 0 tokens; the second warning names the first token the model chose and the last prompt token.

### Fixed

- Answer `400` to a chat request whose `response_format.type` is `json_schema` or anything other than `text` or `json_object`; before, the server returned `200` with free text, so a client relying on schema-conformant output was never told it was not getting it.

- Make nucleus (`top_p`) sampling safe when several requests sample at the same time, and apply it to vocabularies larger than 200,000 tokens (those silently fell back to greedy decoding before).
- Show fenced code blocks in the web UI exactly as written: `**text**` and `` `text` `` inside a ``` block were turned into bold and inline code. Long replies also render faster, because line breaks are now handled in one pass instead of rescanning the whole text for every newline.
- Decode half-precision (fp16) values the same way everywhere: the Q8_0, Q5_0, Q5_1, Q5_K, Q2_0, F16 and LoRA paths and the model loader turned a very small (subnormal) fp16 value into roughly zero, while the Q4_K, Q2_K, Q6_K paths decoded it correctly. All paths now share one exact converter; values that are not subnormal decode exactly as before.
- Make the `--memory-db` vector memory crash-safe and tamper-evident: records and the header now carry CRC-32 checksums (a corrupted file is refused instead of silently returning wrong memories), each store is synced to disk in an order that survives a crash (a half-written record is ignored and overwritten, never misaligns later entries), a second `pz` process opening the same file is refused instead of interleaving writes, an entry count beyond the limit is rejected, and failed opens no longer leak memory or file handles. **Format change:** new files use format v2, which older `pz` versions cannot read. An existing v1 file is still read; the first save into it copies it to `<file>.v1.bak` (never overwriting an earlier backup) and rewrites it as v2. The file is now created with owner-only permissions.
- Make the HTTP server read request sizes strictly: a malformed, negative, overflowing or conflicting `Content-Length` is now answered with `400` (too large with `413`), and a chunked request body is answered with `501` instead of being read as empty. Before, `Content-Length: abc` or a huge number was silently misread.
- Fix HTTP responses (including web UI files and error replies) being cut short when the connection could not take all the bytes in one write; the server now keeps writing until everything is sent or the client has gone.
- Ship `NOTICE` and `LICENSE-LGPL-3.0.txt` in release archives and install them next to `LICENSE` (`share/doc/project-zero`), because the binary statically links an LGPL-3.0-or-later AVX-512 kernel; the README license section now names that exception.
- Fix DeepSeek-V2 (`tokenizer.ggml.pre = deepseek-llm`) and DeepSeek-Coder models being tokenised with the generic splitter, which cut a space off the punctuation after it and fed the model unfamiliar token sequences: DeepSeek-V2-Lite perplexity on a 7.5 KB test text drops from 11.51 to 6.67 (llama.cpp: 6.61), and token IDs now match llama.cpp's.
- Fix the int8 activation quantiser rounding toward zero instead of to nearest: ternary (BitNet) and Q2_0 models on the VNNI, VNNI-256 and ARM dotprod paths lost about 10% perplexity (BitNet 2B-4T on a 7.5 KB test text: 34.39 before, 31.37 after, against 31.32 for float activations; decode speed unchanged at 20.6 vs 19.7 tok/s, 4 threads, within noise). Output stayed fluent, so the loss was easy to miss.
- Fix the official BitNet b1.58 2B-4T GGUF answering nothing (exit 0, empty `content` over the API): its chat template ended the prompt with the end-of-sequence token. `pz` now recognises that template from the model's metadata and uses a built-in corrected one (`/v1/config` `chat_template.source` is `builtin`); `--chat-template-file` still overrides it, so BitNet no longer needs the flag.
- Fix the portable binary (the release download and `install.sh`) decoding BitNet b1.58 2B-4T about 35% slower than a native build on AVX-512 VNNI CPUs: its AVX-512 activation quantiser was a scalar fallback, the pre-quantised VNNI path was missing, and the VNNI kernel never used its faster VBMI bit-unpack. On a 4-vCPU Xeon (AVX-512 VNNI + VBMI) it now decodes at 23.5 tok/s against 25.6 for a native build (16.3 before). Other quantised and F16 paths of the portable binary are still slower than native (SmolLM2-135M F16 about 15%) and are being moved one family at a time.
- Fix `--simd scalar|avx2|avx512f` and the calibration sweep not changing the ternary matmul on builds with AVX-512 VNNI (`make release`, source builds): the matmul always ran the VNNI kernel, so calibration picked a tier from noise and the web UI could report AVX2 while VNNI ran. A forced tier now runs the kernel it names (BitNet 2B decodes at about 7.5 tok/s with `--simd avx2` against about 24.6 by default on a 4-vCPU Xeon).
- Preserve all text parts in chat messages that send content as an array.
- Release the earlier text when a chat message repeats its `content` key (the last key wins); each such request used to leak that text on the server.
- Stop answering `429 "Another generation is already in flight"` to a client that sends its next `/v1/chat/completions` request as soon as it has read the previous answer (most often a streaming client that stops at `data: [DONE]`); the server now frees the generation slot before it writes the end of the response.
- Align the right border of the startup "Hardware Profile" box; rows were padded to different widths.

### Security

- Only show the web UI's release-notes and changelog links when the server gives an `https://` URL; a `javascript:`, `data:` or `http:` value is no longer put into a link.
- Refuse requests whose `Host` header is not `localhost`, `127.0.0.1` or `[::1]` with `403` (except `/health`), which stops DNS-rebinding attacks where a web page's hostname is re-pointed at your local server. Behind a reverse proxy, add the proxy's address with `--cors --cors-origin`.
- Run the agent's shell commands with a minimal environment: only a fixed `PATH`, `TZ`, `LANG` and `LC_*` are passed on, so API keys, tokens and other `PZ_*` / cloud credentials in the server's environment can no longer be read by the commands the model runs.
- Make the agent's file-reading commands (`cat`, `ls`) refuse a path argument when the working directory can no longer be determined (for example because it was deleted), instead of allowing it unchecked.
- Save the settings file (`pz config`, the web UI's save-defaults) safely: it is created private (mode 0600) from the first byte, a pre-planted file or symlink at the temporary name is not followed, and the data is synced to disk before the rename so a crash or power loss leaves the old or the new file, never an empty one. `.env*`, `*.key`, `*.pem` and `config.toml` are now git-ignored, and `config.toml.example` shows the available settings.
- Make `install.sh` download over HTTPS only (curl `--proto =https`, wget `--https-only`, including redirects) with a connection timeout and a 10-minute cap, so a plain-http mirror or a stalled server can no longer feed or hang the installer. A local `file://` mirror still works.
- Reject chat requests whose JSON nests objects or arrays more than 256 levels deep with `400`, instead of scanning an arbitrarily deep body.
- Refuse cross-site POSTs on every API route, not only `/v1/config/defaults`: a POST carrying a browser `Origin` that is neither this server nor an allowed `--cors-origin` now gets `403`. Requests without an `Origin` (curl, SDKs) and the web UI are unaffected; behind a proxy that rewrites `Host`, add the public origin with `--cors --cors-origin`. Before, any web page you visited could trigger chat generations, `/v1/update/check`, cache clears or cancel another request on `127.0.0.1`.

## [0.2.0] - 2026-10-22

### Changed

- Log a connecting client as its network with the last octet masked (`192.168.1.x`) in `--verbose` mode, instead of its full address.
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

- Deprecate the `adaptive_ai_engine` name: `make` and the release archive still provide it as a symlink to `pz` that prints a notice; replaced by `pz` and removed in v0.3.0.

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
- Add parallel serving (`--parallel N`, `--parallel-batch N`): up to N chat requests generate at once, batched into one forward pass per token for dense attention models, with per-slot KV memory taken from a shared budget; `/v1/config`, `/metrics` and Settings → Server defaults show and save it. With 4 concurrent clients aggregate throughput rises from 45 to 101 tok/s (SmolLM2-135M Q8_0) and from 12.7 to 29.8 tok/s (SmolLM2-1.7B Q4_K_M) on a 4-vCPU Xeon at 4 threads.
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
- Add a guided `pz config` setup for non-technical users: each step explains in plain words what it means, shows example choices and their effect, and says how to change it later; it can download a checksum-verified starter model sized to the computer's free memory (list replaceable with `PZ_MODEL_CATALOG`) and open the chat page when done. The installer hands over to it.
- Add faster kernels: K-6 5-trit LUT ternary (AVX-512BW), plain NEON ternary, AVX-512 VNNI Q2_0, llama.cpp-compatible Q4_K x8 repack, fused zero-copy Q6_K and batched speculative-verify kernels.

### Fixed

- Fix chat replies running past the end of the answer when a model ends its turn with a control token its metadata does not declare (e.g. Falcon3-1.58bit's `>>UNUSED_1<<`): in chat mode any control token the chat template never writes now ends the turn, so no `--stop` workaround is needed.
- Fix BitNet I2_S GGUF files (the official BitNet b1.58 2B-4T GGUF, Falcon3-1.58bit) producing incoherent text: weights are now read in bitnet.cpp's interleaved 128-weight layout with the per-tensor scale stored after the packed data.
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
- Fix a streaming API reply continuing to generate after its client disconnected, holding the generation slot until `max_tokens`.
- Fix stopping the server (Ctrl+C/SIGTERM) while a request was generating: the generation now stops before the model is unloaded instead of running on freed memory.
- Fix `pz config set best-of-max` accepting values up to 1024 that `--best-of-max` then rejects at startup (the limit is 64 in both).

### Security

- Fix prompt injection through chat messages: control tokens such as `<|im_start|>` or `<|eot_id|>` typed or pasted into a message (or returned by a tool) are now read as plain text, so they can no longer end a turn or start a forged system turn; prompts containing unknown `<|...|>` text are no longer rejected.
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

[Unreleased]: https://github.com/shifulegend/project-zero/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shifulegend/project-zero/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shifulegend/project-zero/releases/tag/v0.1.0
