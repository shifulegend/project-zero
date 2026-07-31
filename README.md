# Project Zero — CPU LLM Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/language-C99-blue.svg)](src/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--512%20%7C%20AVX2%20%7C%20NEON-green)](src/math/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-OpenBenchmarking.org-orange)](https://openbenchmarking.org/result/2606207-SHIF-PROJECT42)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<p align="center">
  <img src="docs/tty_bitnet.png" width="720" alt="Project Zero: BitNet b1.58 running at 36.25 tok/s on Intel Xeon, no GPU">
</p>

<p align="center">
  <img src="docs/demo_bitnet.gif" width="720" alt="Project Zero: BitNet b1.58-2B-4T live demo on i5-11300H">
</p>

[Benchmarks](#benchmarks) · [Quick Start](#quick-start) · [UI/UX](#ui-ux) · [Help Wanted](#help-wanted) · [Docs](docs/)

---

Pure C, single binary. Runs Microsoft's BitNet b1.58 **up to 5.4× faster than Microsoft's own `bitnet.cpp`** — and dense GGUF models — no GPU, no Python, no ML framework.

- **Pure C, zero runtime deps** — `make release`, one executable, nothing else required
- **3.5–8.3× faster than bitnet.cpp on i5-11300H** (INT4, t=1..8) · **1.33–1.80× faster on 4-core Xeon** ([third-party verified on OpenBenchmarking.org ↓](#benchmarks))
- **35.79 tok/s on i5-11300H (INT4, 500 tokens, best-of-3)** · **36.25 tok/s on Xeon (PGO+LTO)** — 95% of DRAM bandwidth ceiling
- **One binary, two model families** — BitNet ternary and dense F16 GGUF, no per-model rebuild

---

<a id="benchmarks"></a>

## Benchmarks

### Intel i5-11300H @ 3.10 GHz · 16 GB DDR4 · AVX-512 VNNI (2026-06-21, fresh sweep)

**BitNet b1.58-2B-4T — Project Zero vs. Microsoft `bitnet.cpp`** — same model, same machine, same prompt, 500 tokens, sequential, **best of 3 runs per thread count**:

| Threads | PZ BF16 (tok/s) | PZ INT4 (tok/s) | MSFT bitnet.cpp | BF16 Gain | INT4 Gain |
|---|---|---|---|---|---|
| 1 | 11.64 | **16.98** | 2.04 | +471% | +732% |
| 2 | 20.23 | **27.99** | 3.76 | +438% | +644% |
| 3 | 22.31 | **33.82** | 5.17 | +332% | +554% |
| 4 | 23.42 | **35.79** | 6.64 | +253% | +439% |
| 5 | 21.62 | **31.80** | 5.15 | +320% | +517% |
| 6 | 22.28 | **33.68** | 6.01 | +271% | +460% |
| 7 | 21.50 | **32.50** | 6.61 | +225% | +392% |
| 8 | 21.28 | **32.03** | 6.04 | +252% | +430% |

**Peak: PZ INT4 = 35.79 tok/s (t=4) · PZ BF16 = 23.42 tok/s (t=4) · MSFT = 6.64 tok/s (t=4)**

> Project Zero INT4 is **3.9–8.3×** faster than bitnet.cpp across all thread counts. BF16 is **3.3–5.7×** faster. Prompt also faster: MSFT reports 7.00 tok/s prompt eval at t=4 vs ~110+ tok/s for the PZ tokenizer path.

**Peak-run terminal screenshots — same machine, same model, same prompt (500 tokens, best-of-3):**

| PZ INT4 · t=4 · **35.79 tok/s** | PZ BF16 · t=4 · **23.42 tok/s** | MSFT bitnet.cpp · t=4 · **6.64 tok/s** |
|---|---|---|
| ![PZ INT4 peak](benchmark_results/sweep_2026-06-21/screenshots/bitnet_pz_int4_t4.png) | ![PZ BF16 peak](benchmark_results/sweep_2026-06-21/screenshots/bitnet_pz_bf16_t4.png) | ![MSFT peak](benchmark_results/sweep_2026-06-21/screenshots/bitnet_msft_t4.png) |

All 24 screenshots (t=1..8 × 3 engines): [`benchmark_results/sweep_2026-06-21/screenshots/`](benchmark_results/sweep_2026-06-21/screenshots/)

<p align="center">
  <img src="docs/comparison_graph_i5.png" width="720" alt="Tok/s vs threads: PZ BF16, PZ INT4, Microsoft bitnet.cpp on i5-11300H">
</p>

<p align="center">
  <img src="docs/bar_chart_i5.png" width="640" alt="Peak throughput bar chart: PZ INT4 35.79, PZ BF16 23.42, MSFT 6.64 tok/s">
</p>

### Intel Xeon · AVX-512 VNNI (PGO+LTO, from earlier run)

| Threads | Project Zero | bitnet.cpp (i2_s) | Gain |
|---|---|---|---|
| 1 | **5.91 tok/s** | 4.96 | +19% |
| 2 | **12.78 tok/s** | 9.46 | +35% |
| 3 | **18.61 tok/s** | 13.59 | +37% |
| 4 | **21.45 tok/s** | 16.10 | +33% |

Optimized (PGO+LTO, INT4 classifier): **36.25 tok/s = 95% of the analytical DRAM bandwidth ceiling** on a 4-core Xeon.

<p align="center">
  <img src="docs/benchmark_bitnet.png" width="720" alt="BitNet b1.58-2B-4T: Project Zero beats Microsoft bitnet.cpp at every thread count (Xeon)">
</p>

> On dense models, Project Zero leads `llama.cpp` at 1–3 threads (+32%/+4%/+15%) and trails at peak 4-thread. On DeepSeek-V2 MoE it runs ~7× slower — this is the known open problem ([Help Wanted ↓](#help-wanted)).

### SmolLM2-135M F16 — Project Zero vs. llama.cpp (i5-11300H, 2026-06-21, fresh sweep)

Same model, same machine, same prompt, 500 tokens, sequential, **best of 3 runs per thread count**:

| Threads | Project Zero BF16 (tok/s) | llama.cpp (tok/s) | PZ Gain | llama.cpp Prompt tok/s |
|---|---|---|---|---|
| 1 | 58.00 | 53.50 | +8.4% | 319.7 |
| 2 | 86.03 | 84.20 | +2.2% | 588.7 |
| 3 | **100.44** | 106.20 | −5.4% | 700.9 |
| 4 | 98.28 | **106.20** | −3.7% | 1092.0 |
| 5 | 95.93 | 93.90 | +2.2% | 1063.6 |
| 6 | 89.63 | 94.90 | −5.6% | 1019.6 |
| 7 | 90.57 | 95.30 | −5.0% | 1020.4 |
| 8 | 83.32 | 86.30 | −3.5% | 958.1 |

**Peak: PZ = 100.44 tok/s (t=3) · llama.cpp = 106.20 tok/s (t=3)**

> PZ leads at t=1 (+8.4%) and t=2 (+2.2%), trails by 3–6% at peak. No fused Q4K matmul yet — see [Help Wanted ↓](#help-wanted). llama.cpp prompt eval is faster (700–1092 tok/s) because it batches the prompt; PZ does not yet report prompt eval speed separately.

**Peak-run screenshots — SmolLM2 (best-of-3):**

| PZ BF16 · t=3 · **100.44 tok/s** | llama.cpp · t=3 · **106.20 tok/s** |
|---|---|
| ![PZ SmolLM2 peak](benchmark_results/sweep_2026-06-21/screenshots/smollm2_pz_t3.png) | ![llama.cpp SmolLM2 peak](benchmark_results/sweep_2026-06-21/screenshots/smollm2_llama_t3.png) |

All 16 screenshots (t=1..8 × 2 engines): [`benchmark_results/sweep_2026-06-21/screenshots/`](benchmark_results/sweep_2026-06-21/screenshots/)

<p align="center">
  <img src="docs/benchmark_smollm2.png" width="720" alt="SmolLM2-135M F16: Project Zero vs llama.cpp t=1..8">
</p>

**Live terminal runs — Xeon (BitNet b1.58-2B-4T) and i5-11300H (SmolLM2-135M F16):**

| BitNet b1.58-2B-4T (ternary, Xeon) | SmolLM2-135M (F16 dense, i5) |
|---|---|
| ![BitNet live run](docs/tty_bitnet.png) | ![SmolLM2 live run](docs/tty_smollm2.png) |

**Xeon demo — 31-second live recording:**

<p align="center">
  <img src="docs/demo_xeon.gif" width="720" alt="Project Zero BitNet b1.58-2B-4T live demo on Xeon (36.25 tok/s)">
</p>

**Optimization journey — throughput across all phases:**

<p align="center">
  <img src="docs/performance_chart.png" width="720" alt="Project Zero throughput vs bitnet.cpp / llama.cpp across optimization steps">
</p>

*Per-configuration throughput from the [optimization journal](docs/PERFORMANCE_CEILING_REPORT.md).*

📊 **Third-party results on OpenBenchmarking.org** — not self-reported:

| Xeon vs. bitnet.cpp | i5-11300H vs. llama.cpp |
|---|---|
| [![Xeon result](docs/openbenchmarking_xeon_vs_bitnetcpp.png)](https://openbenchmarking.org/result/2606207-SHIF-PROJECT42) | [![i5 result](docs/openbenchmarking_i5_vs_llamacpp.png)](https://openbenchmarking.org/result/2606208-SHIF-PROJECT03) |

**Run it yourself and post your result:** [Discussion #3 — community benchmarks](https://github.com/shifulegend/project-zero/discussions/3)

---

<a id="quick-start"></a>

## Quick Start

**Option A — pre-built binary (Linux x86-64, no compiler needed):**

```bash
wget https://github.com/shifulegend/project-zero/releases/download/v0.1.0/adaptive_ai_engine-0.1.0-x86_64-linux.tar.gz
tar xf adaptive_ai_engine-0.1.0-x86_64-linux.tar.gz
./adaptive_ai_engine --model models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --prompt "The capital of France is"
```

**Option B — build from source (60 seconds):**

```bash
git clone https://github.com/shifulegend/project-zero.git
cd project-zero
make demo   # builds engine + downloads SmolLM2-135M + runs a test prompt
```

Expected output: `The capital of France is Paris.`

No GPU. No Python at runtime. No API key. GCC or Clang + `make` + `curl` — nothing else.

---

<a id="help-wanted"></a>

## Help Wanted

Two open problems where outside expertise would make a real difference:

| Problem | Current state | Target |
|---|---|---|
| **MoE expert weight repacking** | DeepSeek-V2-Lite runs at 1.90 tok/s — 7× behind `llama.cpp`. Top-K expert weights sit at non-contiguous GGUF offsets: **~86% L3 cache miss rate per token**. Fix: repack selected expert weights into contiguous memory at load time, matching llama.cpp's interleaved layout. | ≥ 9 tok/s |
| **Native Q4_K matmul kernel** | Current dense-model path dequants Q4_K → F32 before multiply. A fused mixed-precision kernel would close the remaining gap to `llama.cpp` on dense 4-bit GGUF models. | — |

Existing SIMD work documented in [`docs/KERNEL_INTERNALS.md`](docs/KERNEL_INTERNALS.md).
MoE repacking thread: [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1)

---

## What It Does

Runs [Microsoft's BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) ternary weights and **dense GGUF transformers** (SmolLM2, DeepSeek-V2-Lite) on commodity CPUs — from scratch, in C.

Also included in the same binary: OpenAI-compatible HTTP API (`--server --port 8080`), persistent RAG memory (`--memory-db`), SigLIP vision pipeline (`--vision`), and an agentic tool-use loop (`/agent`).

No GPU required. Python is offline tooling only (model conversion, testing).

> ⚠️ **Before contributing:** read [`GOLDEN_RULES.md`](GOLDEN_RULES.md). No hardcoding. Test after every change.

---

## Hardware

Memory bandwidth is the bottleneck — the engine reads 420–680 MB of weights per token. SIMD backend and thread count are auto-detected at startup.

| RAM config | BitNet tok/s | Notes |
|---|---|---|
| 4 GB | ~8–10 | disable earlyoom |
| 8 GB single-channel DDR4 | ~13 | bandwidth ceiling |
| 16 GB dual-channel DDR4 | ~16 | measured (+24% over single-ch) |

SIMD: AVX-512 VNNI → AVX2 → NEON → Scalar, selected at startup.

---

## Architecture

```
adaptive_ai_engine
├── src/math/       AVX-512 VBMI ternary kernel, VNNI INT8/INT4, AVX2/NEON fallbacks
├── src/core/       mmap weight loader (zero-copy), GGUF architecture-agnostic parser
├── src/sampling/   top-p / temperature (static 200K buffer, no malloc per token)
├── src/threading/  C11 atomic spinlock thread pool (no futex per dispatch)
└── src/transformer/ forward pass, attention, FFN, RoPE, KV cache (int8-quantized)
```

Key design choices: `mmap` + `POSIX_MADV_WILLNEED` for weight loading, runtime SIMD dispatch via function pointers, sliding-window int8 KV cache for 131k context, BF16 embeddings (660 MB smaller vs F32, no precision loss).

---

## Build

```bash
make release      # -O3 -march=native (default)
make debug        # ASan + UBSan
make test         # 3,367 assertions across all modules
make clean
```

Requirements: GCC or Clang, pthreads, libm. No other dependencies.

---

## CLI Reference

```bash
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --prompt  "Your prompt here" \
  --threads 4 \
  --max-tokens 256
```

Key flags: `--temperature`, `--top-p`, `--seed`, `--classifier {bf16|int8|int4|auto}`, `--server --port 8080`, `--memory-db path.vrdb`, `--image photo.jpg --vision vision.bin --proj projector.bin`

Full flag reference and REPL commands: run `./adaptive_ai_engine --help`

---

<a id="ui-ux"></a>

## UI/UX

> Full how-to (starting the server, every web UI control, REPL commands, CLI flags, API routes):
> [`docs/WEBUI_GUIDE.md`](docs/WEBUI_GUIDE.md).

**Web chat UI** — a browser-based chat interface embedded directly in the binary (no separate
install): streaming responses, adjustable sampling parameters, stop/cancel mid-generation, a
dark/light theme, and image upload (when the server is started with `--vision`/`--proj`).

```bash
./adaptive_ai_engine --model models/smollm2.gguf --server --port 8080
# open http://127.0.0.1:8080/ in a browser
```

| Light | Dark |
|---|---|
| ![Web UI — light theme](docs/design/screenshots/03-reply-light-2026-07-15T22-39-24-616Z.png) | ![Web UI — dark theme](docs/design/screenshots/05-dark-2026-07-15T22-39-24-616Z.png) |

**CLI/REPL polish** — colored output (`--color auto\|always\|never`, respects `NO_COLOR`), a
model-load progress indicator, a live tok/s status line during generation, and markdown/code
rendering in the interactive REPL:

![CLI REPL — color, markdown rendering, live tok/s](docs/design/screenshots/cli-repl.png)

**Startup banner** — an animated ASCII-art "PROJECT ZERO" splash (bottom-up slide-in reveal, a
hand-crafted 5-row block font, no external figlet dependency) that finishes with a brief
dim/bold shimmer, shown for the REPL and `--server` mode and suppressed for scripted one-shot
`--prompt` runs — TTY-gated, so no escape codes ever leak into piped/redirected output:

![CLI startup banner — animated reveal and shimmer](docs/demo_banner_shimmer.gif)

Static frame, for reference:

![CLI startup banner — animated ASCII-art "PROJECT ZERO"](docs/design/screenshots/06-cli-startup-banner-2026-07-16T01-03-52Z.png)

**Live "thinking" spinner** — a continuously animated braille spinner (bold cyan), advancing
once per streamed token next to the live tok/s status line, the same idea as Claude Code's
animated indicator while it's actively working:

![CLI live spinner during generation, plus the banner shimmer](docs/design/screenshots/07-cli-spinner-and-shimmer-2026-07-16T01-44-29Z.png)

**HTTP API hardening** — CORS (`--cors`/`--cors-origin`), optional API-key auth (`--api-key`,
off by default), Prometheus metrics (`--metrics` → `GET /metrics`), interactive docs
(`GET /docs`, `GET /openapi.json`), and a cancel endpoint (`POST /v1/chat/completions/cancel`)
that actually stops an in-flight generation, backed by a concurrency rearchitecture (per-
connection threads + a generation mutex) so static/metrics/docs requests are never blocked
behind a running chat completion.

Design decisions are checked against a written [design-principles reference](docs/design/ui-ux-principles.md)
before being accepted — see the [Phase 22.4 review](docs/design/review-2026-07-15.md) for the
full pass/fail breakdown (one real bug and one design-checklist violation were caught and fixed
during that review, not just cosmetic nits).

---

## Docs

| Document | What it covers |
|---|---|
| [KERNEL_INTERNALS.md](docs/KERNEL_INTERNALS.md) | AVX-512 VBMI kernel, MoE scatter problem, thread pool design |
| [PERFORMANCE_CEILING_REPORT.md](docs/PERFORMANCE_CEILING_REPORT.md) | Full optimization journal: 1.4 → 36.25 tok/s, bandwidth math |
| [DEBUGGING_JOURNAL.md](docs/DEBUGGING_JOURNAL.md) | Root-cause log of every major perf regression and fix |
| [ROADMAP.md](.github/ROADMAP.md) | Phase status (✅/🆘/❌), active blockers, planned phases |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Build, test, and contribution protocol |
| [DEVELOPER_ONBOARDING.md](DEVELOPER_ONBOARDING.md) | Testing mandate, QA protocol, branching strategy |

---

*Phase 34+ · BitNet b1.58-2B-4T · DeepSeek-V2-Lite-Chat (GGUF) · SmolLM2-135M F16 · SigLIP vision*
*Best: **35.79 tok/s** (BitNet INT4, i5-11300H, 500 tok, best-of-3) · **36.25 tok/s** (Xeon PGO+LTO) · **100.44 tok/s** (SmolLM2 F16) · **5.4× vs bitnet.cpp** (INT4 @ t=4) · 95% DRAM ceiling*
