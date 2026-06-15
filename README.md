# Project Zero — BitNet Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/language-C99-blue.svg)](src/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](README.md)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--512%20%7C%20AVX2%20%7C%20NEON-green)](src/math/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-OpenBenchmarking.org-orange)](https://openbenchmarking.org/result/2606063-SHIF-PROJECT91)
[![Discussions](https://img.shields.io/github/discussions/shifulegend/project-zero)](https://github.com/shifulegend/project-zero/discussions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> ⚠️ **Before contributing: read [`GOLDEN_RULES.md`](GOLDEN_RULES.md).** No hardcoding. Test after every change. No exceptions.

A from-scratch, single-binary LLM inference engine written in C, built to run
Microsoft's [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
at maximum speed on commodity CPUs — no GPU, no Python, no framework.

---

## Quick Start

```bash
# Build (release mode, -O3 -march=native)
make release

# Run inference
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --prompt  "The capital of France is" \
  --max-tokens 64

# REPL (interactive mode — omit --prompt)
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
```

### All CLI flags

#### Required

| Flag | Description |
|------|-------------|
| `--model <path>` | Path to `.bin` model file |
| `--tokenizer <path>` | Path to `.bin` tokenizer file |

#### Generation

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt <string>` | — | Single-shot prompt; omit to enter REPL |
| `--max-tokens <n>` | 512 | Maximum tokens to generate |
| `--temperature <f>` | 0.7 | Sampling temperature (0.0 = greedy / deterministic) |
| `--top-p <f>` | 0.9 | Nucleus sampling cutoff |
| `--seed <n>` | random | RNG seed — set for reproducible output |
| `--reasoning` | off | Enable `<think>` hidden reasoning mode |

#### System

| Flag | Default | Description |
|------|---------|-------------|
| `--threads <n>` | auto | Worker threads (auto = physical core count, avoids HT throttle) |
| `--classifier <fmt>` | auto | Classifier quantization: `auto`, `bf16`, `int8`, `int4` |
| `--verbose` | off | Print timing, token counts, and RAG injection diagnostics |

#### Multimodal (Phase 11 + 34 — vision pipeline implemented)

| Flag | Default | Description |
|------|---------|-------------|
| `--image <path>` | — | Path to image file for vision queries |
| `--vision <path>` | — | Path to `vision.bin` SigLIP encoder weights (Phase 34) |
| `--proj <path>` | — | Path to `projector.bin` MLP weights (Phase 34) |

Extract vision weights first (one-time):
```bash
python tools/extract_multimodal.py --repo moondream-hf/moondream2 --out models/
# Then: --vision models/vision.bin --proj models/projector.bin --image photo.jpg
```
> Full image recognition requires a multimodal-trained LLM backbone (BitNet is text-only).
> The vision pipeline (SigLIP encoder → projector → KV injection) is fully functional.

#### RAG Memory (Phase 15)

| Flag | Default | Description |
|------|---------|-------------|
| `--memory-db <path>` | — | Path to `.vrdb` persistent memory file (created if absent) |

---

## Performance

### Xeon Cloud Server (Best Result)

Benchmarked on **Intel Xeon @ 2.10 GHz** (Emerald Rapids, 4 cores, no HT, 260 MiB L3):

| Configuration | tok/s | Notes |
|---|---|---|
| Xeon baseline (AVX-512F float FMA) | 16.47 | Ternary float path |
| + INT8 VNNI classifier | 21.20 | +28.7% (dpbusds LM head) |
| + VBMI 3-instruction unpack | 32.65 | +54% (ternary layers 2.7× faster) |
| + INT4 classifier + PGO/LTO | **36.25** | **95% of DRAM ceiling** |

### vs bitnet.cpp (Same Hardware)

| Engine | avg tok/s | Best tok/s | Notes |
|---|---|---|---|
| **Project Zero** | **34.75** | **36.25** | PGO+LTO, INT8 classifier, VNNI |
| bitnet.cpp | 19.33 | 19.83 | I2_S format, clang, 4T |
| **Advantage** | **1.80×** | **1.83×** | Same model, same Xeon |

📊 **OpenBenchmarking results:** [Xeon — Project Zero vs bitnet.cpp](https://openbenchmarking.org/result/2606063-SHIF-PROJECT91)

### Developer Laptop (i5-11300H)

Benchmarked on **Intel i5-11300H** (Tiger Lake, 4 cores / 8 threads, DDR4 dual-channel):

| Configuration | tok/s | Notes |
|---|---|---|
| Baseline (debug build, AVX2, 8 threads, earlyoom on) | 1.4 | Starting point |
| Release + CPU governor = performance | 2.0 | |
| AVX-512 kernels + earlyoom disabled | 10.5 | Single largest gain (+91%) |
| Top-p rewrite + HT fix | **13.0** | Single-channel DDR4 ceiling |
| + dual-channel RAM upgrade | **~16.1** | Measured post-upgrade |

**Memory bandwidth is the bottleneck** — the engine reads 420–680 MB of weights per
token (depending on L3 cache effectiveness). The engine auto-adapts to any hardware:
SIMD backend, thread count, KV cache strategy, and classifier quantization are all
detected at startup.

📊 **OpenBenchmarking results:** [i5-11300H — Project Zero vs llama.cpp/DeepSeek](https://openbenchmarking.org/result/2606062-SHIF-PROJECT21)

> Use `--classifier bf16` to preserve full LM head precision at the cost of speed.
> Use `--classifier int8` or `int4` to trade precision for throughput (default: auto).

---

## Help Wanted

One open problem where outside input would help:

| Problem | Current state | Target | Discussion |
|---|---|---|---|
| **MoE expert weight repacking** | DeepSeek MoE runs at 1.90 tok/s (7× behind llama.cpp). Fused Q4_K matmul kernels are in place — the remaining gap is expert weight scatter: top-K expert weights sit at non-contiguous GGUF offsets, causing ~86% L3 miss rate per token. Fix: repack selected experts into contiguous memory at load time, matching llama.cpp's interleaved layout. | ≥ 9 tok/s | [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1) |

**Community benchmark challenge** — run the engine on your hardware and add your result to the comparison table: [Discussion #3](https://github.com/shifulegend/project-zero/discussions/3)

For a deep-dive into how the existing fast path works: [`docs/KERNEL_INTERNALS.md`](docs/KERNEL_INTERNALS.md)

---

## DeepSeek-V2-Lite-Chat Support (GGUF)

The engine also runs DeepSeek-V2-Lite-Chat in Q4_K_S quantization directly from GGUF files.
This is a 16B-parameter Mixture-of-Experts model with Multi-head Latent Attention (MLA).

### Quick Start

```bash
# Run DeepSeek-V2-Lite-Chat (GGUF, Q4_K_S)
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 30
# Output: " The capital of France is Paris.<｜end▁of▁sentence｜>"
```

Re-validated on `2026-03-22` without `--tokenizer`: the engine auto-loaded the GGUF tokenizer
metadata and produced `The capital of France is Paris.` on the first no-tokenizer validation run.
Note: CLI help text in `src/cli/args.c` still says `--tokenizer` is required, but GGUF runtime
loading in `src/cli/main.c` intentionally supports omission.

### Model Architecture (DeepSeek-V2-Lite)
- **Parameters:** 16B total, ~2.4B active per token (MoE top-6 of 64 experts)
- **Layers:** 27 (1 dense + 26 MoE)
- **Attention:** Multi-head Latent Attention (MLA) with YaRN RoPE
- **KV compression:** LORA=512 latent vectors (vs standard 2048 full KV)
- **Quantization:** Q4_K_S (4-bit keys, mixed precision scales)
- **GGUF metadata:** all config read dynamically (dim, n_heads, eps, rope params, MoE topology)

### MLA Pipeline (Steps 1–10, verified against llama.cpp)
| Step | Operation | Key tensors |
|------|-----------|-------------|
| 1 | Tokenization | chat template → 14 tokens for Paris prompt |
| 2 | BOS injection | token 100000 from `tokenizer.ggml.bos_token_id` |
| 3 | Token embedding | `token_embd.weight` Q4_K, F32 pre-dequant path |
| 4 | Pre-attn RMSNorm | `blk.N.attn_norm.weight` F32, eps=1e-6 from GGUF |
| 5 | Q projection | `blk.N.attn_q.weight` Q4_K [3072×2048] |
| 6 | KV-A compression | `blk.N.attn_kv_a_mqa.weight` Q4_K [576×2048] |
| 7 | KV-A latent norm | `blk.N.attn_kv_a_norm.weight` F32 [512] |
| 8 | KV-B expansion | `blk.N.attn_kv_b.weight` Q4_K [4096×512] |
| 9 | YaRN RoPE | freq_scale=1/40, attn_factor=0.7931, corr_dims=[10,23] |
| 10 | KV cache write | memcpy k_nope, v, k_rope into sliding-window cache |

All 10 steps verified to match llama.cpp output within float32 accumulation noise.
See `DEBUGGING_JOURNAL.md` for the full root cause analysis of the Q4_K nibble bug.

### Performance (best: 2026-03-23, fused kernel: 2026-06-15)
- Current: **1.90 tok/s** (T=4, INT8 classifier, AVX2 — fused Q4K kernels active for MoE dispatch)
- Ceiling: **9.8 tok/s** (analytical DRAM BW ceiling @ 11.7 GB/s, ~1.19 GB/token)
- Realistic ceiling: **2–4 tok/s** today (expert scatter penalty — top-K experts at non-contiguous GGUF offsets)
- llama.cpp reference: 13.79 tok/s (T=4)
- Gap to llama.cpp: ~7× — primary bottleneck: expert weight scatter access pattern (see Help Wanted)
- Next step: repack top-K expert weights into contiguous memory at load time to eliminate scatter penalty

---

## Architecture

```
adaptive_ai_engine
├── src/
│   ├── agent/        tool_interceptor.c, cmd_exec.c, output_inject.c,
│   │                 agent_loop.c, user_approval.c
│   ├── cli/          main.c, args.c, timer.c, repl.c
│   ├── core/         config.c, weights.c, run_state.c
│   ├── math/         parallel_matmul.c, simd_dispatch.c,
│   │                 ternary_matmul_packed_avx512.c, rope.c, …
│   ├── memory/       mapped_file.c (mmap / MapViewOfFile)
│   ├── sampling/     top_p.c, temperature.c, top_k.c, rng.c
│   ├── threading/    thread_pool.c (atomic spinlock), cpu_probe.c
│   ├── tokenizer/    tokenizer_load.c, encode.c, decode.c
│   └── transformer/  forward.c, attention.c, ffn.c, embedding.c,
│                     generate.c, rope.c
└── include/          (mirrors src/ structure)
```

### Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Weight loading | `mmap` + `POSIX_MADV_WILLNEED` | Zero-copy; OS manages page-in |
| SIMD dispatch | Runtime function pointers | AVX-512 → AVX2 → NEON → Scalar |
| Thread pool | C11 atomics + spin-then-sleep | Eliminates futex syscalls per dispatch |
| Ternary matmul | 16-wide AVX-512 packed kernel | 2× throughput vs. AVX2 |
| Embeddings | BF16 (upper 16 bits of float32) | 660 MB smaller; zero precision loss |
| KV cache | int8-quantized, sliding window | Fits 131k context in reasonable RAM |
| Sampling | Static 200K buffer, pre-filter | No malloc per token; 90% fewer exp() calls |

---

## Model: Microsoft BitNet b1.58-2B-4T

| Parameter | Value |
|---|---|
| Architecture | Llama-3-style with BitLinear layers |
| Parameters | 2 billion |
| Weights | Ternary (−1, 0, +1), packed 4/byte |
| Embeddings | BF16 (weight-tied with LM head) |
| dim | 2560 |
| hidden\_dim | 6912 |
| n\_layers | 30 |
| n\_heads | 20 |
| n\_kv\_heads | 5 (GQA, 4:1 ratio) |
| head\_dim | 128 |
| vocab\_size | 128,256 (Llama-3 tokenizer) |
| seq\_len | 4096 |
| Activation | ReLU² |
| RoPE θ | 500,000 |
| Binary size | 1.18 GB (v4, BF16 embeddings) |

---

## Binary Model Format

```
Offset   Size   Content
0        4      Magic: 0x594E5254 ("TNRY")
4        4      Version: 1
8        40     Config struct (dim, n_layers, heads, vocab_size, …)
48       16     Padding (64-byte alignment)
64       …      Weight data (64-byte aligned sections)
```

Weight order per layer:
1. `rms_att_weight` (float32, dim)
2. `rms_ffn_weight` (float32, dim)
3. `wq` + `sq` — packed ternary + float32 scale
4. `wk` + `sk`
5. `wv` + `sv`
6. `rms_attn_sub_norm` (BitNet-specific, float32, dim)
7. `wo` + `so`
8. `w1`/`w3`/`w2` (FFN gate/up/down) + scales
9. `rms_ffn_sub_norm` (BitNet-specific, float32, hidden\_dim)

After layers: `rms_final_weight` + BF16 embedding table (weight-tied LM head).

---

## Build

```bash
# Release (default)
make release      # -O3 -march=native

# Debug (ASan + UBSan)
make debug

# Run all tests
make test

# Packed-weight unit tests only
make test-packed

# Clean
make clean
```

**Requirements:** GCC or Clang, pthreads, libm. No external dependencies.

---

## Hardware Requirements & Scaling

| RAM | Performance | Notes |
|---|---|---|
| 4 GB | ~8–10 tok/s | earlyoom must be disabled |
| 8 GB single-channel DDR4 | ~13 tok/s | Memory-bandwidth ceiling |
| 16 GB dual-channel DDR4-2667 | ~16 tok/s | Measured (+24% over single-channel) |
| 16 GB dual-channel DDR4-3200+ | ~17–18 tok/s | Projected |

The engine auto-detects available RAM and sets KV-cache size accordingly.
Thread count is auto-detected as physical cores (HT disabled for AVX-512 workloads).

---

## SIMD Support

The engine selects the best available backend at compile time with a runtime
dispatch table (`src/math/simd_dispatch.c`):

| Backend | Vector width | Supported on |
|---|---|---|
| AVX-512 | 16 floats | Intel Tiger Lake, Ice Lake, Skylake-X+ |
| AVX2 | 8 floats | Intel Haswell+ / AMD Zen+ |
| NEON | 4 floats | ARM Cortex-A / Apple Silicon |
| Scalar | 1 float | Any CPU |

---

## Repository Structure

```
project-zero/
├── adaptive_ai_engine        Compiled binary
├── src/                      C source files
├── include/                  Header files
├── tests/                    Unit tests
├── tools/                    Python conversion scripts
│   ├── convert_hf_bitnet.py  HuggingFace → binary converter
│   └── convert_tokenizer.py  Tokenizer converter
├── models/                   Model binaries (Git LFS)
│   ├── bitnet-b1.58-2B-4T.bin               1.18 GB (use this)
│   ├── bitnet-b1.58-2B-4T_tokenizer_proper.bin
│   └── microsoft-bitnet-b1.58-2B-4T/        Original HF files
└── docs/
    ├── CHANGELOG.md
    ├── PERFORMANCE_CEILING_REPORT.md
    └── PHASE10_PRE_AUDIT_REPORT.md
```

---

## Agentic Mode (Phase 14)

The engine supports an **agentic loop** where the model can emit XML-style tool tags
during generation. The engine intercepts these tags, safely executes the requested
action, and injects the result back into the model's context.

### Quick start

```bash
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --verbose
```

At the `>` prompt use `/agent` followed by your prompt:

```
/agent You are a focused assistant. For your next single output, print EXACTLY this and nothing else: <exec>echo AGENT-TEST</exec>
```

When asked `Allow? [y/N]` type **y**. Expected output:

```
[AGENT] <exec> detected: echo AGENT-TEST
[AGENT] About to execute: echo AGENT-TEST
Allow? [y/N]: y
[AGENT] Command exit=0 timed_out=0
[AGENT] Output:
AGENT-TEST
```

### Supported tags

| Tag | Action |
|-----|--------|
| `<exec>cmd</exec>` | Execute allow-listed command, inject stdout into KV |
| `<think>text</think>` | Hidden reasoning (logged, not shown to user) |
| `<save_memory>text</save_memory>` | Embed text + save to vector DB (Phase 15 — fully implemented) |
| `<search_memory>query</search_memory>` | Search vector DB + inject top results into context (Phase 15 — fully implemented) |

### Allow-listed commands

`echo`, `ls`, `cat`, `pwd`, `uname`, `date`, `id`

Any other command is blocked before execution (no approval prompt is shown).

### Non-interactive / automated testing

Set `PROJECT_ZERO_AGENT_AUTO_APPROVE=1` (requires PTY — see
[WALKTHROUGH_PHASE14.md](WALKTHROUGH_PHASE14.md) for a complete Python PTY runner).

### More test prompts

```
/agent Run <exec>date</exec> and tell me what day it is.
/agent Run <exec>uname -a</exec> and describe the OS.
/agent Run <exec>pwd</exec> and tell me where we are.
/agent Run <exec>id</exec> and tell me the current user.
/agent Run <exec>ls models</exec> and list available model files.
```

> Full documentation: [WALKTHROUGH_PHASE14.md](WALKTHROUGH_PHASE14.md)

---

## RAG Persistent Memory (Phase 15)

The engine supports **persistent memory** across sessions. On every prompt the engine
vector-searches the memory DB, finds the most relevant stored snippets, and silently
prepends them to the model context before generation begins.

### Quick start

```bash
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --memory-db /tmp/my_memory.vrdb \
  --verbose
```

The `.vrdb` file is created automatically if it does not exist.

### REPL memory commands

| Command | What it does |
|---|---|
| `/memory save <text>` | Embed `<text>` and store to disk (with deduplication) |
| `/memory list` | List all stored memory entries |
| `/memory search <query>` | Vector-search without generating — shows top-5 matches with similarity scores |

### How auto-retrieval works

Every time you type a plain prompt at the `>` prompt, `auto_retrieve_and_inject()`
runs before generation:

```
Your prompt → embed → cosine-search DB → top-3 matches → inject into KV cache → generate
```

With `--verbose` you will see: `[Memory] Injected N tokens of relevant context.`

### Cross-session persistence

Memories survive process restart. Re-launch with the same `--memory-db` path and
the banner will show `[RAG] Memory enabled — N entries in '/path/to/file.vrdb'`.

### Saving via agent tags

In `/agent` mode the model can save and search memory autonomously:

```
/agent Remember that my favourite language is Rust
```

If the model emits `<save_memory>favourite language is Rust</save_memory>` in its output,
the agent loop intercepts it and calls `auto_save_memory()` automatically.
Similarly `<search_memory>language</search_memory>` triggers an in-context search.

### Deduplication

Entries with cosine similarity ≥ 0.95 to an existing entry are silently skipped.
You will see `[Memory] Duplicate detected — entry not saved.`

> Full documentation: [docs/PHASE15_RAG.md](docs/PHASE15_RAG.md)

---

## REPL Slash Commands (all phases)

All commands available at the interactive `>` prompt. Any input **without** a `/` prefix is treated as a plain prompt and sent to the model.

### Navigation

| Command | Phase | What it does |
|---|---|---|
| `/help` | 12 | Print all available commands |
| `/quit` | 12 | Exit the REPL |
| `/exit` | 12 | Alias for `/quit` |
| `/context` | 12 | Show KV cache usage: `Used: N / MAX tokens` |

### Reasoning

| Command | Phase | What it does |
|---|---|---|
| `/think` | 12 | Toggle extended reasoning mode on/off. When enabled, the model may emit `<think>…</think>` blocks; these are logged but not shown to the user. Can also be enabled at launch with `--reasoning`. |

### Agentic Tools

| Command | Phase | What it does |
|---|---|---|
| `/agent <prompt>` | 14 | Run the agentic generation loop. The model may emit tool tags in its output; the engine intercepts and executes them. |

**Tool tags intercepted inside `/agent` mode:**

| Tag | Phase | Effect |
|---|---|---|
| `<exec>cmd</exec>` | 14 | Execute `cmd` if on the allow-list; inject stdout back into context. Requires user confirmation unless `PROJECT_ZERO_AGENT_AUTO_APPROVE=1`. |
| `<think>text</think>` | 14 | Hidden reasoning step — logged, not shown to user. |
| `<save_memory>text</save_memory>` | 15 | Embed `text` and store to the vector DB (deduplication: cosine ≥ 0.95 skips). Requires `--memory-db`. |
| `<search_memory>query</search_memory>` | 15 | Search DB for `query`; inject top-3 results into context. Requires `--memory-db`. |

**Allow-listed exec commands:** `echo`, `ls`, `cat`, `pwd`, `uname`, `date`, `id`  
All other commands are refused before execution.

### Memory (RAG)

All `/memory` commands require `--memory-db <path>` at launch.

| Command | Phase | What it does |
|---|---|---|
| `/memory list` | 15 | List all stored entries (index + text). |
| `/memory search <query>` | 15 | Vector-search the DB for `<query>`; print top-5 matches with cosine similarity scores. Does not generate. |
| `/memory save <text>` | 15 | Embed `<text>` and store to DB. Duplicate-safe (cosine ≥ 0.95 = skip). |

**Auto-retrieval:** Every plain prompt at `>` automatically triggers a vector search and injects the top-3 matching memories before generation. With `--verbose` you will see `[Memory] Injected N tokens of relevant context.`

---

## Converting a Model

```bash
# Activate venv (has safetensors, numpy)
source .venv/bin/activate

# Convert HuggingFace model to binary format
python3 tools/convert_hf_bitnet.py \
  --model models/microsoft-bitnet-b1.58-2B-4T/ \
  --output models/bitnet-b1.58-2B-4T.bin

# Convert tokenizer
python3 tools/convert_tokenizer.py \
  --input  models/microsoft-bitnet-b1.58-2B-4T/tokenizer.json \
  --output models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
```

---

## Key Documents

| Document | What it covers |
|---|---|
| [README.md](README.md) | This file — all CLI flags, REPL commands, quick-start |
| [DEVELOPER_ONBOARDING.md](DEVELOPER_ONBOARDING.md) | Testing mandate, QA protocol, branching strategy |
| [BRANCH_CHRONOLOGY.md](BRANCH_CHRONOLOGY.md) | Full branch history, merge session log, phase table |
| [WALKTHROUGH_PHASE14.md](WALKTHROUGH_PHASE14.md) | Phase 14 agentic tools — architecture, test steps, verified run |
| [docs/PHASE15_RAG.md](docs/PHASE15_RAG.md) | Phase 15 RAG — architecture, module guide, sub-task log |
| [docs/PERFORMANCE_CEILING_REPORT.md](docs/PERFORMANCE_CEILING_REPORT.md) | Full optimization journal, bandwidth math, hardware ceilings, Addendum A/B/C |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | All changes by phase |
| [docs/KERNEL_INTERNALS.md](docs/KERNEL_INTERNALS.md) | VBMI kernel, thread pool, KV cache layout, MoE scatter problem |
| [DEBUGGING_JOURNAL.md](DEBUGGING_JOURNAL.md) | Step-by-step debugging from 1.4 → 16 tok/s |
| [WEIGHT_LOADING_REFERENCE.md](WEIGHT_LOADING_REFERENCE.md) | Complete binary format specification |
| [CPU_LLM_TERNARY_ENGINE.md](CPU_LLM_TERNARY_ENGINE.md) | Original architectural vision — ternary math, hardware adaptation, mmap design |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Complete 37-phase implementation spec (2907 lines) — struct definitions, file inventories, function signatures |
| [MOE_RESEARCH_AND_FIX_PLAN.md](MOE_RESEARCH_AND_FIX_PLAN.md) | DeepSeek MoE optimization research — 8 attempted fixes (P1–P8), profiling data |
| [.github/ROADMAP.md](.github/ROADMAP.md) | Phase status table (✅/🆘/❌), active blockers, full planned phases 17–36 |

---

## Known Limitations

- **40–50 tok/s is physically impossible** on DDR4 with this model — requires ≥45 GB/s
  bandwidth (needs LPDDR5X, HBM, or a much smaller model)
- **8 threads = ~2.4 tok/s** on Tiger Lake due to AVX-512 HT frequency throttle
- `qwen_packed.bin` / `qwen_tokenizer.bin` in the repo root are **legacy files** —
  use `models/bitnet-b1.58-2B-4T.bin` and `models/bitnet-b1.58-2B-4T_tokenizer_proper.bin`

---

*Project Zero — Phase 34+ | BitNet b1.58-2B-4T · DeepSeek-V2-Lite-Chat (GGUF) · Vision pipeline (SigLIP)*
*Best: **36.25 tok/s** (Xeon, PGO+LTO) · **16.1 tok/s** (i5-11300H, dual-channel) · **1.90 tok/s** (DeepSeek MoE, ceiling: 9.8 tok/s)*
*1.80× avg / 1.83× best vs bitnet.cpp on same hardware · 95% of DRAM bandwidth ceiling*
