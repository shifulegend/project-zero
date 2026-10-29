# Project Zero

[![CI](https://github.com/shifulegend/project-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/shifulegend/project-zero/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/shifulegend/project-zero)](https://github.com/shifulegend/project-zero/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An LLM inference engine for CPUs, written from scratch in C. One binary (`pz`), no GPU, no Python,
no third-party libraries. It runs ternary models (BitNet b1.58, Falcon3-1.58bit, Ternary-Bonsai),
mixture-of-experts models (DeepSeek-V2, Qwen3-MoE) and ordinary GGUF models, and serves them over an
OpenAI-compatible API with a built-in web chat.

<p align="center">
  <img src="docs/demo_bitnet.gif" width="720" alt="pz generating text with BitNet b1.58-2B-4T in a terminal on an i5-11300H laptop">
</p>

## Quick start

Linux x86-64:

```bash
curl -fsSL https://raw.githubusercontent.com/shifulegend/project-zero/master/install.sh | sh
curl -fL -o SmolLM2-135M-Instruct-f16.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf
pz --model SmolLM2-135M-Instruct-f16.gguf --prompt "What is the capital of France?"
pz --model SmolLM2-135M-Instruct-f16.gguf --server --port 8080   # web chat at http://localhost:8080
```

The installer checks the release checksum and needs no root. To build from source instead:

```bash
git clone https://github.com/shifulegend/project-zero.git && cd project-zero
make demo   # builds, downloads SmolLM2-135M, prints "The capital of France is Paris."
```

You need GCC or Clang (with its C++ compiler), make, pthreads and libm. GGUF models need a CPU with
at least AVX2 on x86; BitNet runs on any x86-64-v2 CPU. Installer options, manual install and
`pz config` (saved settings): [docs/INSTALL.md](docs/INSTALL.md).

## Benchmarks

Decode speed in tokens per second, best of 3 runs, temperature 0, nothing else running on the
machine. Each machine shows its most recent run. Older runs and methods:
[docs/BENCHMARK.md](docs/BENCHMARK.md).

### Intel Xeon @ 2.10 GHz (model 207, AVX-512 BF16), 4 vCPU — 2026-10-23

500 tokens per run.

| Model | Project Zero, 1 / 2 / 3 / 4 threads | Compared with | Difference |
|---|---|---|---|
| SmolLM2-135M F16 | 29.4 / 59.0 / 81.2 / 106.3 | llama.cpp: 25.0 / 46.3 / 62.9 / 79.5 | 18–34% faster |
| SmolLM2-1.7B Q4_K_M | 4.95 / 9.23 / 12.75 / 16.44 | llama.cpp: 5.01 / 9.27 / 13.40 / 17.65 | 0–7% slower |
| BitNet b1.58 2B-4T, official I2_S GGUF | 6.42 / 11.64 / 16.31 / 19.71 | bitnet.cpp: 4.87 / 8.95 / 12.00 / 15.46 | 27–36% faster |
| same, with `--classifier int4` | 7.74 / 14.10 / 18.48 / 24.72 | bitnet.cpp, as above | 54–60% faster |

Raw logs: [benchmark_results/sweep_h2h_2026-10-23/](benchmark_results/sweep_h2h_2026-10-23/).
Also on this machine:

- DeepSeek-V2-Lite-Chat Q4_K_S, 4 threads: 11.7–12.6 tok/s; llama.cpp 9.4–10.5 (2026-10-21).
- `--parallel 4` with 4 chat clients: 2.2–2.3× the total throughput of serving them one at a time
  ([results](benchmark_results/parallel_2026-10-22/), 2026-10-22).

### Intel Core i5-11300H laptop, 4 cores / 8 threads — 2026-06-21

500 tokens per run. BitNet uses converted weights with an INT4 output layer.

| Threads | BitNet b1.58 2B-4T: Project Zero | bitnet.cpp | SmolLM2-135M F16: Project Zero | llama.cpp |
|---|---|---|---|---|
| 1 | **16.98** | 2.04 | **58.00** | 53.50 |
| 2 | **27.99** | 3.76 | **86.03** | 84.20 |
| 4 | **35.79** | 6.64 | 98.28 | **102.10** |
| 8 | **32.03** | 6.04 | 83.32 | **86.30** |

BitNet is 4.9–8.3× faster than bitnet.cpp across 1–8 threads. Every thread count and 40
screenshots: [docs/BENCHMARK.md](docs/BENCHMARK.md) ·
[OpenBenchmarking.org result](https://openbenchmarking.org/result/2606208-SHIF-PROJECT03).

### Other machines

| Model | Result | Machine, date |
|---|---|---|
| Ternary-Bonsai-27B Q2_0 | 4.2–4.8× faster than PrismML's llama.cpp fork (4 threads: 2.97 vs 0.70 tok/s) | 4-core AVX-512 Xeon VM, 2026-07-18 ([captures](benchmark_results/sweep3_2026-07-18/)) |
| Qwen3-8B Q4_K_M | 2.5–2.7 tok/s; llama.cpp 4.0–4.8 | AVX2-only VM, 2026-07-31 ([investigation](docs/BENCHMARK.md#qwen3-8b)) |
| Ternary-Bonsai-27B Q2_0 | ~0.1 tok/s, reported by a user | i5-8350U, AVX2 ([report](https://old.reddit.com/r/LocalLLaMA/comments/1v3pn1w/built_a_fromscratch_bitnet_inference_engine_in/p0wktod/)) |

Output quality: perplexity is within 0.7% of llama.cpp on SmolLM2-135M/1.7B, Qwen2.5-0.5B,
Qwen3-0.6B and Mamba-130m ([report](docs/reports/ACCURACY_PARITY_2026-10-19.md)).

Post your own numbers in [Discussion #3](https://github.com/shifulegend/project-zero/discussions/3).

## Models

Tested end to end:

| Model | Format |
|---|---|
| BitNet b1.58 2B-4T | converted `.bin`, official I2_S GGUF |
| Falcon3-3B-Instruct-1.58bit | I2_S GGUF |
| Ternary-Bonsai-27B (Qwen 3.5 hybrid) | Q2_0 GGUF |
| DeepSeek-V2-Lite-Chat | Q4_K_S, Q2_K GGUF |
| Qwen3-8B, Qwen3-0.6B, Qwen2.5-0.5B-Instruct | Q4_K_M, Q8_0 GGUF |
| SmolLM2-135M / 1.7B-Instruct | F16, Q8_0, Q4_K_M GGUF |
| Mamba-130m | F16 GGUF |

Other models with these architectures load the same way. Qwen3-MoE loads but has not been tested
yet. The official BitNet GGUF ships a broken chat template; pass
`--chat-template-file docs/templates/bitnet-b1.58.jinja`.

## Features

- Web chat and an OpenAI-compatible `/v1/chat/completions`, with streaming, cancel, optional API
  key and CORS, `/metrics`, and `--parallel N` to serve several users at once
- Sampling controls, JSON mode, speculative decoding, best-of-N and tree search
- LoRA adapters, image input (`--vision`), RAG memory, an agent tool loop, watermarking
- KV cache in f32, int8 or int4, on disk when RAM runs out, with prompt caching
- A model's layers split across machines (`--remote-layers`)
- The fastest SIMD path picked at startup: AVX-512 VNNI, AVX-512, AVX-VNNI, AVX2, ARM NEON/dotprod
  or scalar

Every flag: `pz --help`. Every web UI control and API route: [docs/WEBUI_GUIDE.md](docs/WEBUI_GUIDE.md).

| Web chat, light theme | Settings, dark theme |
|---|---|
| ![Web chat showing a reply with token counts](docs/design/screenshots/webui-reply-light-2026-10-23.png) | ![Settings panel with sampling and server defaults](docs/design/screenshots/webui-settings-dark-2026-10-23.png) |

## Development

```bash
make release   # optimised build for this CPU
make test      # all tests, under AddressSanitizer and UBSan
make dist      # portable x86-64 binary, as shipped in releases
```

Kernels are in [src/math/](src/math/), model loading in [src/core/](src/core/), the forward pass
in [src/transformer/](src/transformer/), the server in [src/api/](src/api/) and the web UI in
[webui/src/](webui/src/). How the kernels work: [docs/KERNEL_INTERNALS.md](docs/KERNEL_INTERNALS.md).
What is done and planned: [docs/architecture/IMPLEMENTATION_PLAN.md](docs/architecture/IMPLEMENTATION_PLAN.md).

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Two open problems where help
is welcome:

- **Q4_K on dense models.** Qwen3-8B Q4_K_M is 1.6–1.9× slower than llama.cpp on the same machine.
  Six attempts are written up in the
  [investigation](docs/BENCHMARK.md#qwen3-8b).
- **INT8/INT4 output layer.** The fix for [issue #27](https://github.com/shifulegend/project-zero/issues/27)
  removed the VNNI fast path in [src/math/parallel_matmul.c](src/math/parallel_matmul.c), and its
  speed has not been measured since.

Changes in each release: [CHANGELOG.md](CHANGELOG.md). Reporting a vulnerability: [SECURITY.md](SECURITY.md).

> If you like my work or if it helped you, buy me a coffee or A100😉
> [![Sponsor Nitika](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://github.com/sponsors/shifulegend)

## License

[MIT](LICENSE)
