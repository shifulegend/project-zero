# Project Zero

[![CI](https://github.com/shifulegend/project-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/shifulegend/project-zero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An LLM inference engine for CPUs, written from scratch in C. One binary (`pz`), no GPU, no Python,
no third-party libraries. It runs ternary models (BitNet b1.58, Falcon3-1.58bit, Ternary-Bonsai),
mixture-of-experts models (DeepSeek-V2, Qwen3-MoE) and ordinary GGUF models, and serves them over an
OpenAI-compatible API with a built-in web chat.

<p align="center">
  <img src="docs/demo_bitnet.gif" width="720" alt="pz answering a question with BitNet b1.58-2B-4T on a 4-vCPU Xeon, 15 tok/s">
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

You need GCC or Clang (with its C++ compiler, for the chat-template parser), make, pthreads and
libm. GGUF models need a CPU with at least AVX2 on x86; BitNet runs on any x86-64-v2 CPU. Installer options, manual install and
`pz config` (saved settings): [docs/INSTALL.md](docs/INSTALL.md).

## Benchmarks

Decode speed in tokens per second, best of 3 runs, temperature 0, nothing else running on the
machine. Older runs, other machines and methods: [docs/BENCHMARK.md](docs/BENCHMARK.md).

### Intel Xeon @ 2.80 GHz (Cascade Lake), 4 vCPU — 2026-10-29

500 tokens per run (DeepSeek: 200). Tok/s at 4 threads; the chart has 1–4 threads.

| Model | Project Zero | llama.cpp | ik_llama.cpp | bitnet.cpp |
|---|---|---|---|---|
| SmolLM2-135M F16 | 70.8 | 70.8 | **75.5** | – |
| SmolLM2-1.7B Q4_K_M | 14.9 | 16.3 | **18.2** | – |
| BitNet b1.58 2B-4T, official I2_S GGUF | 13.3 | – | **19.4** (converted to IQ2_BN) | 14.2 |
| DeepSeek-V2-Lite-Chat Q4_K_S | 12.3 | 11.5 | **14.2** | – |

<p align="center">
  <img src="benchmark_results/sweep_h2h_2026-10-29/throughput_vs_threads.png" width="860" alt="Decode tok/s vs threads on the 2.80 GHz Xeon: Project Zero, llama.cpp, ik_llama.cpp and bitnet.cpp">
</p>

Logs and screenshots: [sweep](benchmark_results/sweep_h2h_2026-10-29/) ·
[DeepSeek](benchmark_results/deepseek_h2h_2026-10-29/).

On a 2.10 GHz Xeon with AVX-512 BF16 (2026-10-23), Project Zero was 27–36% faster than bitnet.cpp
on BitNet (54–60% with `--classifier int4`) and 18–34% faster than llama.cpp on SmolLM2-135M.

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
- Sampling controls, JSON mode (`--json`), speculative decoding (`--draft-model`), best-of-N
  (`--best-of`) and tree search (`--mcts`)
- LoRA adapters (`--lora`), image input (`--vision`), RAG memory (`--memory-db`), a tool-use agent
  (`/agent` in the terminal chat), watermarking (`--watermark`)
- KV cache in f32, int8 or int4 (`--kv-type`), on disk when RAM runs out (`--kv-offload`), with
  prompt caching (`--prompt-cache`)
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

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Open problems where help is
welcome:

- **Q4_K on dense models.** Qwen3-8B Q4_K_M is 1.6–1.9× slower than llama.cpp on the same machine.
  Six attempts are written up in the
  [investigation](docs/BENCHMARK.md#qwen3-8b).
- **INT8/INT4 output layer.** Since the fix for [issue #27](https://github.com/shifulegend/project-zero/issues/27)
  removed the VNNI fast path in [src/math/parallel_matmul.c](src/math/parallel_matmul.c),
  `--classifier int4` is slower than the default at 4 threads on the 2.80 GHz Xeon (12.3 vs 13.3 tok/s).
- **Cascade Lake.** On the 2.80 GHz Xeon, which has no AVX-512 BF16, Project Zero trails
  ik_llama.cpp on every model and bitnet.cpp on BitNet.

Changes in each release: [CHANGELOG.md](CHANGELOG.md). Reporting a vulnerability: [SECURITY.md](SECURITY.md).

> If you like my work or if it helped you, buy me a coffee or A100😉
> [![Sponsor shifulegend](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red)](https://github.com/sponsors/shifulegend)

## License

[MIT](LICENSE)
