# Contributing to Project Zero

Thank you for your interest in contributing! Project Zero is a from-scratch CPU-only LLM inference engine in C, and community help is genuinely needed — especially on the MoE performance bottleneck.

---

## 📋 Before You Start

1. **Read [`GOLDEN_RULES.md`](GOLDEN_RULES.md)** — mandatory for all contributors and AI agents. No hardcoding, test after every change, document every step.
2. **Read [`DEVELOPER_ONBOARDING.md`](DEVELOPER_ONBOARDING.md)** — architecture overview, testing protocol, branching strategy.
3. **Check open Discussions** — especially [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1) (MoE help wanted) and [Discussion #2](https://github.com/shifulegend/project-zero/discussions/2) (phase roadmap).

---

## 🙋 How to Contribute

### Reporting Bugs

Use the **Bug Report** issue template. Please include:
- Your OS, CPU model, compiler version (`gcc --version`)
- Exact command you ran
- Expected output vs actual output
- Any error messages or crash logs

### Suggesting Features / Optimizations

Use the **Feature Request** issue template. For performance ideas, include:
- The specific bottleneck you're targeting
- Theoretical bandwidth/compute analysis if you have it
- Reference implementations (llama.cpp, ggml, etc.)

### Submitting Code

1. **Fork** the repository
2. **Create a branch**: `git checkout -b fix/your-description` or `feat/your-description`
3. **Build and test**: `make release && make test` — all tests must pass
4. **Document your change** — update the relevant `.md` file (README, BENCHMARK_REPORT, DEBUGGING_JOURNAL, etc.)
5. **Open a Pull Request** using the PR template

---

## 🏗️ Architecture Primer

Project Zero is a ternary weight ({−1, 0, +1}) LLM inference engine. Key design choices:
- Weights packed 4-per-byte → 13× smaller than FP16
- `mmap` instead of `malloc+fread` → never OOM, auto-scales to available RAM
- Runtime CPUID dispatch → AVX-512 VNNI / AVX2 / NEON / Scalar
- C11 atomics spin-then-sleep thread pool → eliminates futex syscalls
- KV cache layout `[layer][head][pos][dim]` → sequential reads, no scatter

See [`CPU_LLM_TERNARY_ENGINE.md`](CPU_LLM_TERNARY_ENGINE.md) for a full architectural walkthrough.

---

## 🔥 Highest Impact Areas Right Now

| Area | Description | Skill needed |
|---|---|---|
| **Expert weight repacking** | Interleave Q4_K MoE expert weights at load time for contiguous DRAM reads. Currently 13× behind llama.cpp. | C, GGUF format, memory layout |
| **Native Q4_K matmul kernel** | AVX-512 VNNI kernel operating directly on 4-bit super-blocks (no F32 dequant). | C, AVX-512 intrinsics |
| **P8 NaN fix validation** | Run one command on `DeepSeek-V2-Lite-Chat-Q2_K.gguf` and report output. | Just running a binary |
| **YaRN/NTK context scaling** | Phase 24 — self-contained, well-specified RoPE extension | C, transformer math |

---

## ✅ Code Standards

- **Language:** C99 (no C++, no external libraries, no Python at runtime)
- **No hardcoding:** No hardcoded paths, model dimensions, or magic numbers — read from config/GGUF metadata
- **Test after every change:** `make test` must pass. Add a test for new functionality.
- **Memory:** Use `tn_aligned_calloc()` for SIMD buffers (64-byte aligned). No `malloc` in hot paths.
- **SIMD:** Always add a scalar fallback path. Wrap AVX-512 code in `#ifdef __AVX512F__`.
- **Documentation:** Update the relevant `.md` file. Benchmark before and after for perf changes.

---

## 🔀 Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Bug fix | `fix/<description>` | `fix/moe-nan-cascade` |
| Feature | `feat/<description>` | `feat/q4k-native-kernel` |
| Performance | `perf/<description>` | `perf/expert-repacking` |
| Documentation | `docs/<description>` | `docs/contributing-guide` |

---

## 💬 Questions?

Open a [Discussion](https://github.com/shifulegend/project-zero/discussions) — we prefer discussions over issues for questions.

---

*Project Zero is maintained by the community. All contributions, however small, are welcome.*
