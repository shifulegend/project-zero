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

See [`CPU_LLM_TERNARY_ENGINE.md`](docs/architecture/CPU_LLM_TERNARY_ENGINE.md) for a full architectural walkthrough.

---

## 🔥 Highest Impact Areas Right Now

The full phase roadmap is at [`.github/ROADMAP.md`](.github/ROADMAP.md). The two active blockers that unblock the most downstream phases:

### 🆘 Active Blockers

| Area | Current state | Target | Discussion | Skill needed |
|---|---|---|---|---|
| **MoE expert weight repacking** | DeepSeek MoE at ~1 tok/s — 86% L3 miss rate due to non-contiguous expert offsets in GGUF | ≥ 9 tok/s | [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1) | C, GGUF Q4_K layout, memory layout |
| **Native Q4_K matmul kernel** | Dense layers dequant Q4_K → FP32 before matmul | 4× speedup on DeepSeek dense layers | [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1) | C, AVX-512 intrinsics |

See [`MOE_RESEARCH_AND_FIX_PLAN.md`](docs/architecture/MOE_RESEARCH_AND_FIX_PLAN.md) for 8 previous fix attempts (P1–P8) with profiling data — read this before starting on expert repacking.

### 📋 Next Up (Fully Specified, Ready to Implement)

These phases are completely spec'd in [`IMPLEMENTATION_PLAN.md`](docs/architecture/IMPLEMENTATION_PLAN.md) with struct definitions, function signatures, and file inventories. Good for contributors who want a well-defined scope:

| Phase | Feature | Skill needed |
|---|---|---|
| **37.2–37.5** | GGUF quant types: Q3_K, Q4_1, Q8_1, Q5_1, Q5_K, Q6_K | C, fixed-point arithmetic |
| **19** | LoRA adapters — hot-swappable low-rank inference | C, linear algebra |
| **20** | Grammar-constrained decoding / JSON mode — FSM + BNF parser | C, automata theory |
| **24** | YaRN/NTK context scaling — RoPE frequency extension | C, transformer math |
| **21** | OpenAI-compatible API layer (`/v1/chat/completions`, SSE streaming) | C, HTTP parsing |

### ✅ Recently Completed (For Reference)

| Item | Status |
|---|---|
| P8 NaN guard validation on Q2_K model | ✅ Done — Q2_K dequant implemented (Phase 37.1) |
| Phase 16-S SIMD dispatch (AVX-512 VNNI, AVX-VNNI, ARM NEON) | ✅ Done |
| Calibration system (DRAM BW probe + L3 size detection) | ✅ Done |
| Vision pipeline Phase 11 (SigLIP + MLP projector) | ✅ Done |

---

## ✅ Code Standards

- **Language:** C99 (no C++, no external libraries, no Python at runtime)
- **No hardcoding:** No hardcoded paths, model dimensions, or magic numbers — read from config/GGUF metadata
- **Test after every change:** `make test` must pass. Add a test for new functionality.
- **Memory:** Use `tn_aligned_calloc()` for SIMD buffers (64-byte aligned). No `malloc` in hot paths.
- **SIMD:** Always add a scalar fallback path. Wrap AVX-512 code in `#ifdef __AVX512F__`.
- **Documentation:** Update the relevant `.md` file. Benchmark before and after for perf changes.
- **One logical change per PR, at most 400 changed lines.** Lockfiles, generated
  code and bundles, snapshots, vendored files, translations, test fixtures,
  images and models do not count (`EXCLUDE` in `tools/check_pr_size.py`), so
  they never need an override. CI (`tools/check_pr_size.py`) enforces the limit.
- **`size-override` is a documented exception, not a routine shortcut.** A
  maintainer adds the label only for work that must land in one piece: an
  automated dependency or framework upgrade, a mechanical search-and-replace
  refactor across many files (reformat, rename, moving files), a data migration
  or static-asset addition, or an overall clean-up of the code. The PR
  description must then contain a filled "Size override justification" section
  that says why the change cannot be split into smaller, stacked PRs; CI
  (`tools/check_size_override.py`) fails a labelled PR without it. A reviewer
  who is not convinced asks for the split instead.
- **Test first for behaviour changes and bug fixes:** write the failing test, make it pass, then clean up (Red -> Green -> Refactor), and say so in the PR.
- **Local checks (optional):** `pip install pre-commit && pre-commit install`
  runs `ruff`, `clang-format` and the repository checks (TODO register, Markdown
  fences, header guards, file sizes, root files) before each commit; the config
  is `.pre-commit-config.yaml`, and CI runs the same checks. Editors that read
  `.editorconfig` get the 100-column limit.
- **Review comments:** prefix a non-blocking remark with `Nit:`; it never blocks a merge. Review the whole file's context, not only the diff hunk.

---

## 👀 Review & hotfixes

- **Turnaround:** the maintainer aims to give a first response to a pull request within 7 days. This is a goal, not a guarantee.
- **Hotfixes:** a change merged outside the normal checks (for example an urgent fix with a test left for later) opens a follow-up issue in the same hour, labelled `hotfix-followup`, listing what was skipped; it is closed within 14 days.
- **Nit:** comments are non-blocking (see Code Standards above).

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
