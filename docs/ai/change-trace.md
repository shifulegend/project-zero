# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-06-19.

### 2026-06-19 — Portable `make dist` build + GitHub Release pipeline
- What: Added a portable distribution build and a release workflow that attaches a prebuilt
  x86-64 Linux binary to a GitHub Release. New `make dist` target compiles the bulk at
  `-march=x86-64-v2` with per-file SIMD ISA flags (AVX2/AVX-512/VNNI) so runtime `simd_dispatch`
  lights up the best tier on the host; `simd_dispatch.c` is compiled at the baseline with
  `-DTN_FORCE_DISPATCH_ALL` (new guard, no SIMD codegen there) so all branches are present;
  static `-static-libstdc++ -static-libgcc` leaves only libc/libm deps. Added a `--version`/`-v`
  flag (works without `--model`) and a `-DPZ_VERSION` build stamp (banner no longer hardcodes
  "Phase 16"). CMake gains an off-by-default `PZ_DIST` option mirroring the Makefile.
- Why: user asked for a prebuilt x86-64 binary on a GitHub Release, tested thoroughly; the
  existing `-march=native` release is not distributable on varied CPUs.
- Areas: `Makefile` (dist target, per-TU ISA rules, version stamp), `src/math/simd_dispatch.c`
  (`TN_FORCE_DISPATCH_ALL`), `src/cli/{args.c,main.c}` + `include/cli/args.h` (`--version`),
  `CMakeLists.txt` (`PZ_DIST`, `PZ_VERSION`), `.github/workflows/release.yml` (new),
  `.github/workflows/ci.yml` (dist build-check), `docs/RELEASING.md` (new).
- Result: gcc release/test(46)/debug/dist and clang release/debug/dist green; portable binary
  links only libc/libm; golden output (France→Paris, Germany→Berlin) correct across
  scalar/avx2/avx512f/vnni and T=1/2/8 on the SmolLM2-135M F16 model.
- Commit/PR: on branch `claude/x86-64-github-release-8xduj2`.

### 2026-06-14 — Docs reflect dense GGUF support (SmolLM2 + generic loader)
- What: README, ROADMAP, and project-overview said the engine runs only BitNet and
  DeepSeek-V2-Lite, but the benchmark docs (`.claude/BENCHMARK_SUMMARY.md`,
  `docs/PERFORMANCE_CEILING_REPORT.md`) already benchmark **SmolLM2-135M-Instruct F16**
  (dense GGUF) up to 83.79 tok/s, and `config_from_gguf()` in `src/core/gguf_loader.c` is
  architecture-agnostic. Added a third support tier: dense GGUF transformers (Llama-family)
  via the generic loader, with SmolLM2 as the verified model and other architectures flagged
  as loads-but-untested. MoE/MLA acceleration remains DeepSeek-V2-specific.
- Why: docs understated actual, already-tested capability; user asked for the correct picture.
- Areas: `README.md` (intro, new "Dense GGUF Models" section, footer), `.github/ROADMAP.md`
  (perf snapshot), `docs/ai/project-overview.md` (Purpose). Lean adapters (AGENTS/copilot/
  GEMINI/CLAUDE) left as-is per tool-sync-policy — they describe the targeted/special-cased
  architectures, not an exhaustive model list. Historical benchmark addenda left untouched.
- Branch: `claude/readme-llm-support-docs-3tg13v`.

### 2026-06-14 — README accuracy pass + repo best-practices + docs reorg
- What: (1) Corrected README intro to match canonical scope (BitNet + DeepSeek-V2-Lite
  GGUF + vision/agentic/RAG), kept "written in C", reframed Python as temporary
  dev/test tooling (zero-Python goal), added LLM-agnostic goal. (2) Reconciled the
  Phase 21 HTTP API claim to 🔄 partial/experimental across README, ROADMAP, and
  project-overview (it is real and wired but serial/loopback-only/untested-in-CI).
  (3) Added community-health files: `.github/CODEOWNERS`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`, `.editorconfig`, `CITATION.cff`. (4) Moved 27
  archival/design/report `.md` files out of the repo root into
  `docs/{architecture,phases,reports,weight-loading}/` and `docs/`, leaving 8 entry-point
  docs at root; rewrote all inbound markdown links path-aware and fixed 4 dangling links
  (verified 0 dangling repo-wide).
- Why: README/roadmap/overview contradicted each other and the tree; root had 35 `.md`
  files hurting discoverability; repo was missing standard GitHub best-practice files.
- Areas: `README.md`, `.github/ROADMAP.md`, `docs/ai/project-overview.md`, `.editorconfig`,
  `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/config.yml`,
  `CITATION.cff`, and `docs/{architecture,phases,reports,weight-loading}/**`.
- Branch: `claude/readme-accuracy-review-y9jk7u`.

### 2026-06-07 — Document branch-hygiene convention
- What: Added a "Version control & branch hygiene" section to `engineering-rules.md` (delete
  merged branches; enable auto-delete-head-branches; avoid flag/placeholder branches; don't
  commit artifacts/models/logs).
- Why: post-merge cleanup surfaced redundant branches; this sandbox's git proxy blocks ref
  deletion, so the convention + the repo auto-delete setting prevent future accumulation.
- Areas: `docs/ai/engineering-rules.md`. Canonical-only (adapters stay lean per tool-sync-policy).

### 2026-06-07 — Cross-tool AI development system
- What: Added `docs/ai/**` canonical docs + Claude/Copilot/Antigravity adapters.
- Why: one source of truth; continuity across Claude Code, GitHub Copilot, Antigravity.
- Areas: `docs/ai/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.github/`, `AGENTS.md`,
  `gemini/GEMINI.md`, `.agents/`.
- Commit/PR: (this change) — see commit checkpoints in PR to `master`.

### 2026-06-07 — Green CI + regression verification (PR #6, merged `cb9fa52`)
- What: Fixed the CI cascade and verified no regression across SmolLM2/BitNet/DeepSeek.
- Why: CI had never run to completion; ensure merge-safety and no regression.
- Areas: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`,
  `tests/test_vision_components.c`, `Makefile`, `src/core/run_state.c`,
  `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
- Result: all 7 CI checks green on PR #6 and on `master`; secrets scan clean (215 commits).
