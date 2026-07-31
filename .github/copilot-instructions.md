# GitHub Copilot — project-zero (repo-wide instructions)

**Read the canonical docs first, every session** (source of truth lives in `docs/ai/**`, not here):
`docs/ai/project-overview.md`, `docs/ai/engineering-rules.md`, `docs/ai/mistakes.md`,
`docs/ai/decision-log.md`, `docs/ai/session-start-checklist.md`, `docs/ai/tool-sync-policy.md`.

This file and everything under `.github/instructions/`, `.github/prompts/` are **thin, dynamic
adapters** — update them proactively (no prompt needed) whenever a durable rule or workflow
changes, after first updating the canonical docs (see `docs/ai/tool-sync-policy.md`).

## What this project is
CPU-optimized LLM inference engine in C99/C++17 (BitNet ternary + DeepSeek-V2 MoE/MLA) with an
OpenAI-compatible HTTP API. Build: `Makefile` (primary) + `CMakeLists.txt`. See project-overview.

## Durable rules (summary; full text in `docs/ai/engineering-rules.md`)
- Extreme modularity; reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_aligned_*`,
  `tn_size_mul*`, `tn_get_free_ram`) before adding abstractions.
- No hardcoding: model shape/tokens/quant from GGUF metadata; behavior via CLI/config/flags.
- `memset` structs before partial init; overflow-checked size math; trap absurd allocations
  deterministically (never rely on `calloc` returning NULL — macOS over-commits).
- Cross-platform SIMD (x86 AVX2..VNNI / ARM NEON / scalar) must all compile; feature-gate with
  `TN_HAS_*`; portable prefetch `TN_PREFETCH_T1`.
- Definition of done: `make release/test/debug` green for **gcc and clang**, golden output
  correct, no kernel perf regression (A/B), docs+adapters synced, small commit checkpoints.
- Public repo: never commit secrets. Keep ASan/UBSan green.

## Scoped rules & workflows
- Scoped recurring rules: `.github/instructions/{core,docs,tests,config,api}.instructions.md`.
- Reusable workflows: `.github/prompts/*.prompt.md` (start-session, plan-task, implement-task,
  review-changes, debug-failure, update-project-memory).
