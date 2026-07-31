# GEMINI.md — project-zero (Google Antigravity entrypoint)

> **Read first, every session** (`docs/ai/session-start-checklist.md`):
> `docs/ai/project-overview.md`, `docs/ai/engineering-rules.md`, `docs/ai/mistakes.md`,
> `docs/ai/decision-log.md`, `docs/ai/tool-sync-policy.md`.
> `docs/ai/**` is the **canonical source of truth**; this file is a thin adapter and is kept
> synchronized with `AGENTS.md` on durable repo-wide rules.
> Scoped rules: `.agents/rules/{core,docs,tests,config,api}.md`. Workflows: `.agents/workflows/*.md`.
> **All of these are dynamic — update them proactively, even without being asked**
> (procedure in `docs/ai/tool-sync-policy.md`).

## What this project is
CPU-optimized LLM inference engine in C99/C++17 — BitNet ternary + DeepSeek-V2 (MoE/MLA) — with
an OpenAI-compatible HTTP API. Builds with `Makefile` (primary) and `CMakeLists.txt`.
See `docs/ai/project-overview.md`.

## Durable rules (summary — full text in `docs/ai/engineering-rules.md`)
- Extreme modularity; reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_aligned_*`,
  `tn_size_mul*`, `tn_get_free_ram`) before adding abstractions.
- No hardcoding: model shape/tokens/quant from GGUF metadata; behavior via CLI/config/flags.
- `memset` structs before partial init; overflow-checked size math; trap absurd allocations
  deterministically (never rely on `calloc` returning NULL — macOS over-commits).
- Cross-platform SIMD (x86 AVX2..VNNI / ARM NEON / scalar) must all compile; `TN_HAS_*` gates;
  portable prefetch `TN_PREFETCH_T1`.
- Definition of done: `make release/test/debug` green for **gcc and clang**, golden output
  correct, no kernel perf regression (A/B), docs+adapters synced, small commit checkpoints.
- Public repo: never commit secrets. Keep ASan/UBSan green.

## Build / run (verified)
```bash
make release CC=gcc && make test CC=gcc && make debug CC=gcc   # repeat CC=clang
./adaptive_ai_engine --model models/<m>.gguf --prompt "..." --max-tokens 16 --temperature 0 --threads 4
```
