# AGENTS.md — project-zero (Antigravity / portability adapter)

> Portability mirror of `gemini/GEMINI.md`; keep the two synchronized on durable repo-wide rules.
> **Read first, every session** (`docs/ai/session-start-checklist.md`):
> `docs/ai/project-overview.md`, `docs/ai/engineering-rules.md`, `docs/ai/mistakes.md`,
> `docs/ai/decision-log.md`, `docs/ai/tool-sync-policy.md`.
> `docs/ai/**` is the **canonical source of truth**; this file is a thin adapter.
> Scoped rules: `.agents/rules/{core,docs,tests,config,api}.md`. Workflows: `.agents/workflows/*.md`.
> All of these are living files: update them proactively, without being asked.

## Durable rules (summary — full text in `docs/ai/engineering-rules.md`)
- Extreme modularity; reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_aligned_*`,
  `tn_size_mul*`, `tn_get_free_ram`) before adding abstractions.
- No hardcoding: model shape/tokens/quant from GGUF metadata; behavior via CLI/config.
- `memset` structs before partial init; overflow-checked size math; trap absurd allocations
  deterministically (never rely on `calloc` returning NULL — macOS over-commits).
- Cross-platform SIMD (x86 AVX2..VNNI / ARM NEON / scalar) all compile; `TN_HAS_*` gates.
- Definition of done: `make release/test/debug` green for **gcc and clang**, golden output
  correct, no kernel perf regression (A/B), docs+adapters synced, user-facing changes logged
  under `[Unreleased]` in `CHANGELOG.md` per `docs/ai/changelog-guidance.md`, installer +
  `pz config` schema updated for any new/changed setting (DoD #6), small commit checkpoints.
- Public repo: never commit secrets. Keep ASan/UBSan green.
- Published releases are immutable — fixes ship as a new version; old notes only gain a dated
  notice. See `docs/RELEASING.md` § Maintaining published releases.
- README reviewed before every major commit (DoD #7): user-facing change → fix what it made
  stale in `README.md` in the same commit. Reader-facing docs say only what the reader needs:
  no "real"/"honest", no investigation narrative (link the report) — see `engineering-rules.md` DoD #7.
- No AI signature or attribution (commit trailers, code comments, etc.) unless the user
  explicitly asks for it that time — see `docs/ai/commit-log-guidance.md`.
- Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task; a
  `mistakes.md` entry records it but does not replace the fix. Defer only a genuinely large
  architectural change, and say so to the user. See `engineering-rules.md` § Bug-fix policy.
- Third-party skill plugins **superpowers** and **mattpocock-skills** are used (per-tool install commands in `docs/ai/tool-sync-policy.md`); repo rules override them — see `docs/ai/tool-sync-policy.md` § Third-party skill plugins.

## Build / run (verified)
```bash
make release CC=gcc && make test CC=gcc && make debug CC=gcc   # repeat CC=clang
make test-tsan CC=gcc     # concurrency changes only (thread pool / parallel matmul / API mutex)
./pz --model models/<m>.gguf --prompt "..." --max-tokens 16 --temperature 0 --threads 4
```

## Pointers (read only when needed)
- **Project overview.** project-zero is a from-scratch, CPU-optimized LLM inference engine
  targeting BitNet b1.58 ternary weights and DeepSeek-V2 (MoE + MLA), with an OpenAI-compatible
  HTTP API. Architecture, directory map, terminology and build/test/run details are canonical in
  `docs/ai/project-overview.md` — keep that file (not this one) authoritative.
- **GitNexus MCP** (skills, tools, resources, graph schema, quick start): `.claude/GITNEXUS.md`.
  Use it when exploring architecture, tracing a bug, sizing a blast radius or refactoring through
  the knowledge graph; the gitnexus skills are in `.claude/skills/gitnexus/*/SKILL.md`.
