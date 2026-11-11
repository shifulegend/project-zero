# CLAUDE.md — project-zero (Claude Code entrypoint)

> **Read first, every session** (`docs/ai/session-start-checklist.md`): `docs/ai/project-overview.md`,
> `docs/ai/engineering-rules.md`, `docs/ai/mistakes.md`, `docs/ai/decision-log.md`,
> `docs/ai/tool-sync-policy.md`. `docs/ai/**` is the **canonical source of truth**; this file is a
> thin adapter. Scoped rules load by path from `.claude/rules/{core,docs,tests,config,api}.md`;
> reusable workflows are `.claude/skills/*/SKILL.md`. All of these are living files: update them
> proactively, without being asked (sync procedure in `docs/ai/tool-sync-policy.md`).

## Durable rules (summary — full text in `docs/ai/engineering-rules.md`)
- Extreme modularity; reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_aligned_*`,
  `tn_size_mul*`, `tn_get_free_ram`) before adding abstractions.
- No hardcoding: model shape/tokens/quant come from GGUF metadata; behavior via CLI/config.
- `memset` every struct before partial init (weights/run-state). Trap absurd allocations
  deterministically — never rely on `calloc` returning NULL (macOS over-commits).
- Cross-platform SIMD (x86 AVX2..VNNI / ARM NEON / scalar) all compile; `TN_HAS_*` gates.
- Definition of done: `make release/test/debug` green for **gcc and clang**, golden output
  correct, no kernel perf regression (A/B), docs+adapters synced, user-facing changes
  logged under `[Unreleased]` in `CHANGELOG.md` per `docs/ai/changelog-guidance.md`, installer +
  `pz config` schema updated for any new/changed setting (DoD #6), `README.md` re-read and fixed
  for anything the change made stale (DoD #7), commit checkpoint proposed.
- Public repo: never commit secrets. Keep ASan/UBSan green.
- Published releases are immutable — fixes ship as a new version; old notes only gain a dated
  notice (`docs/RELEASING.md` § Maintaining published releases; details in `.claude/rules/docs.md`).
- No Claude/AI signature or attribution (commit trailers, code comments, etc.) unless the user
  explicitly asks for it that time — see `docs/ai/commit-log-guidance.md`.
- **Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task** —
  record it in `mistakes.md` too, but the record does not replace the fix. Defer only a
  genuinely large architectural change, and say so to the user instead of silently skipping it.
  See `engineering-rules.md` § Bug-fix policy.
- Third-party skill plugins **superpowers** and **mattpocock-skills** are enabled
  (`.claude/settings.json`); repo rules override them — see `docs/ai/tool-sync-policy.md`
  § Third-party skill plugins.

## Large tasks: user stories + checkpointed phases
For work bigger than one sitting (a plan phase, a new subsystem, a multi-file refactor):
- State the goal as user stories (`as a <who>, I can <what>, so that <why>`), each with its
  acceptance check (test, golden output, benchmark number). The stories define done; the plan
  under them may change.
- Split the work into phases that each end at a green checkpoint (build + tests pass, commit
  proposed). Keep a progress file at `.claude/progress/<task>.md`: stories, phase done / next,
  decisions taken, open questions, the exact verify commands. Update it at every checkpoint.
- On resume (new session, or after context compaction) read that file first and continue from
  the last checkpoint instead of re-deriving the plan. Remove the file when the task ships.

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
- **Code references:** `.claude/INDEX.md` (module index), `.claude/CONTEXT.md` (layers, coupling
  points, known issues), `.claude/clusters/CLUSTERS.md`, `.claude/processes/PROCESSES.md`,
  `.claude/BENCHMARK_SUMMARY.md` — large; search them rather than reading them whole.
