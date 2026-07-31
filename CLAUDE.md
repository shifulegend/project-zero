# CLAUDE.md — project-zero (Claude Code entrypoint)

> **Read first, every session** (see `docs/ai/session-start-checklist.md`):
> `docs/ai/project-overview.md`, `docs/ai/engineering-rules.md`, `docs/ai/mistakes.md`,
> `docs/ai/decision-log.md`, `docs/ai/tool-sync-policy.md`.
> `docs/ai/**` is the **canonical source of truth**; this file is a thin adapter.
> Scoped rules live in `.claude/rules/{core,docs,tests,config,api}.md`; reusable workflows in
> `.claude/skills/*/SKILL.md`. **All of these are dynamic — keep them updated proactively,
> even without being asked** (sync procedure in `docs/ai/tool-sync-policy.md`).

## Durable rules (summary — full text in `docs/ai/engineering-rules.md`)
- Extreme modularity; reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_aligned_*`,
  `tn_size_mul*`, `tn_get_free_ram`) before adding abstractions.
- No hardcoding: model shape/tokens/quant come from GGUF metadata; behavior via CLI/config.
- `memset` every struct before partial init (weights/run-state). Trap absurd allocations
  deterministically — never rely on `calloc` returning NULL (macOS over-commits).
- Definition of done: `make release/test/debug` green for **gcc and clang**, golden output
  correct, no kernel perf regression (A/B), docs+adapters synced, commit checkpoint proposed.
- Public repo: never commit secrets. Keep ASan/UBSan green.
- **Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task** —
  document it in `mistakes.md`, but that never substitutes for the fix. Only defer for a
  genuinely large architectural change, and flag that to the user explicitly instead of silently
  skipping it. See `engineering-rules.md` § Bug-fix policy.

## Build / run (verified)
```bash
make release CC=gcc && make test CC=gcc && make debug CC=gcc   # repeat CC=clang
./adaptive_ai_engine --model models/<m>.gguf --prompt "..." --max-tokens 16 --temperature 0 --threads 4
```

<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **project-zero** — a CPU-first LLM inference engine in C/C++ with BitNet/DeepSeek ternary weight support, MoE routing, MLA attention, and an HTTP API layer.

> **Index not yet built.** Run `npx gitnexus analyze` in the project root to generate the live knowledge graph (`.gitnexus/`), populate symbol counts, and enable all MCP tools below.

## Skills

| Skill | When to use |
|-------|-------------|
| `gitnexus-exploring` | Understanding architecture, tracing execution flows |
| `gitnexus-debugging` | Tracing bugs, finding callers, diagnosing failures |
| `gitnexus-impact-analysis` | Blast-radius before changing code |
| `gitnexus-refactoring` | Safe renames, extractions, restructuring |

Skills live at `.claude/skills/gitnexus/*/SKILL.md`.

## MCP Tools (available after `npx gitnexus mcp`)

| Tool | Description |
|------|-------------|
| `gitnexus_query` | Semantic search → execution flows + symbols |
| `gitnexus_context` | 360° view of a symbol (callers, callees, processes) |
| `gitnexus_impact` | Upstream/downstream blast radius |
| `gitnexus_detect_changes` | Map git-diff to affected execution flows |
| `gitnexus_rename` | Multi-file symbol rename with dry-run preview |
| `gitnexus_cypher` | Raw Cypher queries against the knowledge graph |

## Resources

| Resource | What you get |
|----------|-------------|
| `gitnexus://repos` | All indexed repos |
| `gitnexus://repo/project-zero/context` | Stats + staleness check |
| `gitnexus://repo/project-zero/clusters` | Functional areas with cohesion scores |
| `gitnexus://repo/project-zero/processes` | All execution flows |
| `gitnexus://repo/project-zero/process/{name}` | Step-by-step execution trace |

## Graph Schema

```
(Symbol)-[:CodeRelation {type: 'CALLS' | 'IMPORTS' | 'IMPLEMENTS' | 'EXTENDS'}]->(Symbol)
(Symbol)-[:BelongsTo]->(Cluster)
(Symbol)-[:PartOf {step}]->(Process)
```

## Quick Start

```bash
# Build the index (first time or after significant changes)
npx gitnexus analyze

# Start MCP server (add to .mcp.json for persistent access)
npx gitnexus mcp
```
<!-- gitnexus:end -->

## Project Overview
**project-zero** is a from-scratch, CPU-optimized LLM inference engine targeting BitNet b1.58
ternary weights and DeepSeek-V2 (MoE + MLA), with an OpenAI-compatible HTTP API.
Full architecture, directory map, terminology, and build/test/run details are canonical in
**`docs/ai/project-overview.md`** — keep that file (not this section) authoritative.
