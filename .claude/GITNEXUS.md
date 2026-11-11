# GitNexus MCP — project-zero (reference, read when you need the knowledge graph)

> Moved here from `CLAUDE.md` / `AGENTS.md` (2026-11-11) so the root files stay short. Read this
> when you explore architecture, trace a bug, size a blast radius or refactor through GitNexus.
> Companion references in this folder: `INDEX.md` (module index), `CONTEXT.md` (architecture +
> known issues), `clusters/CLUSTERS.md`, `processes/PROCESSES.md`, `BENCHMARK_SUMMARY.md`.

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
