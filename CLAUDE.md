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

**project-zero** is a from-scratch, CPU-optimized LLM inference engine targeting BitNet b1.58 ternary weights and DeepSeek-V2 architecture.

### Key Modules

| Path | Purpose |
|------|---------|
| `src/core/` | Weight loading, quantization, MoE config, debug utilities |
| `src/math/` | SIMD matmul kernels (F16, Q4K, Q5_0, Q5_1, Q5K, VNNI) |
| `src/transformer/` | Attention (MLA), MoE routing, generation loop |
| `src/tokenizer/` | GGUF tokenizer, chat template, BPE |
| `src/api/` | HTTP server, SSE streaming, JSON parsing, chat compilation |
| `src/cli/` | Argument parsing, main entry point |
| `include/` | Public headers mirroring `src/` structure |
| `tests/` | Unit tests for math, MoE, SIMD, API, tokenizer |
| `tools/` | Conversion utilities |

### Build

```bash
cmake -B build && cmake --build build -j$(nproc)
# or
make
```
