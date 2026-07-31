# Tool Sync Policy — project-zero

> Defines canonical vs adapter files and how they stay synchronized.
> Last updated: 2026-07-15.

## Canonical (source of truth)
`docs/ai/**` — overview, engineering-rules, mistakes, decision-log, change-trace,
session-start-checklist, commit-log-guidance, this policy. Durable knowledge lives ONLY here.

## Adapter files (concise; summarize + link to canonical)
| Tool | Entry | Scoped rules | Workflows / skills |
|------|-------|--------------|--------------------|
| Claude Code | `CLAUDE.md` | `.claude/rules/{core,docs,tests,config,api}.md` | `.claude/skills/*/SKILL.md` |
| GitHub Copilot | `.github/copilot-instructions.md` | `.github/instructions/{core,docs,tests,config,api}.instructions.md` | `.github/prompts/*.prompt.md` |
| Antigravity | `gemini/GEMINI.md` + `AGENTS.md` | `.agents/rules/{core,docs,tests,config,api}.md` | `.agents/workflows/*.md` |

`AGENTS.md` is a portability mirror; keep it synchronized with `gemini/GEMINI.md` on durable
repo-wide rules. (Both currently also carry the GitNexus MCP block — preserve it.)

## Canonicality rule
- On a conflict, update `docs/ai/**` first, then the adapters. Adapters must never hold durable
  knowledge that isn't in the canonical docs.

## Dynamic-file rule (no user prompt required)
All canonical docs AND all adapter files are living documents. Update them proactively whenever:
a new durable lesson appears, a convention is adopted, a repeated mistake is corrected,
architecture/workflow changes, or docs diverge from reality.

## Sync procedure when a durable lesson is found
1. Update `docs/ai/**` (the right file: rules / mistakes / decision-log).
2. Update the affected adapter files (only the scopes that changed).
3. Record it in `docs/ai/change-trace.md`.
4. Propose the commit checkpoint (see `commit-log-guidance.md`).

## When to add OPTIONAL scoped files (only with repo evidence)
Add `api`/`infra`/`database`/etc. scoped rule/instruction files only when a recurring scoped
rule actually exists. This repo's natural scopes today: `core` (engine/memory/safety), `docs`,
`tests`, `config` (build + GGUF/runtime config). A `simd/math` scope is a candidate future
addition — add only when justified.

**`api` scope — added Phase 22.1** (`src/api/` gained CORS/auth/metrics/OpenAPI/cancel/
concurrency-rearchitecture surface). Adapters added: `.claude/rules/api.md`,
`.github/instructions/api.instructions.md`, `.agents/rules/api.md`.

## Drift check
At session start and before task completion, confirm adapters still match the canonical docs;
if not, reconcile (canonical wins) and note it in `change-trace.md`.
