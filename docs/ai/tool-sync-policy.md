# Tool Sync Policy — project-zero

> Defines canonical vs adapter files and how they stay synchronized.
> Last updated: 2026-10-21.

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

## Third-party skill plugins (2026-10-21)
Two external skill packs are part of the workflow. They are *plugins* (self-updating), never
vendored copies — installing both a plugin and copies duplicates every skill.

| Pack | What it adds | Claude Code (repo-enabled) | Other tools (per user, optional) |
|------|--------------|----------------------------|----------------------------------|
| [obra/superpowers](https://github.com/obra/superpowers) | brainstorming, writing/executing plans, TDD, systematic debugging, verification-before-completion, code review, worktrees, subagent-driven development; SessionStart hook loads `using-superpowers` | `superpowers@claude-plugins-official` | Antigravity: `agy plugin install https://github.com/obra/superpowers`; Codex/Cursor: their plugin marketplaces |
| [mattpocock/skills](https://github.com/mattpocock/skills) | grill-me / grill-with-docs, tdd, diagnosing-bugs, codebase-design, to-spec / to-tickets / triage, research, handoff | `mattpocock-skills@claude-plugins-official` | Copilot: `copilot plugin marketplace add mattpocock/skills` + `copilot plugin install mattpocock-skills@mattpocock`; Gemini: `gemini skills install https://github.com/mattpocock/skills.git --path skills/engineering` (and `skills/productivity`); Codex: `codex plugin marketplace add mattpocock/skills` |

- Claude Code: `.claude/settings.json` declares the `claude-plugins-official` marketplace and
  enables both; Claude Code offers to install them when the repo is trusted. Manual:
  `claude plugin install <id> --scope project`.
- **Precedence:** these docs and the repo adapters override any plugin skill. No AI attribution
  in commits; no PR unless asked; benchmarks strictly sequential; autonomous/scheduled runs do
  not block on brainstorming or grilling questions; repo-local skills (`implement-task`,
  `review-and-verify`, `session-start`, `project-memory`) stay the definition of done.
- `/setup-matt-pocock-skills` (issue tracker, triage labels, glossary/ADR layout) edits
  `CLAUDE.md` and is interactive — run only when the user asks; its output then goes in
  `docs/ai/` first, adapters second.
