---
name: project-memory
description: Maintain project-zero's durable memory and keep all cross-tool instruction files synchronized. Use whenever a durable lesson, new convention, decision, mistake, or architecture/workflow change occurs (proactively, without being asked) to update docs/ai/** and the Claude/Copilot/Antigravity adapters.
---

# Project Memory Maintenance

When something durable changes:
1. Update the canonical file in `docs/ai/**`:
   - lesson/mistake → `mistakes.md` (use the template) and, if a rule, `engineering-rules.md`
   - decision → `decision-log.md`; notable change → `change-trace.md`
2. Synchronize the adapters that the change touches:
   - Claude: `CLAUDE.md`, `.claude/rules/*`
   - Copilot: `.github/copilot-instructions.md`, `.github/instructions/*`
   - Antigravity: `gemini/GEMINI.md`, `AGENTS.md`, `.agents/rules/*`
3. Record the sync in `change-trace.md`; propose the commit (see `commit-log-guidance.md`).
Canonical wins on conflict. Do this proactively, no user prompt required.
Full policy: `docs/ai/tool-sync-policy.md`.
