---
mode: agent
description: Update canonical memory and synchronize all tool adapter files.
---
Per `docs/ai/tool-sync-policy.md` (do this proactively, no user prompt needed):
1. Update the right canonical file first: rule → `engineering-rules.md`; mistake →
   `mistakes.md` (template, newest first); decision → `decision-log.md`.
2. Sync only affected adapters. Scoped rules: edit only `.claude/rules/*.md`, then regenerate
   `.agents/rules/*` and `.github/instructions/*` with `tools/sync_ai_adapters.py` (CI fails on a
   stale copy). Entry files, by hand: `.github/copilot-instructions.md`, `CLAUDE.md`,
   `gemini/GEMINI.md` + `AGENTS.md`.
3. Record in `docs/ai/change-trace.md`. 4. Propose the commit checkpoint.
Canonical (`docs/ai/**`) wins on conflict; adapters must not hold knowledge absent there.
