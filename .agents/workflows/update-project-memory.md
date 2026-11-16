# Workflow: update-project-memory

Per `docs/ai/tool-sync-policy.md` (proactive, no user prompt needed):
1. Update canonical first: rule → `engineering-rules.md`; mistake → `mistakes.md` (template,
   newest first); decision → `decision-log.md`.
2. Sync affected adapters. Scoped rules: edit only `.claude/rules/*.md`, then regenerate
   `.agents/rules/*` and `.github/instructions/*` with `tools/sync_ai_adapters.py` (CI fails on a
   stale copy). Entry files, by hand: `gemini/GEMINI.md` ⇄ `AGENTS.md`, `CLAUDE.md`,
   `.github/copilot-instructions.md`.
3. Record in `docs/ai/change-trace.md`. 4. Propose the commit checkpoint.
Canonical (`docs/ai/**`) wins on conflict.
