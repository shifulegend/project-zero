# Workflow: update-project-memory

Per `docs/ai/tool-sync-policy.md` (proactive, no user prompt needed):
1. Update canonical first: rule → `engineering-rules.md`; mistake → `mistakes.md` (template,
   newest first); decision → `decision-log.md`.
2. Sync affected adapters: Antigravity (`gemini/GEMINI.md` ⇄ `AGENTS.md`, `.agents/rules/*`),
   Claude (`CLAUDE.md`, `.claude/rules/*`), Copilot (`.github/copilot-instructions.md`,
   `.github/instructions/*`).
3. Record in `docs/ai/change-trace.md`. 4. Propose the commit checkpoint.
Canonical (`docs/ai/**`) wins on conflict.
