# Session-Start Checklist — project-zero

> Mandatory at the start of EVERY session, for every tool, before planning or coding.

1. Read your tool's entrypoint + relevant scoped rules:
   - Claude Code: `CLAUDE.md` + `.claude/rules/*.md`
   - GitHub Copilot: `.github/copilot-instructions.md` + relevant `.github/instructions/*.instructions.md`
   - Antigravity: `gemini/GEMINI.md`, `AGENTS.md` + relevant `.agents/rules/*.md`
2. Read the canonical shared docs:
   - `docs/ai/project-overview.md`
   - `docs/ai/engineering-rules.md`
   - `docs/ai/mistakes.md`  (newest entries first)
   - `docs/ai/decision-log.md`
   - `docs/ai/tool-sync-policy.md`
3. Summarize in-session: relevant repo rules · recent mistakes · recent decisions · open
   risks/UNKNOWNs · your assumptions for this task.
4. State a short plan and the verification you will run (build/test/lint/golden output).
5. Confirm git state (`git status`, current branch) before changing anything.

## During / end of session (auto, no user prompt needed)
- New durable lesson → add to `mistakes.md` and/or `engineering-rules.md`, then sync adapters.
- New decision → `decision-log.md`. Notable change → `change-trace.md`.
- Before each commit: code ⇄ docs synchronized; propose an exact commit message.
- At task end: update `docs/ai/**` first, then Copilot/Claude/Antigravity adapters; record in
  `change-trace.md`.
- Verify (`make release/test/debug` for gcc+clang; golden output) and state what is
  verified vs UNVERIFIED.
