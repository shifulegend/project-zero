# Session-Start Checklist — project-zero

> Mandatory at the start of EVERY session, for every tool, before planning or coding.

0. Sync with GitHub first (2026-11-11). GitHub `master` is the source of truth; every working
   copy (a server, a laptop, a cloud session) is a temporary copy, and other sessions land work
   there. Run `tools/git_sync_check.sh` (Claude Code runs it automatically at session start). If
   `master` is behind and the tree is clean, `git pull --ff-only origin master`. Start new work
   on a short-lived branch from the updated `master`; `master` is protected and only changes
   through a PR with green CI.
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
- Before each commit: code ⇄ docs synchronized; user-facing change → `CHANGELOG.md` `[Unreleased]`
  line per `docs/ai/changelog-guidance.md`; propose an exact commit message.
- At task end: update `docs/ai/**` first, then Copilot/Claude/Antigravity adapters; record in
  `change-trace.md`.
- Verify (`make release/test/debug` for gcc+clang; golden output) and state what is
  verified vs UNVERIFIED.
- Before stopping: commit and push the branch (`git push -u origin <branch>`), so no work exists
  only on one machine. A PR is opened when the owner asks; once it merges, delete the branch.
