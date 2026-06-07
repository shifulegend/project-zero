---
name: session-start
description: Run the mandatory project-zero session-start routine — read tool entrypoint + canonical docs/ai memory (overview, engineering-rules, mistakes, decision-log), then summarize rules, recent mistakes/decisions, open risks, and assumptions before planning or coding. Use at the very start of any work session.
---

# Session Start

1. Read `CLAUDE.md` and the relevant `.claude/rules/*.md`.
2. Read canonical memory: `docs/ai/project-overview.md`, `engineering-rules.md`, `mistakes.md`
   (newest first), `decision-log.md`, `tool-sync-policy.md`.
3. Summarize for this session: relevant rules · recent mistakes · recent decisions · open
   risks/UNKNOWNs · assumptions.
4. Check git state (`git status`, branch). State a short plan + the verification you will run.
5. Full procedure: `docs/ai/session-start-checklist.md`.
