# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-06-07.

### 2026-06-07 — Document branch-hygiene convention
- What: Added a "Version control & branch hygiene" section to `engineering-rules.md` (delete
  merged branches; enable auto-delete-head-branches; avoid flag/placeholder branches; don't
  commit artifacts/models/logs).
- Why: post-merge cleanup surfaced redundant branches; this sandbox's git proxy blocks ref
  deletion, so the convention + the repo auto-delete setting prevent future accumulation.
- Areas: `docs/ai/engineering-rules.md`. Canonical-only (adapters stay lean per tool-sync-policy).

### 2026-06-07 — Cross-tool AI development system
- What: Added `docs/ai/**` canonical docs + Claude/Copilot/Antigravity adapters.
- Why: one source of truth; continuity across Claude Code, GitHub Copilot, Antigravity.
- Areas: `docs/ai/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.github/`, `AGENTS.md`,
  `gemini/GEMINI.md`, `.agents/`.
- Commit/PR: (this change) — see commit checkpoints in PR to `master`.

### 2026-06-07 — Green CI + regression verification (PR #6, merged `cb9fa52`)
- What: Fixed the CI cascade and verified no regression across SmolLM2/BitNet/DeepSeek.
- Why: CI had never run to completion; ensure merge-safety and no regression.
- Areas: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`,
  `tests/test_vision_components.c`, `Makefile`, `src/core/run_state.c`,
  `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
- Result: all 7 CI checks green on PR #6 and on `master`; secrets scan clean (215 commits).
