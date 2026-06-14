# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-06-14.

### 2026-06-14 — README accuracy pass + repo best-practices + docs reorg
- What: (1) Corrected README intro to match canonical scope (BitNet + DeepSeek-V2-Lite
  GGUF + vision/agentic/RAG), kept "written in C", reframed Python as temporary
  dev/test tooling (zero-Python goal), added LLM-agnostic goal. (2) Reconciled the
  Phase 21 HTTP API claim to 🔄 partial/experimental across README, ROADMAP, and
  project-overview (it is real and wired but serial/loopback-only/untested-in-CI).
  (3) Added community-health files: `.github/CODEOWNERS`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`, `.editorconfig`, `CITATION.cff`. (4) Moved 27
  archival/design/report `.md` files out of the repo root into
  `docs/{architecture,phases,reports,weight-loading}/` and `docs/`, leaving 8 entry-point
  docs at root; rewrote all inbound markdown links path-aware and fixed 4 dangling links
  (verified 0 dangling repo-wide).
- Why: README/roadmap/overview contradicted each other and the tree; root had 35 `.md`
  files hurting discoverability; repo was missing standard GitHub best-practice files.
- Areas: `README.md`, `.github/ROADMAP.md`, `docs/ai/project-overview.md`, `.editorconfig`,
  `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/config.yml`,
  `CITATION.cff`, and `docs/{architecture,phases,reports,weight-loading}/**`.
- Branch: `claude/readme-accuracy-review-y9jk7u`.

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
