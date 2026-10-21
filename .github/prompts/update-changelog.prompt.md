---
mode: agent
description: Add or curate CHANGELOG.md entries (Keep a Changelog + SemVer) for a user-facing change or a release.
---
Follow `docs/ai/changelog-guidance.md` (canonical) and, for releases, `docs/RELEASING.md`:
1. User-observable change? (flags, env vars, API routes/fields, web UI, defaults, model/quant support,
   behaviour, perf, fixes, security, deprecations). If not, no entry.
2. Add one imperative, self-describing line under `## [Unreleased]` in the right section
   (order: Changed, Deprecated, Removed, Added, Fixed, Security), naming the interface in backticks.
3. Breaking → `**Breaking:**` + what to do, listed first; deprecation → replacement + removal release;
   perf → measured numbers; security → impact + affected versions (+ advisory link).
4. Same commit as the code. No AI attribution. Run `python3 tools/changelog.py check --require-unreleased`.
5. Release: curate, rename to `## [X.Y.Z] - YYYY-MM-DD`, new empty `[Unreleased]`, update links,
   preview `tools/changelog.py notes vX.Y.Z`, commit, then tag.
