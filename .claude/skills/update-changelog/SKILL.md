---
name: update-changelog
description: Add or curate project-zero CHANGELOG.md entries (Keep a Changelog + SemVer) — use in the same commit as any user-facing change (flag, env var, API route/field, web UI control, default, model/quant support, behaviour, perf, fix, security, deprecation), and when preparing a release.
---

# Update the changelog

Canonical rules: `docs/ai/changelog-guidance.md` (read it first). Release steps: `docs/RELEASING.md`.

## Per change (same commit)
1. Decide if a user can observe it (guidance §2). Internal refactors, tests, CI, benchmarks, AI/process
   docs → no entry.
2. Pick the section under `## [Unreleased]`: Changed / Deprecated / Removed / Added / Fixed / Security
   (keep that order; create the heading if missing).
3. Write one line, imperative, self-describing, naming the interface in backticks:
   `- Add \`--kv-type i4\` for int4 KV cache storage.`
   - Breaking → `**Breaking:**` prefix + what users must do; put it first in the section.
   - Deprecation → replacement + the release that removes it.
   - Performance → measured number + model/threads/host.
   - Security → impact + affected versions; advisory link if any; no details before the fix ships.
   - Link the PR/issue in parentheses if one exists. No author names, no AI attribution.
4. If the change revises something already under `[Unreleased]`, edit that line instead of adding one.
5. `python3 tools/changelog.py check --require-unreleased` must pass.

## Preparing a release
1. Choose the version by SemVer from the sections present (guidance §5).
2. Curate `[Unreleased]` as a user would read it: merge duplicates, drop noise, breaking first,
   verify every named flag/field still exists.
3. Rename to `## [X.Y.Z] - YYYY-MM-DD`, add an empty `## [Unreleased]` above, update the compare links.
4. `python3 tools/changelog.py check` and preview `python3 tools/changelog.py notes vX.Y.Z`; commit,
   then tag `vX.Y.Z` (the release workflow publishes that section as the release notes).
