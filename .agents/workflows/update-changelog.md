# Workflow: update-changelog

Canonical rules: `docs/ai/changelog-guidance.md`; release steps: `docs/RELEASING.md`.
1. Only user-observable changes get an entry (flags, env, API, web UI, defaults, model/quant support,
   behaviour, perf, fixes, security, deprecations).
2. One imperative line under `## [Unreleased]`, right section (Changed, Deprecated, Removed, Added,
   Fixed, Security — in that order), interface named in backticks; same commit as the code.
3. `**Breaking:**` + what to do (first in section); deprecations name replacement + removal release;
   perf with numbers; security with impact/affected versions; no AI attribution.
4. `python3 tools/changelog.py check --require-unreleased` passes.
5. Release: curate → `## [X.Y.Z] - YYYY-MM-DD` + empty `[Unreleased]` + links → preview
   `tools/changelog.py notes vX.Y.Z` → commit → tag.
