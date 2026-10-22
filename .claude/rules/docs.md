# Claude rule: docs

Adapter for `docs/ai/` policy — read `docs/ai/tool-sync-policy.md` and `commit-log-guidance.md`.

- `docs/ai/**` is canonical; this and all adapter files are dynamic — update proactively, no prompt needed.
- Document non-obvious decisions/risks/workarounds in code comments **with a timestamp**.
- On a durable lesson: update `docs/ai/**` first → sync adapters → record in `change-trace.md`.
- Keep docs concise, factual, repository-specific; follow existing report style.
- Mark unknowns as TODO / UNKNOWN / ASSUMPTION; never invent commands/architecture.
- **Changelog** (`docs/ai/changelog-guidance.md`): every user-facing change adds one imperative line
  under `## [Unreleased]` in `CHANGELOG.md` in the same commit (Keep a Changelog sections, order
  Changed/Deprecated/Removed/Added/Fixed/Security; `**Breaking:**` prefix + what to do; deprecations
  name replacement + removal release; perf claims carry numbers). `tools/changelog.py check` must pass.
- **Releases** (`docs/RELEASING.md` § Maintaining published releases): published releases are
  immutable (no asset replacement, tag moves or note rewrites — fixes ship as a new version);
  triage every landed fix against the latest release; security fixes via a private advisory.
