# Claude rule: docs

Adapter for `docs/ai/` policy — read `docs/ai/tool-sync-policy.md` and `commit-log-guidance.md`.

- `docs/ai/**` is canonical; this and all adapter files are dynamic — update proactively, no prompt needed.
- Document non-obvious decisions/risks/workarounds in code comments **with a timestamp**.
- On a durable lesson: update `docs/ai/**` first → sync adapters → record in `change-trace.md`.
- Keep docs concise, factual, repository-specific; follow existing report style.
- Mark unknowns as TODO / UNKNOWN / ASSUMPTION; never invent commands/architecture.
