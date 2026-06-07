# Antigravity scoped rules: docs (adapter for docs/ai/tool-sync-policy.md)

- `docs/ai/**` is canonical; all instruction/rule/workflow files are dynamic — update proactively.
- On a durable lesson: update `docs/ai/**` first → sync adapters (Claude/Copilot/Antigravity,
  incl. keeping `gemini/GEMINI.md` ⇄ `AGENTS.md`) → record in `docs/ai/change-trace.md`.
- Timestamp doc updates and non-obvious code comments (decisions/risks/workarounds).
- Concise, factual, repository-specific. Mark unknowns TODO / UNKNOWN / ASSUMPTION.
