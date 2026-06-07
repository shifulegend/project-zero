---
applyTo: "docs/**,**/*.md"
---
# Copilot scoped rules: docs (adapter for docs/ai/tool-sync-policy.md)

- `docs/ai/**` is canonical; all instruction/prompt files are dynamic — update proactively.
- On a durable lesson: update `docs/ai/**` first → sync adapters (Claude/Copilot/Antigravity) →
  record in `docs/ai/change-trace.md`.
- Timestamp doc updates and non-obvious code comments (decisions/risks/workarounds).
- Concise, factual, repository-specific; follow existing report style. Mark unknowns as
  TODO / UNKNOWN / ASSUMPTION; never invent commands or architecture.
