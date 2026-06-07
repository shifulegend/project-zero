---
mode: agent
description: Produce a tight, verifiable plan for a project-zero task.
---
Before coding, produce a short plan:
- Restate the goal and explicit assumptions; keep scope tight (no speculative refactors).
- Identify the smallest modular sub-steps and the files/modules touched.
- Note reuse of existing patterns (`simd_dispatch`, GGUF reader, `tn_*` helpers) instead of new
  abstractions; flag any hardcoding risk (must come from GGUF metadata / config instead).
- State the verification per step (build release/test/debug gcc+clang; golden output; A/B tok/s
  for kernels) and the commit checkpoints you will propose.
- If scope/architecture understanding changed, update `docs/ai/**` per `tool-sync-policy.md`.
