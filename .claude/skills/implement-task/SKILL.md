---
name: implement-task
description: Implement a project-zero change end-to-end with discipline — plan, state assumptions, make small modular edits reusing existing patterns, verify, and update project memory + commit in small checkpoints. Use for any non-trivial code change.
---

# Implement Task

1. Plan: restate the goal, assumptions, scope (keep tight; no speculative refactors).
2. Recon: reuse existing patterns (`simd_dispatch`, GGUF reader, `tn_*` helpers) before new code.
3. Implement in the smallest coherent sub-steps; honor `.claude/rules/*` (modularity, no
   hardcoding, memset-before-init, configurability).
4. Verify each sub-step (see `review-and-verify`); fix the whole cascade if `make test` reveals more.
5. Update `docs/ai/**` (mistakes/decision-log/change-trace) and sync adapters as you go.
6. Propose an exact commit message per sub-step (see `docs/ai/commit-log-guidance.md`).
