---
mode: agent
description: Implement a project-zero change in small, verified, documented sub-steps.
---
Implement per `docs/ai/engineering-rules.md`:
1. Make the smallest coherent edit; reuse existing patterns; no hardcoding; honor memory-safety
   rules (`memset` before partial init; overflow-checked, RAM-guarded allocations).
2. After each sub-step: run `review-changes.prompt.md` verification; update `docs/ai/**`
   (mistakes/decision-log/change-trace) and sync adapters via `update-project-memory.prompt.md`;
   propose an exact commit message (`docs/ai/commit-log-guidance.md`).
3. At completion: update canonical docs first, then adapters; report verified vs UNVERIFIED.
