---
mode: agent
description: Verify and review project-zero changes before completion.
---
Verify before claiming done (definition of done: `docs/ai/engineering-rules.md`):
1. `make release CC=gcc && make test CC=gcc && make debug CC=gcc` (repeat `CC=clang`).
2. Golden output for behavior changes: run `adaptive_ai_engine` on "What is the capital of
   France?" → output contains "Paris".
3. Kernel/perf changes: A/B tok/s vs a known-good commit on the same host — no regression.
4. Diff review: modularity, no hardcoding, memset-before-init, no new warnings, secrets check.
5. Report exactly what was verified and what remains UNVERIFIED.
