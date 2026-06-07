# Workflow: review-and-verify

Definition of done: `docs/ai/engineering-rules.md`.
1. `make release CC=gcc && make test CC=gcc && make debug CC=gcc` (repeat `CC=clang`).
2. Golden output for behavior changes: `adaptive_ai_engine` on "What is the capital of France?"
   → contains "Paris".
3. Kernel/perf changes: A/B tok/s vs a known-good commit on the same host — no regression.
4. Diff review: modularity, no hardcoding, memset-before-init, no new warnings, secrets check.
5. Report what is verified vs UNVERIFIED. Never claim done otherwise.
