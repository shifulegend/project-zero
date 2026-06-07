---
name: review-and-verify
description: Verify a project-zero change before claiming completion — build release/test/debug for gcc and clang, run golden-output and (for kernels) A/B tok/s checks, review the diff, and report what is verified vs unverified. Use before committing or marking a task done.
---

# Review & Verify

1. Build + test the full sequence for both compilers:
   `make release CC=gcc && make test CC=gcc && make debug CC=gcc` (repeat `CC=clang`).
2. Golden output: run a known prompt (e.g. capital-of-France → "Paris") on a representative model.
3. Kernel/perf changes: A/B tok/s vs a known-good commit on the same host.
4. Review the diff for modularity, no-hardcoding, memset-before-partial-init, and no new warnings.
5. Report: what was verified, what remains UNVERIFIED, and the exact commit message.
Bar: `make test` runs every `tests/*.c` and aborts on first failure — full green required.
Details: `docs/ai/engineering-rules.md`, `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
