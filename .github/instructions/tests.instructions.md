---
applyTo: "tests/**"
---
# Copilot scoped rules: tests (adapter for docs/ai/engineering-rules.md verification)

- `make test` builds & runs **every** `tests/*.c` under ASan/UBSan, aborting on first failure.
  The bar is a full green run on gcc **and** clang.
- Zero structs before partial init in test setup (`memset` before `weights_alloc_pointers`;
  zero feature-selecting fields like `VisionProjector.scale_factor`).
- Asset/CPU/env-dependent tests must skip gracefully, never hard-fail on a missing optional asset.
- Verify the full CI sequence locally (release+test+debug, gcc+clang); a step only validates what
  it reaches — expect a cascade when unblocking the first failing test.
- Add a golden-output check for behavior changes (e.g. "capital of France" → "Paris").
- Any test that samples (temperature > 0) passes a fixed `--seed` (2026-10-22): a check that
  passes or fails by luck is broken; a re-run turning red green is a finding, not a fix.
- `make test` builds the library as the `debug` (ASan/UBSan) variant, so engine code is sanitized
  too (2026-10-21); it replaces a release `./pz` — run `make release` again before benchmarking.
  The build stamp also tracks compiler and host ISA, so CPU/compiler changes force a clean rebuild.
