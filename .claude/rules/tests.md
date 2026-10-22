# Claude rule: tests

Adapter for the verification section of `docs/ai/engineering-rules.md`.

- `make test` builds & runs **every** `tests/*.c` (find-derived), under ASan/UBSan, aborting on
  first failure. A full green run on gcc **and** clang is the bar.
- Tests own their setup: `memset` structs before `weights_alloc_pointers`/run-state fillers;
  zero any struct with feature-selecting fields (e.g. `VisionProjector.scale_factor`).
- Make asset/CPU/env-dependent tests skip gracefully (see `test_vision_components`,
  `test_vision_e2e`, `test_simd`), never hard-fail when an optional asset is absent.
- Verify the full CI sequence locally (release+test+debug, gcc+clang) — a step only validates
  what it reaches; expect a cascade when unblocking the first failing test.
- Add a golden-output check for behavior changes (e.g. "capital of France" → "Paris").
- Any test that samples (temperature > 0) passes a fixed `--seed` (2026-10-22): a check that
  passes or fails by luck is broken; a re-run turning red green is a finding, not a fix.
- `make test` builds the library as the `debug` (ASan/UBSan) variant, so engine code is sanitized
  too (2026-10-21); it replaces a release `./pz` — run `make release` again before benchmarking.
  The build stamp also tracks compiler and host ISA, so CPU/compiler changes force a clean rebuild.
