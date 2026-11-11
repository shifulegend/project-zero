# Antigravity scoped rules: tests

<!-- Generated from .claude/rules/tests.md by tools/sync_ai_adapters.py; edit that file and re-run the script (CI checks this copy). -->

Adapter for the verification section of `docs/ai/engineering-rules.md`.

- `make test` builds & runs **every** `tests/*.c` (find-derived), under ASan/UBSan, aborting on
  first failure. A full green run on gcc **and** clang is the bar.
- Tests own their setup: `memset` structs before `weights_alloc_pointers`/run-state fillers;
  zero any struct with feature-selecting fields (e.g. `VisionProjector.scale_factor`).
- Asset/CPU/env-dependent tests skip gracefully (see `test_vision_components`,
  `test_vision_e2e`, `test_simd`) rather than hard-failing when an optional asset is absent:
  CI and other hosts don't have the assets.
- Verify the full CI sequence locally (release+test+debug, gcc+clang) — a step only validates
  what it reaches; expect a cascade when unblocking the first failing test.
- Add a golden-output check for behavior changes (e.g. "capital of France" → "Paris").
- Any test that samples (temperature > 0) passes a fixed `--seed` (2026-10-22): a check that
  passes or fails by luck is broken; a re-run turning red green is a finding, not a fix.
- `make test` builds the library as the `debug` (ASan/UBSan) variant, so engine code is sanitized
  too (2026-10-21); it replaces a release `./pz` — run `make release` again before benchmarking.
  The build stamp also tracks compiler and host ISA, so CPU/compiler changes force a clean rebuild.
- Benchmark screenshots are real terminal sessions captured with `tools/screenshots/cli/capture.mjs`
  (banner visible, numbers from that run) — never images rendered from logs (2026-10-22).
