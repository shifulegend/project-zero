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
