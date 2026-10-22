# Antigravity scoped rules: tests (adapter for docs/ai/engineering-rules.md verification)

- `make test` builds & runs **every** `tests/*.c` under ASan/UBSan, aborting on first failure;
  the bar is a full green run on gcc **and** clang.
- Zero structs before partial init in test setup; zero feature-selecting fields.
- Asset/CPU/env-dependent tests skip gracefully; never hard-fail on a missing optional asset.
- Verify the full CI sequence locally (release+test+debug, gcc+clang); expect a cascade when
  unblocking the first failing test. Add golden-output checks for behavior changes.
- `make test` builds the library as the `debug` (ASan/UBSan) variant, so engine code is sanitized
  too (2026-10-21); it replaces a release `./pz` — run `make release` again before benchmarking.
  The build stamp also tracks compiler and host ISA, so CPU/compiler changes force a clean rebuild.
- Any test that samples (temperature > 0) passes a fixed `--seed` (2026-10-22): a check that
  passes or fails by luck is broken; a re-run turning red green is a finding, not a fix.
