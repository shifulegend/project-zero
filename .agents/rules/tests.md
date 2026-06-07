# Antigravity scoped rules: tests (adapter for docs/ai/engineering-rules.md verification)

- `make test` builds & runs **every** `tests/*.c` under ASan/UBSan, aborting on first failure;
  the bar is a full green run on gcc **and** clang.
- Zero structs before partial init in test setup; zero feature-selecting fields.
- Asset/CPU/env-dependent tests skip gracefully; never hard-fail on a missing optional asset.
- Verify the full CI sequence locally (release+test+debug, gcc+clang); expect a cascade when
  unblocking the first failing test. Add golden-output checks for behavior changes.
