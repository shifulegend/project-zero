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
