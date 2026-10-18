# Claude rule: config (build + runtime configuration)

Adapter for the configurability rules in `docs/ai/engineering-rules.md`.

- Two build systems: `Makefile` (primary; per-file SIMD flags) and `CMakeLists.txt` (audit CI).
  Keep both in sync when adding source files or flags.
- Per-file SIMD flags live in the `Makefile` overrides section; 256-bit VNNI needs
  `-mavx512vnni -mavx512vl` (clang is strict). `debug` must keep `-march=native` and link the
  sanitizer runtime.
- Runtime config via CLI/env, not literals: `--model --tokenizer --prompt --max-tokens
  --temperature --top-p --seed --threads --simd --classifier --calibrate`.
- Model config/tokenizer/quant come from GGUF metadata, never hardcoded.
- New tunables: prefer flags/metadata/feature-detection over magic numbers; document defaults.
- Generation flags (`--top-k --min-p --repeat-penalty --presence-penalty --frequency-penalty
  --repeat-last-n --seed --system-prompt --stop --watermark*`) fill `GenerationParams` via
  `args_fill_gen_params()`; in `--server` mode they are the defaults published by `/v1/config`.
  A new knob must reach CLI + API + web UI (see `engineering-rules.md` § No hardcoding).
- Makefile tracks header deps (`-MMD -MP`); keep it.
