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
- Context sizing: `--ctx-size`, `--kv-ram-fraction`, `--kv-type f32|i8`, `--kv-offload <dir>`
  (`KVStrategyOptions`; with offload the cap budgets free disk, not RAM).
- Update checks: `--no-update-check --update-interval --update-channel --update-repo
  --update-api --check-update`, env `PZ_NO_UPDATE_CHECK`/`DO_NOT_TRACK`, off under `CI`; build
  default repo `PZ_UPDATE_REPO` (Makefile + CMake).
- Calibration (`~/.project-zero/calibration.bin`, v3) only overrides the auto SIMD tier when a
  forced tier is clearly faster; bump `TN_CALIB_VERSION` whenever selection logic changes.
