---
applyTo: "Makefile,CMakeLists.txt,**/*.cmake,**/*.mk,.github/workflows/**"
---
# Copilot scoped rules: config / build (adapter for docs/ai/engineering-rules.md)

- Two build systems: `Makefile` (primary; per-file SIMD flags) and `CMakeLists.txt` (audit CI).
  Keep both in sync when adding sources or flags.
- Per-file SIMD flags in the Makefile overrides section; 256-bit VNNI needs
  `-mavx512vnni -mavx512vl` (clang strict). `debug` keeps `-march=native` and links the
  sanitizer runtime.
- Runtime config via CLI/env, not literals: `--model --tokenizer --prompt --max-tokens
  --temperature --top-p --seed --threads --simd --classifier --calibrate`.
- Model config/tokenizer/quant from GGUF metadata; new tunables via flags/metadata, not magic
  numbers (document defaults).
- Generation flags (`--top-k --min-p --repeat-penalty --presence-penalty --frequency-penalty
  --repeat-last-n --seed --system-prompt --stop --watermark*`) fill `GenerationParams` via
  `args_fill_gen_params()`; in `--server` mode they are the defaults published by `/v1/config`.
  A new knob must reach CLI + API + web UI (see `engineering-rules.md` § No hardcoding).
- Makefile tracks header deps (`-MMD -MP`); keep it.
- Context sizing: `--ctx-size`, `--kv-ram-fraction`, `--kv-type f32|i8|i4`, `--kv-offload <dir>`
  (`KVStrategyOptions`; with offload the cap budgets free disk, not RAM).
- Update checks: `--no-update-check --update-interval --update-channel --update-repo
  --update-api --check-update`, env `PZ_NO_UPDATE_CHECK`/`DO_NOT_TRACK`, off under `CI`; build
  default repo `PZ_UPDATE_REPO` (Makefile + CMake).
- Phase 30 search: `--mcts N --mcts-expand --mcts-depth --mcts-step-tokens --mcts-c --mcts-temperature
  --mcts-delim --mcts-max`, judge `--prm <gguf> --prm-format --prm-good --prm-bad` (no hardcoded judge).
- DRAM bandwidth is cached per host (`core/bw_cache`: `$PZ_CACHE_DIR|$XDG_CACHE_HOME/project-zero|~/.cache/project-zero/membw`, 30 days); `--calibrate` or `PZ_BW_REPROBE=1` re-measures.
- `--chat-template-file <f>` replaces a GGUF's `tokenizer.chat_template` (CLI + `--server`);
  `docs/templates/bitnet-b1.58.jinja` fixes the official BitNet GGUF's broken template.
- Calibration (`~/.project-zero/calibration.bin`, v3) only overrides the auto SIMD tier when a
  forced tier is clearly faster; bump `TN_CALIB_VERSION` whenever selection logic changes.
- GGUF conventions follow llama.cpp: RoPE layout from the architecture (`--rope-layout` to
  override), BPE pre-tokenizer from `tokenizer.ggml.pre`. `--raw-prompt` skips the chat template
  (plain completion, for benchmarks vs llama-completion); `--perplexity <file>` (+`--ppl-ctx`,
  `--ppl-chunks`) reproduces llama-perplexity for accuracy checks.
