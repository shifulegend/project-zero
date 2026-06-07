# Antigravity scoped rules: config / build (adapter for docs/ai/engineering-rules.md)

- Two build systems: `Makefile` (primary; per-file SIMD flags) and `CMakeLists.txt` (audit CI);
  keep both in sync when adding sources/flags.
- Per-file SIMD flags in the Makefile overrides; 256-bit VNNI needs `-mavx512vnni -mavx512vl`
  (clang strict); `debug` keeps `-march=native` and links the sanitizer runtime.
- Runtime config via CLI/env, not literals (`--model --tokenizer --prompt --max-tokens
  --temperature --top-p --seed --threads --simd --classifier --calibrate`).
- Model config/tokenizer/quant from GGUF metadata; new tunables via flags/metadata.
