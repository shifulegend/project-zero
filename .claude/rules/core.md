# Claude rule: core (engine / memory / safety)

Adapter for `docs/ai/engineering-rules.md` — read the canonical file for full text.

- Modularity: small single-responsibility TUs; `src/<area>/x.c` ⇄ `include/<area>/x.h`.
  Reuse `simd_dispatch`, GGUF reader, `tn_aligned_*`, `tn_size_mul*`, `tn_get_free_ram`.
- No hardcoding: shape/tokens/quant/RoPE/MoE from GGUF metadata (`hdr->arch` prefix).
- Memory safety: `memset` structs before partial init; overflow-checked size math; 64-byte
  aligned SIMD buffers; trap absurd allocations deterministically (not via malloc-NULL).
- Cross-platform: x86 (AVX2..VNNI) + ARM (NEON/dotprod) + scalar must all compile; portable
  prefetch `TN_PREFETCH_T1`; feature-gate with `TN_HAS_*`.
- No new compiler warnings (`-Wall -Wextra -Wpedantic`). Keep ASan/UBSan green.
- Public repo: never commit secrets.
- **Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task** —
  document it in `mistakes.md` too, but documenting is not a substitute for fixing. Only defer
  when the fix itself is a large architectural change; flag that to the user explicitly instead
  of silently fixing or silently skipping. See `engineering-rules.md` § Bug-fix policy.
