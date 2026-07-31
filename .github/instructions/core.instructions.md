---
applyTo: "src/**,include/**"
---
# Copilot scoped rules: core engine (adapter for docs/ai/engineering-rules.md)

- Small single-responsibility TUs; `src/<area>/x.c` ⇄ `include/<area>/x.h`. Reuse
  `simd_dispatch`, GGUF reader, `tn_aligned_*`, `tn_size_mul*`, `tn_get_free_ram`.
- No hardcoding: shape/tokens/quant/RoPE/MoE from GGUF metadata (`hdr->arch` prefix).
- Memory safety: `memset` structs before partial init; overflow-checked size math; 64-byte
  aligned SIMD buffers; trap absurd allocations deterministically (not via malloc-NULL).
- Cross-platform: AVX2..VNNI / NEON+dotprod / scalar all compile; `TN_HAS_*` gates; portable
  `TN_PREFETCH_T1`.
- No new `-Wall -Wextra -Wpedantic` warnings; keep ASan/UBSan green.
- Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task — log it
  in `mistakes.md` too, but that's not a substitute for fixing. Only defer for genuinely
  large architectural fixes, and flag those to the user explicitly. See `engineering-rules.md`
  § Bug-fix policy.
