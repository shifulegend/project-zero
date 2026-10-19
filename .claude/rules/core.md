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
- No Claude/AI signature or attribution (commit trailers, code comments, etc.) unless the user
  explicitly asks for it that time — see `docs/ai/commit-log-guidance.md`.
- **Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task** —
  document it in `mistakes.md` too, but documenting is not a substitute for fixing. Only defer
  when the fix itself is a large architectural change; flag that to the user explicitly instead
  of silently fixing or silently skipping. See `engineering-rules.md` § Bug-fix policy.
- Attention heads run in parallel over the thread pool (`attn_heads_task`, threshold
  `--attn-parallel-min-ctx`): per-head code must only write its own `s->att` row and `s->xb`
  slice. Concurrency changes there need `make test-tsan` (test_forward is in TSAN_TESTS).
- Generic KV may be int8 (`RunState.kv_type == KV_TYPE_I8`: `key_cache` is NULL, use
  `key_cache_q`/`key_scale`). Any new reader/writer of the dense KV cache must handle both.
- Distributed (`--remote-layers`, Phase 28): layers can run in a worker process. Per-layer state
  that maps position → cache slot (sliding window) is part of the RPC contract; anything that
  restores/copies KV for all layers (prompt snapshots) must be bypassed when `dist_plan_active()`.
- Anything that calls `transformer_forward()` directly gets a default sliding window if none was
  set, but should `sw_init()` itself when it owns a sequence (generate_ex, perplexity, worker).
