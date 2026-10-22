# Antigravity scoped rules: core engine (adapter for docs/ai/engineering-rules.md)

- Small single-responsibility TUs; `src/<area>/x.c` ⇄ `include/<area>/x.h`. Reuse
  `simd_dispatch`, GGUF reader, `tn_aligned_*`, `tn_size_mul*`, `tn_get_free_ram`.
- No hardcoding: shape/tokens/quant/RoPE/MoE from GGUF metadata (`hdr->arch` prefix).
- Memory safety: `memset` structs before partial init; overflow-checked size math; 64-byte
  aligned SIMD buffers; trap absurd allocations deterministically (not via malloc-NULL).
- Cross-platform: AVX2..VNNI / NEON+dotprod / scalar all compile; `TN_HAS_*` gates; portable
  `TN_PREFETCH_T1`. No new warnings; keep ASan/UBSan green.
- Any bug found gets fixed in the same pass, even if pre-existing/unrelated to the task — log it
  in `mistakes.md` too, but that's not a substitute for fixing. Only defer for genuinely
  large architectural fixes, and flag those to the user explicitly. See `engineering-rules.md`
  § Bug-fix policy.
- Attention heads run in parallel over the thread pool (`attn_heads_task`, threshold
  `--attn-parallel-min-ctx`): per-head code must only write its own `s->att` row and `s->xb`
  slice. Concurrency changes there need `make test-tsan` (test_forward is in TSAN_TESTS).
- Startup counts as performance: no O(n²) init (tokenizer lookups via the sorted index), per-host
  probes cached (`core/bw_cache`), large zeroed buffers lazily committed (`tn_aligned_calloc`
  uses `MADV_DONTNEED` on Linux — no full memset). Benchmarks report wall time + RSS too.
- GGUF stop tokens: `tokenizer_collect_eog()` (declared EOS/EOT/EOM + end-of-turn control
  tokens). Generation loops must check all of `tok->eos_list`, not just `eos_token_id`.
- Prompt-injection defence (2026-10-22): untrusted text -- chat message content, tool output,
  retrieved memories -- goes through `tokenizer_neutralize_control()` before it enters a prompt
  (CONTROL specials become placeholders that encode as plain text; USER_DEFINED like `<think>`
  stay tokens). Any new path that puts user/tool text into a prompt must do the same;
  `tests/test_prompt_injection.c` (CI golden job) guards it.
- LoRA: every attention/FFN path that runs a projection must apply `s->active_lora` (single and
  batched); Q width is `q_dim` (n_heads*head_dim) — not `dim` — on Qwen3.
- Generic KV may be quantized (`RunState.kv_type` I8 or I4: `key_cache` is NULL; row r =
  `KV_SCALE_IDX` lives at `key_cache_q + r*kv_row_bytes`, scales at `key_scale + r*kv_row_scales`;
  use `kvq_store/kvq_dot_row/kvq_axpy_row`). Any new reader/writer of the dense KV cache must handle all.
- Distributed (`--remote-layers`, Phase 28): layers can run in a worker process. Per-layer state
  that maps position → cache slot (sliding window) is part of the RPC contract; anything that
  restores/copies KV for all layers (prompt snapshots) must be bypassed when `dist_plan_active()`.
- Anything that calls `transformer_forward()` directly gets a default sliding window if none was
  set, but should `sw_init()` itself when it owns a sequence (generate_ex, perplexity, worker).
