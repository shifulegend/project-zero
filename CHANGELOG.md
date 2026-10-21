# Changelog

## [Unreleased] — 2026-03-17 (Phase 16-S: SIMD Performance Kernels)

### Added (Phase 16-S: Multi-tier SIMD Dispatch & int8 Quantization)

- `src/math/ternary_matmul_packed_vnni.c` — AVX-512 VNNI kernel using
  `_mm512_dpbusd_epi32` for 64 int8 MACs/cycle. Targets Ice Lake Server,
  Tiger Lake, Zen 4, Sapphire Rapids. Uses the `w_enc = w + 1` bias trick:
  `dot(w_enc, q_x) - sum(q_x)` to stay in unsigned×signed domain.
- `src/math/ternary_matmul_packed_avx_vnni.c` — AVX-VNNI 256-bit kernel using
  `_mm256_dpbusd_epi32` for 32 int8 MACs/cycle. Targets Alder Lake (12th Gen),
  Raptor Lake (13th Gen), Zen 3 — i.e. CPUs that have 256-bit VNNI but no
  AVX-512.
- `src/math/ternary_matmul_packed_dotprod.c` — ARM NEON+dotprod kernel using
  `vdotq_s32` for 16 int8 MACs/cycle. Targets Apple M1/M2/M3/M4, Cortex-A75+,
  Snapdragon 8 Gen 1+. Uses signed×signed dot product directly (no bias trick
  needed).
- `src/math/quantize_i8.c` — Runtime float32→int8 quantization of activation
  vectors (`quantize_row_to_i8`), with scalar, AVX2, and AVX-512 variants.
  Activations are quantized at inference time and never stored to disk.
- `src/math/cpu_features.c` — Runtime CPUID/`getauxval` detection for all
  SIMD tiers: AVX2, AVX-512F, AVX-512 VNNI, AVX-VNNI, ARM NEON, ARM dotprod,
  SVE2, BF16. Result is cached after first call.
- `tests/test_simd_vnni.c` — Correctness and dispatch tests for Phase 16-S:
  quantize, sum_i8, VNNI matmul vs float baselines, CPU detection, dispatch
  selection, KV strategy threshold fix (K-3).

### Changed (Phase 16-S)

- `src/math/simd_dispatch.c` — Expanded dispatch table with 7-tier priority:
  AVX-512 VNNI > AVX-VNNI > AVX-512F > AVX2 > ARM dotprod > ARM NEON > Scalar.
  Dispatch is always runtime-selected via `tn_cpu_features_detect()`; compile-
  time `TN_HAS_*` guards only prevent emitting ISA instructions the compiler
  doesn't know about.
- `src/kv_cache/kv_strategy.c` — Fixed K-3 threshold: 16 GB systems (≥8 GB
  free) now correctly select `KV_QUANT_I8` instead of falling to `KV_SLIDING_I4`.
- `src/memory/mapped_file.c` — Added `MADV_HUGEPAGE` hint on Linux for the
  weight mmap region, reducing TLB pressure for large models.

### Model Conversion: No Changes Required

**Phase 16-S does not change the on-disk weight format.** The packed binary
format (`.tnry`) established in Phase 10 is identical:

```
[TN_MAGIC | TN_VERSION | config | packed_weights[]]
  └─ 2 bits per weight, 4 weights per byte, LSB-first
  └─ encoding: w_enc = w + 1  →  {-1,0,1} → {0,1,2}
```

The `w_enc = w + 1` encoding was deliberately chosen in Phase 10 to be
VNNI-compatible: it keeps `w_enc` in `[0, 2]` which is unsigned-safe for
`_mm512_dpbusd_epi32`. The new VNNI/dotprod kernels consume `tn_u8 *packed_w`
directly without any format change. **`convert_hf_bitnet.py`, `convert_hf.py`,
`pack_ternary.py`, and `convert_tokenizer.py` are all unmodified.**

---

## [Unreleased] — 2026-03-13 (Phase 10 Complete)

### Added (Phase 10: Weight Packing & Conversion Tools)

- `include/core/unpack.h` — 2-bit ternary weight unpacking API with inline
  `unpack_ternary()`, `pack_ternary()`, and `packed_bytes()` helpers.
- `src/core/unpack.c` — Scalar reference unpacker (`unpack_ternary_block`).
- `src/core/unpack_avx2.c` — AVX2 SIMD unpacker processing 32 weights per
  iteration via byte-replication + shift-extract + blend selection.
- `include/math/ternary_matmul_packed.h` — Packed ternary matmul API with
  per-matrix and per-group scale support.
- `src/math/ternary_matmul_packed.c` — Scalar reference packed matmul.
- `src/math/ternary_matmul_packed_avx2.c` — AVX2 fused unpack+matmul kernel.
- `tests/test_packed_weights.c` — 13 test functions (1469 assertions) covering
  pack/unpack round-trip, AVX2 correctness, packed matmul per-matrix/per-group
  scales, edge cases, and large dimensions.
- `tools/pack_ternary.py` — Python ternary weight packer/unpacker with binary
  format writer (magic, version, config, scale_mode, weights+scales).
- `tools/convert_hf_bitnet.py` — Converts HuggingFace BitNet safetensors models
  to packed binary format with ternary validation and scale computation.
- `tools/convert_tokenizer.py` — Converts HuggingFace tokenizer.json or
  SentencePiece .model files to binary tokenizer format.

### Fixed (Phase 10 Deep Audit — 3 additional bugs)

- **PRE10-BUG-004 (HIGH):** Value cache indexing in `attention.c` Step 5c used
  raw loop index `t` instead of mapped physical position, causing silent output
  corruption after the sliding window wraps. Now uses `sw_map_position()`
  consistently with Step 5a's key cache lookup.

- **PRE10-BUG-005 (MEDIUM):** RNG in `reasoning_generate.c` was re-seeded every
  iteration with `step * 1337`, making stochastic sampling deterministic. Moved
  RNG initialization before the loop with proper time-based seed.

- **PRE10-BUG-006 (HIGH):** Memory leak in `weights_alloc_pointers()` when a
  middle allocation fails. Interleaved alloc+check via TN_CHECK leaked prior
  allocations on failure. Restructured to alloc-all-then-check with
  `weights_free_pointers()` cleanup.

### Fixed (Phase 10 Pre-Audit — 3 new bugs found and verified)

- **PRE10-BUG-001 (HIGH):** File descriptor and flock leak in `mapped_file_open()` when
  opening a zero-size file. The `TN_CHECK(file_size > 0)` macro performed a bare return
  without closing the already-open fd or releasing the shared flock. Also fixed a secondary
  issue where a negative `off_t` cast to `size_t` would produce a huge positive value
  passing the `> 0` check. (`src/memory/mapped_file.c:36`)

- **PRE10-BUG-002 (HIGH):** FILE handle leak in `tokenizer_load()` when vocab_size or
  max_token_len validation fails. The `TN_CHECK` macro on lines 87-88 returned without
  calling `fclose(fp)`, leaking the FILE handle on invalid input.
  (`src/tokenizer/tokenizer_load.c:87-88`)

- **PRE10-BUG-003 (MEDIUM):** NULL pointer dereference in `tokenizer_decode()` when
  `t->vocab[token]` is NULL and `prev_token == 1` (BOS). The hex-byte check had a NULL
  guard but the BOS-strip path at line 42 accessed `piece[0]` unconditionally, causing
  SIGSEGV. (`src/tokenizer/tokenizer_decode.c:42`)

### Fixed (QA Forensic Audit)

- **QA-BUG-001 (CRITICAL):** Stack buffer overflow in `sample_top_k()` when `k > 1024`.
  Added `TOP_K_MAX` constant and clamping logic to prevent stack-allocated arrays
  from being overrun. (`src/sampling/top_k.c`)

- **QA-BUG-002 (HIGH):** Buffer overflow in `thought_filter_process()` — output buffer
  had no size parameter, allowing unbounded writes. Added `output_buf_size` parameter
  to the function signature and bounds checks on all three output write paths.
  Updated all callers. (`src/reasoning/thought_filter.c`, `include/reasoning/thought_filter.h`,
  `src/reasoning/reasoning_generate.c`, `tests/test_reasoning.c`)

- **QA-BUG-003 (MEDIUM):** Windows HANDLE truncation in `MappedFile` struct. On 64-bit
  Windows, the file handle was stored as `int`, truncating the pointer-sized `HANDLE`.
  Changed to `void *handle` under `_WIN32`. (`include/memory/mapped_file.h`,
  `src/memory/mapped_file.c`)

- **Pre-existing:** Missing includes in `reasoning_generate.c` causing implicit function
  declaration warnings for `transformer_forward`, `sample_argmax`, `apply_temperature`,
  `rng_seed`, and `sample_top_p`.

### Improved

- **QA-ISS-004 (LOW):** Made `byte_decode_buf` in `tokenizer_decode.c` thread-local
  using `__thread` (POSIX) / `__declspec(thread)` (MSVC) to prevent data races in
  concurrent decoding. (`src/tokenizer/tokenizer_decode.c`)

- **QA-ISS-005 (INFO):** Added TODO comment in `include/math/simd_dispatch.h` noting
  the need for runtime CPUID detection to replace compile-time SIMD dispatch.

### Added

- `tests/test_bugfixes.c` — Regression tests for QA-BUG-001, QA-BUG-002, QA-BUG-003,
  PRE10-BUG-001, PRE10-BUG-002, and PRE10-BUG-003 (15 test functions, 35 assertions).
- `docs/FIX_PLAN.md` — Detailed fix plan for all QA audit findings.
- `docs/PHASE10_PRE_AUDIT_REPORT.md` — Comprehensive forensic investigation report
  documenting 20 testing rounds, root cause analysis of recurring bugs, and Phase 10
  readiness assessment.

### Changed (Phase 10 SIMD Dispatch)

- `include/math/simd_dispatch.h` — Added `tn_matmul_packed_fn` and `tn_unpack_fn`
  function pointer types; added `tn_ternary_matmul_packed` and `tn_unpack_block`
  global dispatch pointers.
- `src/math/simd_dispatch.c` — Wires packed matmul and unpack dispatch to AVX2
  or scalar implementations at runtime via `tn_simd_init()`.
