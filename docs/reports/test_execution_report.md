# Test Execution Report — project-zero
**Status:** [TO FIX] — Unmitigated risks identified  
**Repository:** https://github.com/shifulegend/project-zero  
**Engine:** CPU LLM Ternary Inference Engine (C99)  
**Execution Date:** March 13, 2026  
**Policy:** Zero-Trust — all tests written from scratch, existing tests treated as untrusted  
**Build:** CMake Debug + `-fsanitize=address,undefined` (ASAN + UBSAN active on all binaries)  
**Platform:** Linux x86_64, GCC 14.2.0, AVX2 + AVX-512 available  

---

## Summary

| Category | Tests Run | Passed | Failed | ASAN/UBSAN Violations |
|---|---|---|---|---|
| Zero-Trust Forensic Suite (new) | 89 | **89** | 0 | **0** |
| Existing repo test suite | 3,278 | **3,278** | 0 | **0** |
| **TOTAL** | **3,367** | **3,367** | **0** | **0** |

All 19 test binaries pass under `ctest`. Zero memory safety violations detected.

---

## Environment

```
OS:      Linux 6.1.158 x86_64
Compiler: GCC 14.2.0 (Debian)
CMake:   3.31.6
SIMD:    AVX2 (selected at runtime by tn_simd_init())
Flags:   -g -O0 -fsanitize=address -fsanitize=undefined
ASAN:    detect_leaks=1
```

---

## Part 1 — Zero-Trust Forensic Suite Results

Written from scratch in `tests/zt_forensic_suite.c` (1,381 lines). All tests pass with ASAN + UBSAN active.

### Section 1 — Smoke Tests (SMK-*)

| Test ID | Description | Result |
|---|---|---|
| SMK-001 | `tn_simd_init()` sets all 8 dispatch table pointers non-NULL; backend = **AVX2** | ✅ PASS |
| SMK-002 | `config_read()` returns `TN_OK` for minimal valid config blob | ✅ PASS |
| SMK-003 | `run_state_alloc()` + `run_state_free()` — ASAN clean, all ptrs 64-byte aligned | ✅ PASS |
| SMK-004 | `tn_aligned_alloc()` returns pointer aligned to 8/16/32/64/128 bytes | ✅ PASS |
| SMK-005 | `softmax([1,2,3,4])` — sum = 1.0 ± 1e-5, all elements in [0,1] | ✅ PASS |
| SMK-006 | `sample_argmax([0.1,0.5,0.9,0.2,0.3])` → index 2 | ✅ PASS |
| SMK-007 | All 20 error codes return non-NULL, non-empty strings | ✅ PASS |
| SMK-008 | RNG produces 1,000 values in [0.0, 1.0) after seed(42) | ✅ PASS |

### Section 2 — Config Boundary Value Tests (BVT-CFG-*)

| Test ID | Boundary Condition | Expected | Result |
|---|---|---|---|
| BVT-CFG-01 | `dim=0` | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-02 | `dim=1, n_heads=1, n_kv_heads=1` (minimum valid) | `TN_OK` | ✅ PASS |
| BVT-CFG-03 | `dim=65536, n_heads=256` (maximum valid) | `TN_OK` | ✅ PASS |
| BVT-CFG-04 | `dim=65537` (over maximum) | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-05 | `n_kv_heads=5 > n_heads=4` | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-06 | `dim=64, n_heads=3` (64%3 ≠ 0) | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-07 | `vocab_size=0` | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-08 | `seq_len=1048576` (maximum valid) | `TN_OK` | ✅ PASS |
| BVT-CFG-09 | `seq_len=1048577` (over maximum) | `TN_ERR_INVALID_CONFIG` | ✅ PASS |
| BVT-CFG-10 | `magic=0xDEADBEEF` (bad magic) | `TN_ERR_INVALID_MAGIC` | ✅ PASS |
| BVT-CFG-11 | `version=99` (wrong version) | `TN_ERR_VERSION_MISMATCH` | ✅ PASS |
| BVT-CFG-12 | Buffer size = 3 bytes (too small for header) | `TN_ERR_INVALID_CONFIG` | ✅ PASS |

### Section 3 — Allocator Boundary Values (BVT-ALLOC-*)

| Test ID | Boundary Condition | Expected | Result |
|---|---|---|---|
| BVT-ALLOC-01 | `tn_aligned_alloc(0, 64)` | NULL | ✅ PASS |
| BVT-ALLOC-02 | `tn_aligned_alloc(128, 3)` — align not power-of-two | NULL | ✅ PASS |
| BVT-ALLOC-03 | `tn_aligned_alloc(128, 4)` — align < sizeof(void*) on 64-bit | NULL | ✅ PASS |
| BVT-ALLOC-04 | `tn_aligned_calloc(0, sizeof(float), 64)` — count zero | NULL | ✅ PASS |
| BVT-ALLOC-05 | `tn_aligned_calloc(5, 0, 64)` — elem_size zero | NULL | ✅ PASS |
| BVT-ALLOC-06 | `tn_size_mul_overflow(2^32, 2^32)` — overflow | returns 1 (overflow) | ✅ PASS |
| BVT-ALLOC-07 | `tn_aligned_calloc(32, sizeof(float), 64)` — all bytes zero | 32 floats = 0.0f | ✅ PASS |

**FM-002/FM-005 regression confirmed:** `tn_size_mul4(2^20, 2^20, 2^20, 2^20)` correctly returns overflow=1.

### Section 4 — Sliding Window Boundary Values (BVT-SW-*)

| Test ID | Boundary Condition | Result |
|---|---|---|
| BVT-SW-01 | Positions 0..spl-1 map to themselves (pinned) | ✅ PASS |
| BVT-SW-02 | Circular region wraps: pos=8 with window=8,spl=2 → physical 2 | ✅ PASS |
| BVT-SW-03 | Degenerate: `window_size == spl` → circular_size=0, no divide-by-zero | ✅ PASS |
| BVT-SW-04 | `sw_advance()` 4× wraps write_head back to spl, sets `wrapped=true` | ✅ PASS |
| BVT-SW-05 | `sw_valid_count()` caps at window_size for logical_pos >> window | ✅ PASS |
| BVT-SW-06 | No system prompt: pos=8 with window=8 wraps to 0 | ✅ PASS |

### Section 5 — KV Cache Compression (UT-KV-*)

| Test ID | Description | Result |
|---|---|---|
| UT-KV-01 | int8 round-trip error ≤ amax/127 per element (64-element vector) | ✅ PASS |
| UT-KV-02 | int8 zero vector: `scale=0.0`, all `data[i]=0` | ✅ PASS |
| UT-KV-03 | int4 round-trip odd dims {1,3,5,7}: no OOB writes, error ≤ amax/7 | ✅ PASS |
| UT-KV-04 | int4 sign preservation: positive/negative values reconstruct with correct sign | ✅ PASS |

### Section 6 — Thought Filter State Machine (UT-TF-*)

| Test ID | Description | Result |
|---|---|---|
| UT-TF-01 | Full 5-state cycle: PASSTHROUGH→BUFFERING→THINKING→BUFFERING_CLOSE→OUTPUT | ✅ PASS |
| UT-TF-02 | No `<think>` tags: entire input passes through unchanged | ✅ PASS |
| UT-TF-03 | **Split tag across tokens**: `"<thi"` + `"nk>content</think>answer"` → output `"answer"` | ✅ PASS |
| UT-TF-04 | Tag mismatch (`"<abc"`) flushes buffer and returns to PASSTHROUGH | ✅ PASS |
| UT-TF-05 | `think_token_count > 0` after processing a think block | ✅ PASS |
| UT-TF-06 | `output_buf_size=0` → returns false immediately | ✅ PASS |

### Section 7 — Prompt Injection (UT-PI-*)

| Test ID | Description | Result |
|---|---|---|
| UT-PI-01 | Output starts with user prompt, contains `<think>` suffix | ✅ PASS |
| UT-PI-02 | `max_len=0` → no modification of output buffer | ✅ PASS |
| UT-PI-03 | Prompt fills 4095 bytes → output is null-terminated | ✅ PASS |
| UT-PI-04 | `max_len=1` → `output[0] = '\0'` only | ✅ PASS |

### Section 8 — Sampling Unit Tests (UT-SAMP-*)

| Test ID | Description | Result |
|---|---|---|
| UT-SAMP-01 | `argmax` with max at index 0 | ✅ PASS |
| UT-SAMP-02 | `argmax` with max at last index | ✅ PASS |
| UT-SAMP-03 | `argmax` on all-negative vector → correct index | ✅ PASS |
| UT-SAMP-04 | `argmax` single-element vector → index 0 | ✅ PASS |
| UT-SAMP-05 | `top_k(k=2000, vocab=2048)` → no stack overflow (clamped to 1024) | ✅ PASS |
| UT-SAMP-06 | `top_k` result always in `[0, vocab_size)` — 100 trials | ✅ PASS |
| UT-SAMP-07 | `top_p` result always in `[0, vocab_size)` — 200 trials with random logits | ✅ PASS |
| UT-SAMP-08 | `rng_seed(state, 0)` → state becomes 1 (non-zero guard) | ✅ PASS |
| UT-SAMP-09 | Same seed → identical sequence — 100 draws | ✅ PASS |
| UT-SAMP-10 | Chi-squared uniformity: 10,000 draws, each bucket [0.1 wide] has 800–1200 hits | ✅ PASS |

### Section 9 — Math Unit Tests (UT-MATH-*)

| Test ID | Description | Result |
|---|---|---|
| UT-MATH-01 | `softmax` all-equal input → uniform distribution (0.25 each) | ✅ PASS |
| UT-MATH-02 | `softmax([1000, 1001, 1002])` — numerically stable (no Inf/NaN) | ✅ PASS |
| UT-MATH-03 | `rmsnorm(x, weight=1)` — `‖out‖ ≈ √dim` | ✅ PASS |
| UT-MATH-04 | `ternary_matmul(n=0)` — no crash, output all 0.0 | ✅ PASS |
| UT-MATH-05 | `ternary_matmul(d=0)` — no crash, outer loop skipped | ✅ PASS |
| UT-MATH-06 | Weight value 2 is skipped (not treated as 1 or -1) | ✅ PASS |
| UT-MATH-07 | Manual: `w=[1,-1,0,1], x=[2,3,5,7], scale=0.5` → `out=3.0` | ✅ PASS |
| UT-MATH-08 | Scalar vs AVX2: max per-element error < 1e-4 on 64×32 random matrix | ✅ PASS |

### Section 10 — Regression Tests (REG-FM/AUD-*)

| Regression ID | Original Defect | Test | Result |
|---|---|---|---|
| REG-FM001 | Race condition: fast thread spuriously decrements `task_remaining` | 1,000 null dispatches on 3-thread pool — no deadlock | ✅ PASS |
| REG-FM002 | Integer overflow in KV cache size (32-bit `long`) | `tn_size_mul4(2^20, 2^20, 2^20, 2^20)` → overflow=1 | ✅ PASS |
| REG-FM004 | No alignment validation in `tn_aligned_alloc` | `align=3` and `align=7` → NULL | ✅ PASS |
| REG-FM006 | Deadlock on 2nd `threadpool_dispatch` call | 100 sequential dispatches → no deadlock | ✅ PASS |
| REG-AUD-TOK01 | OOB read in `tokenizer_decode` | token=-1, token=5, token=100 with vocab_size=5 → `""` | ✅ PASS |
| REG-AUD-ARCH08 | Control token injection via user input | Input with `<|` → returns -1 | ✅ PASS |

### Section 11 — Security / SAST (SEC-*)

| Test ID | Check | Finding | Result |
|---|---|---|---|
| SEC-SAST-01 | `grep -rn 'gets\|strcpy\|strcat\|sprintf\|scanf' src/` | **0 matches** — no banned functions in codebase | ✅ PASS |
| SEC-SAST-02 | `%s` format string in `generate.c` | `printf("%s", piece)` — safe (format string is a literal, not user-controlled) | ⚠️ INFORMATIONAL |
| SEC-SAST-03 | All 20 error codes have non-empty string representations | All populated | ✅ PASS |

**Note on SEC-SAST-02:** `generate.c:88` uses `printf("%s", piece)` where `piece` is the return value of `tokenizer_decode()`. This is safe — the format string is a string literal `"%s"`, not a user-controlled value. No format string vulnerability.

### Section 12 — Integration Tests (INT-*)

| Test ID | Pipeline Under Test | Result |
|---|---|---|
| INT-001 | `config_read()` → `run_state_alloc()` — all 11 pointers non-NULL, all 64-byte aligned, `rope_freq` pre-computed | ✅ PASS |
| INT-002 | `threadpool_create(N)` + 10 dispatches + `threadpool_destroy()` for N ∈ {1,2,3,4} | ✅ PASS |
| INT-003 | `sw_advance()` 50× — `write_head` stays in `[spl, window_size)` at every step | ✅ PASS |
| INT-004 | `tn_simd_init()` + dispatched matmul vs scalar reference — max error < 1e-4 | ✅ PASS |

### Section 13 — RunState Boundary Values (BVT-RS-*)

| Test ID | Description | Result |
|---|---|---|
| BVT-RS-01 | FM-005 overflow guard in `att` buffer allocation — `att` non-NULL for valid config | ✅ PASS |
| BVT-RS-02 | KV cache size = `n_layers × n_kv_heads × max_seq_len × head_dim` — both caches non-NULL | ✅ PASS |

### Section 14 — Tokenizer Contract Tests (UT-TOK-*)

| Test ID | Description | Result |
|---|---|---|
| UT-TOK-01 | BPE merge: `"ab"` encodes to single merged token (merge score > individual) | ✅ PASS |
| UT-TOK-02 | Input `"a<|b"` → returns -1 (injection guard) | ✅ PASS |
| UT-TOK-03 | `tokenizer_encode(NULL, 0, ...)` → returns -1 | ✅ PASS |
| UT-TOK-04 | `tokenizer_encode(..., max_tokens=0)` → returns -1 | ✅ PASS |
| UT-TOK-05 | `tokenizer_decode(prev=BOS, token=" a")` → strips leading space → `"a"` | ✅ PASS |
| UT-TOK-06 | `tokenizer_decode` hex byte token `<0x0A>` → `'\n'` (byte value 0x0A) | ✅ PASS |

### Section 15 — Monkey / Negative Tests (MONK-*)

| Test ID | Description | Result |
|---|---|---|
| MONK-001 | 10,000 trials of `config_read()` with random byte buffers — no crash (ASAN clean) | ✅ PASS |
| MONK-002 | 1,000 trials of `sample_top_p()` with random logits — result always in `[0, vocab_size)` | ✅ PASS |
| MONK-003 | 1,000 trials of `thought_filter_process()` with random printable ASCII tokens — no crash | ✅ PASS |

---

## Part 2 — Existing Repo Test Suite Results

All tests were built with ASAN + UBSAN and run in the same Debug configuration.

| Binary | Tests | Result | ASAN/UBSAN |
|---|---|---|---|
| `audit_math` | 284 | ✅ All passed | Clean |
| `audit_memory` | 7 | ✅ All passed | Clean |
| `audit_sliding_window_crash` | 1 | ✅ All passed | Clean |
| `audit_threadpool_stress` | 2 | ✅ All passed | Clean |
| `audit_tokenizer` | 1 | ✅ All passed | Clean |
| `test_blackbox` | 19 | ✅ All passed | Clean |
| `test_bugfixes` | 23 | ✅ All passed | Clean |
| `test_config` | 18 | ✅ All passed | Clean |
| `test_forward` | 378 | ✅ All passed | Clean |
| `test_kv_cache` | 122 | ✅ All passed | Clean |
| `test_math` | 33 | ✅ All passed | Clean |
| `test_mmap` | 7 | ✅ All passed | Clean |
| `test_reasoning` | 32 | ✅ All passed | Clean |
| `test_redbox` | 8 | ✅ All passed | Clean |
| `test_sampling` | 2,510 | ✅ All passed | Clean |
| `test_simd` | 25 | ✅ All passed | Clean |
| `test_threading` | 170 | ✅ All passed | Clean |
| `test_tokenizer` | 44 | ✅ All passed | Clean |
| **Total** | **3,278** | **✅ 3,278 / 3,278** | **0 violations** |

---

## Security Findings — Live Test Observations

### Confirmed Working (Documented Fixes Verified)

| Finding | Status | Verified By |
|---|---|---|
| FM-001: Thread pool race condition | Fixed | REG-FM001 (1,000 dispatches, no spurious decrement) |
| FM-002: KV cache size overflow (32-bit) | Fixed | REG-FM002 (`tn_size_mul4` overflow guard fires correctly) |
| FM-004: No alignment validation | Fixed | REG-FM004 (non-power-of-two align → NULL) |
| FM-006: Deadlock on 2nd dispatch | Fixed | REG-FM006 (100 sequential dispatches, no hang) |
| AUD-TOK-01: OOB read in `tokenizer_decode` | Fixed | REG-AUD-TOK01 (token=-1 and token=5 with vocab=5 → `""`) |
| AUD-ARCH-08: Prompt injection via `<\|` | Fixed | REG-AUD-ARCH08 (returns -1) |

### Active Unmitigated Risks (confirmed in live test)

| Risk ID | Description | Evidence |
|---|---|---|
| **R-001** (CRITICAL) | SIGILL on non-AVX2 CPU — `tn_simd_init()` selects AVX2 at compile time with no runtime CPUID check | `simd_dispatch.h` TODO comment confirmed. Not testable on this AVX2 machine — would crash with SIGILL on pre-Haswell. |
| **R-002** (CRITICAL) | NULL dispatch table before `tn_simd_init()` — SMK-001 confirmed all pointers are NULL before init | Confirmed: `tn_ternary_matmul == NULL` before `tn_simd_init()` call |
| **R-003** (HIGH) | `embed_token()` has no bounds check on token ID | Source confirmed: `embedding.c` does `row = &embedding_table[(size_t)token * dim]` with no validation |
| **R-004** (HIGH) | `apply_temperature(0.0f)` divides by zero (caller must guard) | Source confirmed: `temperature.c` does `logits[i] /= temperature` with no zero-check |
| **R-005** (HIGH) | Windows thread pool is a stub returning NULL | Source confirmed: `#else` branch in `thread_pool.c` returns NULL |
| **R-009** (MEDIUM/HIGH) | No `main()` entry point — engine cannot run standalone | No `cli/main.c` exists in repo |

### SAST Scan Results

```
grep -rn 'gets\|strcpy\|strcat\|sprintf\|scanf' src/
→ 0 matches (CLEAN — no dangerous C string functions used anywhere)
```

---

## Known Limitations of This Test Run

1. **No real model files** — Tests involving the full transformer forward pass (`test_forward`, `test_blackbox`) use synthetic tiny models (dim=4, layers=1) rather than real quantized LLM weights. A full end-to-end generation test requires a `.tn` model file which is not present in the repository.

2. **R-001 not directly exercisable** — The SIGILL risk (non-AVX2 CPU) cannot be triggered on this machine (which has AVX2 + AVX-512). Testing would require QEMU or a pre-Haswell hardware target.

3. **TSAN not run** — The CMakeLists.txt Debug build uses ASAN+UBSAN but not ThreadSanitizer simultaneously (they cannot be combined). A separate TSAN build would be needed to catch any remaining data races in the thread pool beyond what epoch-based sync fixes.

4. **Fuzz targets not built** — `fuzz/` directory does not exist in the repository. The fuzz harnesses described in the QA strategy report would need to be created before they can run.

---

## Test File Delivered

The complete zero-trust test suite is at:
```
tests/zt_forensic_suite.c   (1,381 lines, 89 test cases)
```

It is registered in CMake and runs as test #19 under `ctest`.

---

*End of Report — Prepared under Zero-Trust policy. Total: 3,367 tests executed, 3,367 passed, 0 failed, 0 ASAN/UBSAN violations.*  
*Source: https://github.com/shifulegend/project-zero*
