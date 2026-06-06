# Project Zero — Comprehensive Analysis Report

**Generated:** 2026-03-16
**Branch:** claude/project-analysis-report-bOpTK
**Scope:** Full codebase analysis — architecture, performance, security, quality

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture & Code Structure](#2-architecture--code-structure)
3. [Performance Analysis](#3-performance-analysis)
4. [Key Components Deep-Dive](#4-key-components-deep-dive)
5. [Security & Code Quality](#5-security--code-quality)
6. [Findings & Recommendations](#6-findings--recommendations)

---

## 1. Executive Summary

Project Zero is a **from-scratch, single-binary LLM inference engine** written in
pure C, designed to run Microsoft's BitNet b1.58-2B-4T model at maximum speed on
commodity x86-64 CPUs — without any GPU, Python, or external ML framework.

### What It Does

The engine loads a 1.18 GB ternary-weight model (2-bit packed weights + BF16
embeddings), runs autoregressive token generation, and exposes a REPL and
CLI interface. A Phase 14 agentic layer adds tool execution (shell, file I/O)
on top of the core inference loop.

### Key Achievements (as of Phase 14)

| Metric | Value |
|---|---|
| Throughput (single DDR4 channel) | **13 tok/s** |
| Throughput (dual-channel upgrade) | **~16 tok/s** |
| Improvement over baseline | **9.3×** (1.4 → 13 tok/s) |
| Model size on disk | **1.18 GB** (.bin), ~660 MB saved vs. float32 embeddings |
| Build artifact | Single static binary (`adaptive_ai_engine`) |
| Lines of C | ~5,000 (source), ~1,000 (headers) |
| Phases completed | 14 |

### Architectural Philosophy

- Zero dynamic allocations in the hot path
- OS-managed memory via `mmap` + `MADV_SEQUENTIAL`
- Runtime SIMD dispatch (AVX-512 → AVX2 → NEON → scalar)
- Memory-bandwidth-bound by design; every optimization targets DRAM throughput

---

## 2. Architecture & Code Structure

### 2.1 Module Breakdown

The project is cleanly partitioned into 11 subsystems, each with a mirrored
`src/` and `include/` directory:

| Module | Files (`.c`) | Purpose |
|---|---|---|
| `agent/` | 5 | Agentic tool loop, shell execution, PTY runner, user approval |
| `cli/` | 4 | Entry point, CLI argument parsing, REPL, timer |
| `core/` | 5 | Config, weight loading, run-state, ternary unpack |
| `kv_cache/` | 3 | KV cache compression, eviction strategy, sliding window |
| `math/` | 12 | Matmul (scalar/AVX2/AVX512), RMSNorm, softmax, RoPE, elementwise |
| `memory/` | 2 | Aligned allocator, mmap wrapper |
| `multimodal/` | 5 | Vision encoder, patch extraction, projector (stub/WIP) |
| `reasoning/` | 3 | `<think>` prompt injection, hidden thought filter |
| `sampling/` | 5 | Top-p, top-k, temperature, argmax, RNG |
| `threading/` | 2 | Thread pool (spinlock + cond_var), CPU probe |
| `tokenizer/` | 3 | BPE load, encode, decode |
| `transformer/` | 5 | Forward pass, attention, FFN, embedding, generate |

**Total:** ~57 `.c` files, ~85 headers, **~5,000 lines of C**

### 2.2 Dependency Graph (High Level)

```
main.c
 └─ args.c / repl.c / timer.c
     └─ generate.c               ← token loop
         ├─ forward.c            ← single transformer step
         │   ├─ embedding.c      ← token → float vector
         │   ├─ attention.c      ← multi-head attention + RoPE
         │   ├─ ffn.c            ← SwiGLU / ReLU² FFN
         │   └─ simd_dispatch.c  ← runtime SIMD selection
         ├─ weights.c            ← mmap pointer arithmetic
         ├─ kv_cache/            ← sliding window KV store
         └─ sampling/            ← top-p / temperature / argmax
 └─ agent_loop.c (Phase 14)
     ├─ tool_interceptor.c       ← parse tool calls from output
     ├─ cmd_exec.c               ← PTY shell execution
     ├─ output_inject.c          ← inject tool results back
     └─ user_approval.c          ← interactive approval gate
```

### 2.3 Build System

The project uses both `CMakeLists.txt` and a hand-written `Makefile`. The
`Makefile` is the primary build surface and supports three targets:

| Target | Flags | Purpose |
|---|---|---|
| `make` (default) | `-O2 -g` | Debug build with assertions |
| `make release` | `-O3 -march=native -DNDEBUG` | Production; enables AVX-512 on Tiger Lake |
| `make test` | `-O2 -g --coverage` | Unit tests with gcov coverage |

Key compile-time feature flags from `include/core/platform.h`:

```c
TN_HAS_AVX512  // Defined when -mavx512f is present
TN_HAS_AVX2    // Defined when -mavx2 is present
TN_POSIX       // POSIX (Linux/macOS)
TN_WIN32       // Windows (MapViewOfFile path)
```

### 2.4 Data Flow: Token Generation

```
1. Prompt tokens → tokenizer_encode()
2. For each position:
   a. embedding_lookup(token_id)          → x[dim]  (BF16 → float32)
   b. for each layer l in 0..n_layers-1:
      i.   rmsnorm(x) → x_norm
      ii.  wq,wk,wv ternary_matmul → q,k,v   (per-layer scale)
      iii. rope_apply(q, k, pos)
      iv.  kv_cache_store(k, v, l, pos)
      v.   attention_forward(q, kv_cache) → attn_out
      vi.  residual: x += wo * attn_out
      vii. rmsnorm(x) → x_norm2
      viii.ffn_forward(x_norm2) → ffn_out   (ReLU² for BitNet)
      ix.  residual: x += ffn_out
   c. rmsnorm(x, w_final_norm)
   d. lm_head_matmul(x) → logits[vocab_size]
   e. sample(logits, temp, top_p) → next_token_id
3. Decode next_token_id → string; print
```

### 2.5 Weight Format (Custom Binary `.bin`)

The model file is a flat binary with 64-byte-aligned sections in a fixed order:

1. **Header** — `Config` struct (magic, version, dim, n_layers, vocab_size, …)
2. **Token embedding table** — BF16, `vocab_size × dim × 2` bytes
3. **Per-layer weights** (×24 layers):
   - Input/post-attention RMSNorm weights (float32)
   - Q/K/V/O packed ternary weights + per-layer float32 scale
   - Sub-norm weights (BitNet architectural requirement)
   - FFN w1/w2/w3 packed ternary + scales
4. **Final RMSNorm** weight (float32)
5. **LM head** — tied to embedding table (zero-copy pointer alias)

Ternary packing: 4 values per byte (2 bits each, values ∈ {-1, 0, +1}).
A group size of 64 elements shares one float32 scale factor.


---

## 3. Performance Analysis

### 3.1 Hardware Profile (Benchmarked System)

| Component | Specification |
|---|---|
| CPU | Intel Core i5-11300H (Tiger Lake, 4C/8T) |
| AVX-512 units | 1 FMA unit/core (consumer Tiger Lake) |
| Effective clock under AVX-512 | ~3.01 GHz (HT throttle active at 8 threads) |
| RAM | 8 GB DDR4-3200, **single channel** |
| Measured DRAM bandwidth | **15.4 GB/s** (vs. theoretical 25.6 GB/s) |
| DRAM latency | 115.1 ns |
| NVMe (cold read) | 1.08 GB/s |

### 3.2 Optimization Timeline

| Phase | Change | tok/s |
|---|---|---|
| Baseline | Debug build, 8 threads, earlyoom running | 1.4 |
| CPU governor | `performance` mode | 2.0 |
| Binary format | float32 embeddings → BF16 | 4.5 |
| SIMD kernels | AVX-512 ternary matmul | 5.5 |
| earlyoom removed | Memory pressure eliminated | 10.5 |
| Thread count | 8 → 4 (avoid AVX-512 HT cliff) | 11.5 |
| Top-p sampling | Static 200K buffer, pre-filter, no malloc | **13.0** |
| (projected) dual DDR4 | Second DIMM → dual-channel | **~16.1** |

**9.3× total improvement** from baseline to ceiling.

### 3.3 Bandwidth Ceiling Analysis

Each generated token requires reading the **full model weight set** once:

- Packed ternary weights: ~0.52 GB (2 bits/param)
- BF16 embedding table: ~0.52 GB
- Norms/scales: ~0.14 GB
- **Total per token: ~1.18 GB**

At 15.4 GB/s effective bandwidth: `15.4 / 1.18 = 13.05 tok/s`

This matches the measured 13.0 tok/s exactly — the engine is
**99.5% memory-bandwidth-bound**. No algorithmic improvement can
exceed the DRAM ceiling without a hardware upgrade.

### 3.4 The AVX-512 Hyperthreading Cliff

Tiger Lake's consumer AVX-512 implementation has one FMA unit per core
(vs. two on server SKUs). When Hyperthreading is active (8 logical threads),
two HW threads share one FMA unit, causing AVX-512 port contention
and a measurable throughput drop:

- 8 threads: **~11.5 tok/s** (contention)
- 4 threads: **~13.0 tok/s** (no sharing)

`cpu_probe.c` auto-detects physical core count from `/proc/cpuinfo` and
sets `n_threads = physical_cores` at startup to avoid this cliff.

### 3.5 Key Optimizations

#### Sampling Rewrite (largest single gain after earlyoom)
- Old: allocate `n × float` per token, call `qsort`, `malloc`/`free` per call
- New: **static 200K-entry buffer** pre-allocated at startup; radix-like
  threshold filter eliminates 90%+ of candidates before sort; zero heap
  activity in hot path

#### BF16 Embedding Storage
- Upper 16 bits of IEEE 754 float32 ≈ bfloat16 (same exponent, truncated mantissa)
- Embedding table: `vocab × dim` → `32K × 4096 × 2 bytes = 262 MB` (saved 262 MB)
- Expansion: `bf16 << 16` → float32 at decode time; zero precision loss for
  embedding lookup

#### Packed Ternary Matmul
- 4 weights per byte (2-bit packing): `{-1, 0, +1}` → `{0b11, 0b00, 0b01}`
- AVX-512 kernel processes 16 packed bytes (= 64 ternary values) per cycle
- Per-group (64-wide) float32 scale applied post-accumulate
- Fallback: AVX2 (8-wide) → scalar

#### Thread Pool Design
- C11 atomics + spin-then-sleep (`SPIN_LIMIT = 40,000` iterations, ~160 µs)
- Workers **never** reach `pthread_cond_wait` during active inference
  (inter-dispatch < 100 µs); eliminates all futex syscall overhead
- `spin_epoch` counter: atomic release write by dispatcher, acquire reads
  by workers — lock-free dispatch protocol


---

## 4. Key Components Deep-Dive

### 4.1 Attention (`src/transformer/attention.c`)

The attention module implements multi-head attention with **Grouped Query
Attention (GQA)** and BitNet's **sub-norm** variant.

Key design points:
- **GQA multiplier**: `kv_mul = n_heads / n_kv_heads`. For BitNet-2B:
  `32 heads / 8 KV heads = 4× reuse` — only 8 KV caches stored per layer,
  reducing KV memory by 4×.
- **Sliding window circular buffer**: `sw_map_position()` maps logical sequence
  positions to physical ring buffer slots. Allows context > `max_seq_len` with
  constant memory.
- **BitNet sub-norm** (Phase 12 fix): After the output concatenation but before
  the output projection, `rms_attn_sub_norm` is applied if present. This matches
  the HuggingFace reference architecture and was a critical correctness fix.
- **SIMD-dispatched inner products**: `tn_vec_dot()` and `tn_vec_saxpy()` resolve
  at runtime to AVX-512 / AVX2 / scalar based on CPU capability.
- **Historical bug fixed (PRE10-BUG-004)**: Value vector lookup used raw `t`
  instead of `sw_map_position()` after window wrap — caused wrong V reads.
  Fixed in Phase 10.

### 4.2 KV Cache (`src/kv_cache/`)

Three files implement a multi-strategy KV cache:

| File | Role |
|---|---|
| `kv_compress.c` | int8 absmax quantization (save ~4× KV memory) and int4 packing |
| `sliding_window.c` | Circular buffer ring management |
| `kv_strategy.c` | Strategy selector: full/sliding/compressed |

**int8 KV compression**:
- Per-slot `absmax` scale factor stored alongside 8-bit values
- Dequantize on read: `float = int8 × scale`
- Quantization error < 0.8% for typical activation distributions

**Sliding window**: Fixed `max_seq_len` (default 4096) ring buffer enables
131K+ context with a fixed memory footprint of `n_layers × n_kv_heads ×
max_seq × head_dim × sizeof(float)` ≈ 200 MB for BitNet-2B.

### 4.3 Math / SIMD Dispatch (`src/math/`)

The dispatch layer (`simd_dispatch.c`) populates a global function pointer table
at startup based on compile-time CPUID flags:

```c
tn_matmul_packed_fn  tn_ternary_matmul_packed;  // hot path
tn_rmsnorm_fn        tn_rmsnorm;
tn_softmax_fn        tn_softmax;
tn_vec_dot_fn        tn_vec_dot;
tn_vec_saxpy_fn      tn_vec_saxpy;
// … etc.
```

Priority: `AVX-512 > AVX2 > NEON > scalar`.

The packed AVX-512 ternary matmul (`ternary_matmul_packed_avx512.c`) is the
hottest function in the engine, called 3×N_heads + 3×FFN = ~90 times per
transformer layer per token.

### 4.4 Thread Pool (`src/threading/thread_pool.c`)

Custom C11 atomic thread pool with a spin-then-sleep protocol:

```
Dispatcher:
  1. fill task parameters (n_threads slices of output rows)
  2. atomic_store(spin_epoch, epoch+1)  // release fence — workers wake
  3. spin-wait on spin_remaining == 0   // or cond_wait fallback

Worker:
  1. spin-read spin_epoch until changed  // SPIN_LIMIT = 40,000 (~160 µs)
  2. atomic_fetch_add(spin_claimed, 1)  // claim slice idx
  3. execute kernel slice
  4. atomic_fetch_sub(spin_remaining, 1)
  5. if last: signal cond_done
```

This eliminates all futex syscalls during inference while still releasing
CPU during idle REPL wait periods.

### 4.5 Agentic Tool Loop (`src/agent/`)

Phase 14 adds a streaming tag parser that intercepts model output for tool
calls:

```
model output stream → ti_process_piece() → detect <exec>...</exec>
                                            detect <think>...</think>
                                            detect <save_memory>
                                            detect <search_memory>
```

When `<exec>cmd</exec>` is detected:
1. `user_approval.c` prompts the user (unless `CLAUDE_AUTO_APPROVE=1`)
2. `cmd_exec.c` forks a PTY child, captures stdout/stderr
3. `output_inject.c` formats result and re-tokenizes into the context
4. Generation continues with tool result in context

The agent loop (`agent_loop.c`) replicates the `generate()` hot path but
with the additional interceptor step after each decoded piece.

### 4.6 Tokenizer (`src/tokenizer/`)

BPE tokenizer compatible with Qwen/LLaMA vocabulary:

- **Load** (`tokenizer_load.c`): reads custom `.bin` format; vocabulary +
  merge scores; sorted by score for O(log n) merge lookup
- **Encode** (`tokenizer_encode.c`): UTF-8 → byte tokens → BPE merges
- **Decode** (`tokenizer_decode.c`): ID → string with byte-fallback for
  `<0xNN>` tokens

Vocabulary size: 32,000 tokens (matches BitNet-2B HF config).


---

## 5. Security & Code Quality

### 5.1 Security Posture Summary

| Category | Status | Detail |
|---|---|---|
| Buffer overflow (weight loading) | **Mitigated** | Strict bounds checking on every mmap read |
| Integer overflow (size calculations) | **Mitigated** | `size_t` arithmetic; config validation before alloc |
| OOB KV cache write (FM-007) | **Fixed** | `sw_map_position()` now correctly bounds physical slot |
| Thread pool race (FM-001) | **Fixed** | Atomic `spin_epoch` + separate `spin_claimed` counter |
| OOB tokenizer byte-token (AUD-TOK-01) | **Fixed** | Byte boundary checked before vocab index |
| Command injection (agent `<exec>`) | **Partial** | User approval gate; no shell metachar escaping |
| Model weight integrity | **Gap** | No SHA-256 or HMAC on weight file bulk data |
| Adversarial prompt injection | **Gap** | No sanitization before agent tool calls |
| CI/CD pipeline | **Gap** | No `.github/workflows/` present in repo |

### 5.2 Confirmed Bugs Fixed

#### FM-001 — Thread Pool Race Condition (Critical)
**Root cause:** Fast-finishing workers looped back, claimed extra slice indices
(≥ `num_threads`), and decremented `task_remaining` extra times, causing the
dispatcher to return before all output rows were written.
**Fix:** `spin_claimed` is now an `_Atomic` counter; workers bail out if
claimed index ≥ `num_threads`. The epoch protocol prevents re-entry.
**Status:** Verified fixed; 1,000 rapid-fire dispatch stress test passes.

#### FM-007 — KV Cache Out-of-Bounds Write (High)
**Root cause:** `sw_map_position()` could return a physical slot ≥ `max_seq_len`
after the ring buffer wrapped, causing `memcpy` to write past the KV array.
**Fix:** Physical slot clamped to `[n_pinned, max_seq_len)` range.
**Status:** Fixed + regression test `audit_sliding_win` passes.

#### PRE10-BUG-004 — Wrong Value Vectors After Window Wrap
**Root cause:** Attention's V-vector lookup used raw loop counter `t` instead
of `sw_map_position(&sw, hist_logical)`.
**Fix:** Added `v_mapped = sw_map_position(&sw, v_hist)` — matching the K lookup.
**Status:** Fixed in Phase 10.

### 5.3 Outstanding Security Gaps

#### Gap 1 — No Weight File Integrity Check
The engine validates the binary header (magic number, version) but does **not**
verify a checksum over the weight payload (~1.18 GB). Bit-rot or deliberate
tampering would produce silently wrong inference output.

**Recommendation:** Compute and store SHA-256 of the weight body at conversion
time; verify on load (adds ~100ms for a cold hash of 1.18 GB, negligible).

#### Gap 2 — Agent Shell Command Injection
`cmd_exec.c` executes the content of `<exec>...</exec>` tags verbatim via PTY.
No shell metacharacter sanitization is performed. A jailbroken model or adversarial
prompt could craft `<exec>rm -rf /</exec>`.

**Recommendation:** Either use `execvp()` with explicit argv (no shell) or apply
an allowlist of safe characters before passing to the PTY. The user approval gate
(`user_approval.c`) is a UX control, not a security control.

#### Gap 3 — `strtok` in `build_argv_from_command()`
`agent_loop.c` uses `strtok()` to split the command string. `strtok` is
non-reentrant (global state); if any callback re-enters the function the static
buffer will be corrupted.

**Recommendation:** Replace with `strtok_r()` (POSIX) or a manual tokenizer.

### 5.4 Code Quality Observations

**Strengths:**
- Consistent module boundaries; each `.c` file has a matching header
- Error codes (`TernaryError` enum) used consistently; no magic integer returns
- SIMD dispatch is cleanly separated from math logic — adding NEON is
  a matter of adding one branch in `simd_dispatch.c`
- Inline comments explain non-obvious algorithmic choices (GQA multiplier,
  AVX-512 HT cliff, sliding window wrap logic)
- 3,652 passing tests with zero failures across all test suites

**Areas for Improvement:**
- `agent_loop.c` duplicates significant portions of `generate.c` — the two
  should share a common token-generation callback hook
- `weights.c:weights_free_pointers()` has all `free()` calls on one long line
  with no bounds guard; the pattern is correct but hard to audit
- The multimodal module (`src/multimodal/`) is largely stub code with no
  tests; it increases binary size with no current utility
- No valgrind/ASan CI job; dynamic analysis gaps remain

### 5.5 Test Coverage Summary

| Suite | Tests | Scope |
|---|---|---|
| `test_config` | 18 | Config parsing, run-state allocation |
| `test_math` | 33 | Matmul, RMSNorm, Softmax, RoPE, elementwise |
| `test_mmap` | 7 | mmap open/close/bounds |
| `test_simd` | 23 | AVX2 vs. scalar correctness |
| `test_forward` | 378 | Embedding, MHA, GQA, FFN, full forward |
| `test_sampling` | 2,510 | RNG, argmax, temperature, top-p, top-k |
| `test_kv_cache` | 122 | int8/int4, sliding window, strategy |
| `test_threading` | 170 | Thread pool + FM-001/FM-006 regression |
| `test_tokenizer` | 44 | BPE encode/decode/load |
| `test_reasoning` | 32 | ThoughtFilter, prompt injector |
| `audit_*` | 288 | Race condition, KV OOB, math precision |
| `test_redbox/blackbox` | 27 | Adversarial + functional integrity |
| **Total** | **3,652** | **All passing** |


---

## 6. Findings & Recommendations

### 6.1 Priority Matrix

| Priority | Finding | Effort | Impact |
|---|---|---|---|
| P0 | Agent shell command injection (no metachar sanitization) | Low | Critical |
| P0 | `strtok` → `strtok_r` in `agent_loop.c` | Low | High |
| P1 | Weight file SHA-256 integrity check on load | Medium | High |
| P1 | CI/CD pipeline (GitHub Actions) — none exists | Medium | High |
| P2 | Merge `agent_loop.c` duplicate token loop into `generate.c` callback | Medium | Medium |
| P2 | Add valgrind/ASan test job | Low | Medium |
| P3 | Multimodal stubs — remove or gate behind compile flag | Low | Low |
| P3 | Dual-channel DDR4 upgrade for ~24% throughput gain | Hardware | High |

### 6.2 Performance Roadmap

The engine has reached the **single-channel DDR4 memory bandwidth ceiling**.
No software optimization can exceed the current 13 tok/s on this hardware.
Remaining improvements require hardware or model changes:

1. **Dual-channel DDR4** — Install a matched DIMM in slot BANK 1.
   Projected: ~16.1 tok/s (measured bandwidth × 1.24).

2. **Int4 KV cache** — Already implemented in `kv_compress.c`, not yet
   activated in the default strategy. Halves KV memory footprint; allows
   longer contexts within the same RAM.

3. **Speculative decoding** — A small (125M) draft model could propose 4–8
   tokens per step, verified by the 2B model. Potential 2–4× throughput
   on CPU at the cost of a second model load.

4. **Weight streaming / tiling** — For CPUs with large L3 (e.g., AMD EPYC),
   tiling weight reads to maximize cache reuse could add 10–20%.

### 6.3 Architecture Recommendations

1. **Shared generation callback**: Extract the token decode + output loop from
   both `generate.c` and `agent_loop.c` into a common `token_callback_t`
   function pointer. Pass it to a single `run_inference()` function.

2. **Config versioning**: Add a minor version field to the `.bin` header to
   allow forward-compatible format evolution without breaking existing binaries.

3. **Graceful OOM**: `weights_alloc_pointers()` returns `TN_ERR_OOM` correctly,
   but `run_state_alloc()` calls `malloc()` without consistent OOM propagation.
   Audit and ensure all allocation failures surface to the caller.

4. **Multimodal compile gate**: Add `#ifdef TN_BUILD_MULTIMODAL` around the
   vision encoder stub. It currently compiles unconditionally, increasing binary
   size and maintenance surface for unfinished code.

### 6.4 Summary Assessment

| Dimension | Grade | Notes |
|---|---|---|
| **Correctness** | A | 3,652 tests; all critical bugs fixed |
| **Performance** | A+ | At theoretical hardware ceiling |
| **Architecture** | A− | Clean modules; minor duplication in agent loop |
| **Security** | B− | Agent exec injection is a real risk |
| **Test Coverage** | A | Broad suite; missing valgrind/ASan |
| **Documentation** | A | Extensive docs, walkthroughs, audit reports |
| **Code Quality** | B+ | Consistent style; a few strtok/free pattern issues |

**Overall: A− (Excellent for a from-scratch C inference engine)**

---

*Report generated by Claude Code analysis on 2026-03-16.*
*Branch: `claude/project-analysis-report-bOpTK`*

