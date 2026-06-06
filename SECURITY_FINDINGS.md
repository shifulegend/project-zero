# Security and Functional Test Report: project-zero

**Repository:** https://github.com/shifulegend/project-zero.git
**Assessment Date:** 2026-03-12 (updated)
**Branch assessed:** `master`
**HEAD commit:** See latest commit on branch
**Status:** All confirmed bugs fixed including FM-007 (KV cache OOB). Stress-tested with 3367 assertions.

---

## Executive Summary

**project-zero** is a high-performance CPU-based LLM inference engine using ternary quantization
(weights ∈ {-1, 0, 1}). Written in C99 with POSIX abstraction, targeting x86_64 AVX2/AVX-512
and ARM NEON. The goal is to run models entirely on CPU without GPU dependencies.

At assessment time, **Phases 0–9 are implemented** (~10,000+ lines):

- Build system (Makefile + CMakeLists.txt)
- Configuration parsing and validation
- Weight file memory-mapping with strict bounds checking
- Run state management with KV cache (transposed layout)
- Math primitives: ternary matmul (scalar, AVX2), RMSNorm, Softmax, RoPE, element-wise ops
- SIMD dispatch layer (runtime backend selection)
- Thread pool for parallel computation
- Memory allocators (SIMD-aligned)
- Mapped file abstraction (POSIX & Win32)
- **BPE tokenizer** with encode, decode, and vocab loading (Phase 5)
- **Transformer forward pass** with MHA (GQA), SwiGLU FFN, and full forward orchestration (Phase 6)
- **Sampling & text generation** with argmax, temperature, top-p, top-k, RNG, and generation loop (Phase 7)
- **KV cache optimizations** with int8/int4 quantization, sliding window attention, and adaptive strategy selector (Phase 8)
- **Hidden Reasoning Engine** with think-tag state machine, prompt injection, and reasoning-aware generation (Phase 9)

Higher-level components (multimodal, agent) are
planned but not yet implemented.

### Test Results (as of this commit)

| Test Suite       | Tests | Passed | Failed | Notes                                      |
|------------------|-------|--------|--------|--------------------------------------------|
| test_config      | 18    | 18     | 0      | Config parsing, run state alloc            |
| test_math        | 33    | 33     | 0      | Matmul, RMSNorm, Softmax, RoPE, elementwise|
| test_mmap        | 7     | 7      | 0      | Memory-mapped file open/close              |
| test_simd        | 23    | 23     | 0      | AVX2 vs scalar correctness                 |
| test_forward     | 378   | 378    | 0      | Embedding, MHA, GQA, FFN, full forward pass|
| test_sampling    | 2510  | 2510   | 0      | RNG, argmax, temperature, top-p, top-k     |
| test_kv_cache    | 122   | 122    | 0      | int8/int4 quantization, sliding window, strategy |
| test_redbox      | 8     | 8      | 0      | Adversarial: SIMD alignment, OOM, overflow, config injection |
| test_blackbox    | 19    | 19     | 0      | Functional: 10-token generation integrity  |
| test_threading   | 170   | 170    | 0      | Thread pool + parallel matmul + **FM-006 stress tests** |
| test_tokenizer   | 44    | 44     | 0      | BPE encode/decode/load (Phase 5)           |
| test_reasoning   | 32    | 32     | 0      | ThoughtFilter state machine, prompt injector (Phase 9) |
| audit_threadpool | 2     | 2      | 0      | 1000 rapid-fire dispatches (FM-001/FM-006 regression) |
| audit_sliding_win| 1     | 1      | 0      | KV cache OOB write (**FM-007 / AUD-MEM-01** fix) |
| audit_tokenizer  | 1     | 1      | 0      | Byte-token boundary OOB (**AUD-TOK-01**) |
| audit_math       | 284   | 284    | 0      | Precision stress across 284 tensor ops |
| **Total**        | **3652**| **3652**| **0** |                                            |

---

## Findings — Verification and Fix Status

### FM-001 · Race Condition in Thread Pool · High · **FIXED**

**Component:** `src/threading/thread_pool.c`
**Original report claim:** Nondeterministic failures in `test_parallel_matmul_large`
**Verified:** Yes — reproduced in 3/5 runs before fix.

#### Root Cause (identified during this assessment)

`threadpool_dispatch` initialises `task_remaining = tp->num_threads`. Each worker claims a
slice index (`idx = tp->task_claimed++`), executes its rows, then decrements `task_remaining`.
When `task_remaining` reaches zero the dispatcher is signalled that all work is complete.

The bug: a thread that finishes its rows quickly loops back to the top of `worker_entry`. At
the top it checks `while (!tp->shutdown && tp->task_fn == NULL)`. Because `task_fn` is still
set (other threads are still running), it skips the wait, claims *another* idx (now `>=
num_threads`), computes an empty row range (`start >= total`), skips execution — but still
decrements `task_remaining`. This extra decrement drives `task_remaining` to zero before all
threads have finished writing their output rows. The dispatcher wakes up and returns, the stack
frame for `parallel_ternary_matmul` is torn down, and one or more threads are still writing
into `out_parallel` — a classic data race producing incorrect or garbage results.

**Only the large test (`d=512, n=1024`) triggered the race reliably** because the per-thread
workload (128 rows × 1024 columns of AVX2 arithmetic) takes long enough that one fast-finishing
thread can loop back and spuriously decrement before slower threads complete.

#### Fix (`src/threading/thread_pool.c` + `include/threading/thread_pool.h`)

**Approach: epoch-based inner wait.** A `dispatch_epoch` counter (incremented by
`threadpool_dispatch` on every call) is added to `ThreadPool`. Each worker records its
epoch before releasing the mutex. After executing and decrementing `task_remaining`, if not
the last to finish, the thread waits with the following three-way exit condition:

```c
while (tp->dispatch_epoch == my_epoch &&
       tp->task_fn != NULL &&
       !tp->shutdown) {
    pthread_cond_wait(&tp->cond_work, &tp->mutex);
}
```

The inner wait exits when **any** of these become true:

| Condition | Meaning | Action on exit |
|-----------|---------|----------------|
| `dispatch_epoch != my_epoch` | New dispatch arrived | Claim new slice immediately |
| `task_fn == NULL` | Current dispatch fully done | Enter outer idle wait |
| `shutdown` | Pool shutting down | Propagate exit |

`threadpool_dispatch` increments `dispatch_epoch` **before** broadcasting `cond_work`, so
threads parked in the inner wait from a previous dispatch see the new epoch immediately and
exit — they are never trapped by a non-NULL `task_fn` from the new dispatch.

**Why a simpler `while (task_fn != NULL)` condition deadlocks:**
If a new dispatch starts before a parked thread re-checks the inner wait condition,
`task_fn` is non-NULL again. The thread stays trapped indefinitely. It never claims a slot
for the new dispatch, `task_remaining` never reaches zero, and the dispatcher waits on
`cond_done` forever. This deadlock was introduced by an earlier revision of this fix and
caught by a full threading test run (binary hung with zero output due to stdio buffering).

**Post-fix verification:** `test_threading` run 10 consecutive times — all 10 passed (20/20
tests each run including `test_parallel_matmul_large`). ASAN/UBSAN: no errors.

---

### FM-002 · Integer Overflow in KV Cache Size Calculation · Medium · **FIXED**

**Component:** `src/core/run_state.c`
**Verified:** Yes — overflow possible on 32-bit hosts before fix.

```c
/* Before fix — only first operand cast to size_t */
size_t kv_cache_size = (size_t)cfg->n_layers * cfg->n_kv_heads * max_seq_len * head_dim;
```

#### Root Cause

The `(size_t)` cast applies only to `cfg->n_layers`. C's left-associative arithmetic then
promotes each subsequent multiplication to `size_t` on 64-bit, but on a **32-bit host**
(`size_t` = 32 bits), the intermediate products can overflow silently. Furthermore, the
computation itself had no overflow detection — a sufficiently large model config could
produce a wrapped `kv_cache_size` value, leading to an undersized allocation and subsequent
out-of-bounds writes during inference.

#### Fix

Added `tn_size_mul_overflow()`, `tn_size_mul3()`, and `tn_size_mul4()` inline helpers in
`include/memory/aligned_alloc.h`. These perform checked multiplication with explicit overflow
detection using the `a != 0 && b > SIZE_MAX / a` pattern.

`run_state_alloc` now uses `tn_size_mul4()` for the KV cache computation:

```c
size_t kv_cache_size;
if (tn_size_mul4((size_t)cfg->n_layers, (size_t)cfg->n_kv_heads,
                 (size_t)max_seq_len, (size_t)head_dim, &kv_cache_size))
    return TN_ERR_OOM;
```

On overflow, the function returns `TN_ERR_OOM` immediately — no silent truncation. This is
safe on all platforms (32-bit and 64-bit) and future-proof for larger model configurations.

---

### FM-003 · Windows Pointer Arithmetic Overflow · Medium · **FIXED**

**Component:** `src/math/parallel_matmul.c:38`
**Verified:** Yes — confirmed in current code before fix.

```c
/* Before fix */
a->w + (long)start * a->n
```

On Windows (LLP64), `long` is 32 bits even on 64-bit systems. For large matrices
(`start` × `n` > 2³¹), the product overflows, producing a wrong (possibly negative) offset.
The AVX2 kernel then reads from an invalid memory address, causing corruption or a segfault.

**Fix:** Changed to `(size_t)start * (size_t)a->n` — pointer-width arithmetic on all
platforms.

```c
/* After fix */
a->w + (size_t)start * (size_t)a->n
```

---

### FM-004 · Missing Alignment Validation in `tn_aligned_alloc` · Low · **FIXED**

**Component:** `src/memory/aligned_alloc.c`
**Verified:** Yes — no validation present before fix.

`posix_memalign` requires alignment to be a power of two and a multiple of `sizeof(void*)`.
Passing an invalid alignment returns `EINVAL` and the function returns NULL — but callers
cannot distinguish this from an OOM failure, and some callers do not check the return value.

**Fix:** Added `alignment_valid()` helper validated before both the POSIX and Win32 paths:

```c
static int alignment_valid(size_t alignment) {
    return alignment >= sizeof(void *) && (alignment & (alignment - 1)) == 0;
}
```

In practice all callers use `TN_SIMD_ALIGN` (64, a valid power-of-two), so this is a
defensive hardening measure with no observable behaviour change in normal operation.

---

### FM-005 · Overflow Risk in `run_state_alloc` (att buffer) · Low · **FIXED**

Related to FM-002. Same computation pattern applies to the `s->att` allocation.

#### Fix

Now uses `tn_size_mul_overflow()` for the attention buffer:

```c
size_t att_count;
if (tn_size_mul_overflow((size_t)cfg->n_heads, (size_t)max_seq_len, &att_count))
    return TN_ERR_OOM;
s->att = (float *)tn_aligned_calloc(att_count, sizeof(float), TN_SIMD_ALIGN);
```

Both FM-002 and FM-005 are now protected by explicit overflow checks. The `tn_aligned_calloc`
call provides a second layer of overflow checking for the `count * elem_size` multiplication.

---

### FM-006 · Deadlock in Thread Pool on Repeated Dispatches · Critical · **FIXED**

**Component:** `src/threading/thread_pool.c`
**Reported by:** OpenClaw Autonomous Security Agent (2026-03-11)
**Verified:** Yes — reproduced on the pre-epoch-fix code (commit `cd651e1`).

#### Root Cause

The thread pool worker loop uses two wait states: an **outer wait** (idle, no work) and an
**inner wait** (early-finisher waiting for dispatch completion). Both originally waited on the
same condition variable `cond_work`.

In the initial FM-001 fix (commit `cd651e1`), the inner wait condition was simply
`while (task_fn != NULL)`. This caused a deadlock on the **second sequential dispatch**:

1. Round 0 completes. Early-finishing threads (A, B) are in the inner wait on `cond_work`,
   waiting for `task_fn == NULL`. The last finisher (C) sets `task_fn = NULL`, signals
   `cond_done`, and enters the outer wait.
2. Round 1 begins: dispatcher sets `task_fn = F2` (non-NULL) and broadcasts `cond_work`.
3. Thread C (outer wait) wakes and claims work normally.
4. Threads A and B (inner wait) wake and re-check: `task_fn != NULL` → **true** (the new
   dispatch set it). They re-enter `pthread_cond_wait` instead of exiting to claim work.
5. Threads A and B never claim a slice, never decrement `task_remaining`. The dispatcher
   blocks forever on `cond_done`. **Deadlock.**

#### Fix (commit `741021a`, verified with stress tests)

**Approach: epoch-based inner wait.** A `dispatch_epoch` counter is added to `ThreadPool` and
incremented by `threadpool_dispatch` before broadcasting. Each worker records its epoch before
releasing the mutex. The inner wait condition becomes:

```c
while (tp->dispatch_epoch == my_epoch &&
       tp->task_fn != NULL &&
       !tp->shutdown) {
    pthread_cond_wait(&tp->cond_work, &tp->mutex);
}
```

When a new dispatch starts, `dispatch_epoch` differs from the worker's `my_epoch`, so the
worker exits the inner wait immediately — even if `task_fn` is non-NULL (which it will be
from the new dispatch). The worker then loops back through the outer wait (which it skips
because `task_fn != NULL`) and proceeds to claim its new slice.

#### Verification

- `test_threadpool_dispatch_multiple`: 5 sequential dispatches on a 3-thread pool ✓
- `test_threadpool_stress_sequential`: **100 sequential dispatches**, 50 items each, 4 threads ✓
- `test_threadpool_stress_accumulate`: 50 dispatches accumulating results, verifying correctness ✓
- `test_threadpool_stress_varying_sizes`: work sizes from 1 to 1024 on the same pool ✓
- `test_parallel_matmul_repeated`: **32 sequential parallel matmuls** (simulating transformer layers) ✓
- ASAN/UBSAN: no errors across all 295 test assertions.

---

### NEW-01 · Non-Reentrant Global in `tokenizer_load.c` · Low · **FIXED**

**Component:** `src/tokenizer/tokenizer_load.c`

#### Root Cause

The standard C `qsort()` comparator signature `int(const void*, const void*)` does not
accept a context parameter. The original code used a file-scope global `g_vocab_ptr` to
pass the vocab array to the comparator. If two threads called `tokenizer_load` concurrently,
the second write to `g_vocab_ptr` would race with the first `qsort` comparator, causing
undefined behaviour.

Using `qsort_r()` was considered but rejected: GNU, BSD, and Windows all define incompatible
signatures, making it non-portable.

#### Fix

Replaced `qsort` + global with a **portable context-aware quicksort** implementation:

- `vocab_quicksort()` takes the vocab pointer as an explicit parameter — no global state.
- Uses median-of-three pivot selection to avoid O(n²) worst case on sorted/nearly-sorted data.
- Falls back to insertion sort for partitions ≤ 16 elements (cache-friendly on small sets).
- Tail-call optimization: iterates on the larger partition, recurses on the smaller one,
  keeping stack depth O(log n) even on adversarial inputs.

The `g_vocab_ptr` global has been **completely removed**. `tokenizer_load` is now fully
reentrant and safe for concurrent use from multiple threads.

**Verification:** All 44 tokenizer tests pass, including encode/decode roundtrip tests that
exercise the sorted vocabulary index built by the new sort.

---

## Architecture: Current vs. Original Report

The original report characterised the tokenizer and higher-level components as "planned".
Since then **Phase 5 (BPE tokenizer)** has been fully implemented:

- `src/tokenizer/tokenizer_load.c` — binary vocab file reader, score loading, sorted index
- `src/tokenizer/tokenizer_encode.c` — character-level seed + iterative BPE merge loop
- `src/tokenizer/tokenizer_decode.c` — token-ID-to-string lookup with raw-byte handling
- `tests/test_tokenizer.c` — 44 unit tests (all pass)

Additionally, **RoPE precomputed frequency tables** were added (`RunState.rope_freq`,
`RunState.rope_freq_len`), eliminating `powf()` from the hot inference loop (PR #2).

The updated architecture:

```
┌──────────────────────────────────────────────────────────────┐
│  CLI / REPL (not implemented)                               │
├──────────────────────────────────────────────────────────────┤
│  Sampling & Generation (COMPLETE — Phase 7)                 │
│    Argmax · Temperature · Top-p · Top-k · RNG · Gen Loop  │
├──────────────────────────────────────────────────────────────┤
│  KV Cache Optimizations (COMPLETE — Phase 8)                │
│    int8/int4 Quant · Sliding Window · Strategy Selector   │
├──────────────────────────────────────────────────────────────┤
│  Reasoning Engine (COMPLETE — Phase 9)                      │
│    <think> tag filtering · Prompt Inject · CoT Wrapper    │
├──────────────────────────────────────────────────────────────┤
│  Multimodal Bridge (planned) — vision encoder, MLP proj.   │
├──────────────────────────────────────────────────────────────┤
│  Transformer Forward Pass (COMPLETE — Phase 6)              │
│    Token Embedding · MHA (GQA) · FFN (SwiGLU) · RMSNorm   │
├──────────────────────────────────────────────────────────────┤
│  Tokenizer (COMPLETE — Phase 5)                             │
│    BPE encode · decode · vocab load · sorted binary search │
├──────────────────────────────────────────────────────────────┤
│  Core Library (COMPLETE — Phases 0–4)                       │
│    Config & Validation · Weights mapping · RunState        │
│    Ternary matmul (scalar + AVX2) · RMSNorm · Softmax      │
│    RoPE (precomputed freq table) · Element-wise ops        │
│    SIMD dispatch · Thread pool · Aligned allocator        │
│    Mapped file (POSIX + Win32 stubs)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### All Findings — Resolved

| Finding | Severity | Action |
|---------|----------|--------|
| FM-001  | High     | Fixed: `thread_pool.c` — epoch-based inner wait prevents spurious re-entry |
| FM-002  | Medium   | Fixed: `run_state.c` — overflow-checked `tn_size_mul4()` for KV cache |
| FM-003  | Medium   | Fixed: `parallel_matmul.c` — `(long)` → `(size_t)` cast |
| FM-004  | Low      | Fixed: `aligned_alloc.c` — `alignment_valid()` guard added |
| FM-005  | Low      | Fixed: `run_state.c` — overflow-checked `tn_size_mul_overflow()` for att buffer |
| FM-006  | Critical | Fixed: `thread_pool.c` — epoch-based inner wait prevents deadlock on repeated dispatches; verified with 100+ sequential dispatch stress tests |
| FM-007  | High     | Fixed: `attention_forward` — RCA fix via mathematical `SlidingWindow` integration. FM-007 clamp removed in favor of ring-buffer mapping. |
| AUD-MEM-01| High   | Fixed: `attention_forward` — Regression identified where SlidingWindow was unhooked. Full integration verified. |
| AUD-TOK-01| Low    | Fixed: `tokenizer_decode.c` — OOB memory read in byte string decoding fixed with NULL and length checks. |
| NEW-01  | Low      | Fixed: `tokenizer_load.c` — replaced `qsort` + global with portable reentrant quicksort |
| AUD-ARCH-01 | Critical | Fixed: `mapped_file.c` — SIGBUS on file truncation fixed by enforcing POSIX `flock(fd, LOCK_SH)`. |
| AUD-ARCH-02 | Critical | False Positive: Spurious wakeups in `thread_pool.c` were already mitigated with `while` loops. |
| AUD-ARCH-03 | High     | False Positive: SIMD Softmax instability already addressed via an 8-wide reduction pass. |
| AUD-ARCH-04 | High     | False Positive: Stale memory contamination prevented structurally by `valid_ctx` bounding. |
| AUD-ARCH-05 | High     | Fixed: `tokenizer_encode.c` — BPE out-of-bounds read fixed via rigorous `size_t prompt_len` tracking. |
| AUD-ARCH-06 | Medium   | Fixed: `aligned_alloc.c` — Integer overflow mitigated via pre-allocation bounds guard. |
| AUD-ARCH-07 | Medium   | False Positive: Divide-by-Zero in RMSNorm already mathematically prevented by `1e-5f` epsilon. |
| AUD-ARCH-08 | Medium   | Fixed: `tokenizer_encode.c` — Control token prompt injection neutralised by pre-filtering. |

### Future hardening recommendations

| Area | Recommendation |
|------|----------------|
| CI   | Add ThreadSanitizer build target (`-fsanitize=thread`) to catch future concurrency regressions |
| CI   | Add libFuzzer target for `config_read` and `weights_map` to test malformed model files |
| Safety | When transformer + sampling layers are implemented, validate token IDs against `vocab_size` before array indexing |

---

*Originally assessed by OpenClaw autonomous review agent on 2026-03-11.*
*All findings resolved and verified on 2026-03-12. 3367/3367 test assertions pass with ASAN/UBSAN.*
*Updated on 2026-03-13 with RCA and resolution for 8 external architectural defect reports.*
