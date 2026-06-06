# CPU LLM Ternary Engine — Granular Implementation Plan

> Every module is broken down to the smallest compilable/testable unit.
> Dependencies between modules are explicitly marked.
> Each task produces exactly one file or one function.

---

## PHASE 0: Project Scaffolding & Build System ✅

### 0.1 — Directory Structure Creation ✅
Create the canonical source tree:
```
project-zero/
├── src/
│   ├── core/          # Data structures, config, error handling
│   ├── memory/        # mmap, malloc wrappers, platform abstraction
│   ├── math/          # Ternary matmul, RMSNorm, softmax, SIMD
│   ├── transformer/   # Attention, FFN, embedding, forward pass
│   ├── threading/     # Thread pool, worker dispatch, CPU probing
│   ├── kv_cache/      # KV cache alloc, compression, sliding window
│   ├── reasoning/     # Hidden thought loop, state machine
│   ├── tokenizer/     # BPE tokenizer, vocab loading
│   ├── multimodal/    # Vision encoder bridge, image patch injection
│   ├── sampling/      # Temperature, top-p, top-k, argmax
│   ├── agent/         # Tool interceptor, command executor, agentic loop
│   ├── rag/           # Embedding, vector DB, similarity search, auto-retrieve
│   └── cli/           # Argument parsing, REPL, main entry point
├── include/           # All public .h headers (mirrors src/ subdirs)
├── tools/             # Python weight conversion scripts
├── tests/             # Unit tests per module
├── Makefile
└── CMakeLists.txt
```
**Deliverable:** Shell script or manual mkdir commands; empty placeholder `.gitkeep` files.

### 0.2 — Makefile (Primary Build) ✅
- File: `Makefile`
- Targets: `all`, `clean`, `test`, `debug`, `release`, `objs`
- Flags: `-O3 -march=native -pthread -lm` for release; `-g -O0 -fsanitize=address` for debug
- Pattern rule: `src/**/*.c` → `build/**/*.o` → `adaptive_ai_engine`
- Test target builds independently of main executable (no `main()` required)
- **No external dependencies.** Pure POSIX + C99.

### 0.3 — CMakeLists.txt (Cross-Platform) ✅
- File: `CMakeLists.txt`
- Minimum CMake 3.10, C99 standard
- Option flags: `-DENABLE_AVX2=ON`, `-DENABLE_AVX512=OFF`, `-DENABLE_NEON=OFF`, `-DENABLE_TESTS=ON`
- Auto-detect platform (Linux/macOS/Windows) and set compiler flags accordingly
- MSVC support: `/arch:AVX2`, `/arch:AVX512` flags, no `-pthread` on Windows
- Test targets: auto-discovered, each linked independently with sanitizer flags
- **This is the recommended build path for Windows native builds (outside WSL)**

### 0.4 — Top-Level Error Code Header ✅
- File: `include/core/error.h`
- Define: `enum TernaryError { OK = 0, ERR_FILE_OPEN, ERR_FILE_STAT, ERR_MMAP_FAILED, ERR_OOM, ERR_INVALID_CONFIG, ERR_INVALID_WEIGHTS, ERR_THREAD_CREATE, ERR_TOKENIZER_LOAD, ... }`
- Macro: `#define TERNARY_CHECK(expr, err) do { if (!(expr)) return (err); } while(0)`

### 0.5 — Platform Detection Header ✅
- File: `include/core/platform.h`
- `#ifdef _WIN32` / `#ifdef __APPLE__` / `#ifdef __linux__` guards
- Typedef: `typedef int64_t tn_i64; typedef uint8_t tn_u8;` etc.
- Define `TN_POSIX` or `TN_WIN32` macro for conditional compilation throughout.

---

## PHASE 1: Core Data Structures ✅

### 1.1 — Config Struct ✅
- File: `include/core/config.h`
- Define `typedef struct { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; } Config;`
- Function declaration: `TernaryError config_read_from_file(Config *cfg, const void *mapped_ptr);`
- Validates all fields > 0 and within sane bounds.

### 1.2 — Config Implementation ✅
- File: `src/core/config.c`
- `config_read_from_file()`: Reads first `sizeof(Config)` bytes from mapped pointer, byte-swaps if needed, validates.
- `config_print(const Config *cfg)`: Pretty-prints config to stdout for boot diagnostics.

### 1.3 — TransformerWeights Struct ✅
- File: `include/core/weights.h`
- Define `TransformerWeights` struct with:
  - `int8_t *token_embedding_table`
  - `int8_t **wq, **wk, **wv, **wo` (per-layer attention)
  - `int8_t **w1, **w2, **w3` (per-layer FFN)
  - `float **rms_att_weight, **rms_ffn_weight` (per-layer norms)
  - `float *rms_final_weight`
  - `int8_t *wcls` (output classifier)
  - `float *scales` (per-matrix ternary scale factors)
- Function declarations: `alloc_weight_pointers()`, `free_weight_pointers()`, `memory_map_weights()`.

### 1.4 — TransformerWeights Implementation ✅
- File: `src/core/weights.c`
- `alloc_weight_pointers(TransformerWeights *w, const Config *p)`: mallocs the `int8_t**` and `float**` pointer arrays (n_layers entries each). Does NOT allocate weight data (that's mmap'd).
- `free_weight_pointers(TransformerWeights *w)`: Frees only the pointer arrays.
- `memory_map_weights(TransformerWeights *w, const Config *p, int8_t *mapped_ptr)`: Pointer-arithmetic walk over the mapped blob, assigning each sub-pointer. Reads per-matrix scale floats from the tail of each layer's block.

### 1.5 — RunState Struct ✅
- File: `include/core/run_state.h`
- Define `RunState`:
  - `float *x, *xb, *xb2` — scratch vectors (dim)
  - `float *hb, *hb2` — FFN hidden buffers (hidden_dim)
  - `float *q, *k, *v` — attention head buffers
  - `float *att` — attention scores (n_heads * seq_len)
  - `float *logits` — output logits (vocab_size)
  - `float *key_cache, *value_cache` — KV cache **(TRANSPOSED layout — see below)**
  - `int current_pos` — current token position
  - `int max_seq_len` — dynamically computed max context
- Function declarations: `allocate_run_state()`, `free_run_state()`.
- **KV Cache Memory Layout** *(Critical Fix: Cache Miss Elimination)*:
  - **Naive layout (WRONG):** `[layer][position][head][dim]` — When computing attention for a single head across all cached positions, the CPU must jump by `n_heads * head_dim` floats between each position. This causes catastrophic L1/L2 cache misses on long sequences.
  - **Transposed layout (CORRECT):** `[layer][head][position][dim]` — All positions for a single head are **contiguous in memory**. The attention dot-product loop becomes a linear memory sweep, which AVX2/AVX-512 instructions can process at full bandwidth without stalls.
  - Total size: `n_layers * n_kv_heads * max_seq_len * head_dim * sizeof(float)`.
  - Indexing macro: `#define KV_CACHE_IDX(layer, head, pos, dim) ((layer) * kv_heads * max_seq * head_dim + (head) * max_seq * head_dim + (pos) * head_dim + (dim))`

### 1.6 — RunState Implementation ✅
- File: `src/core/run_state.c`
- `allocate_run_state(RunState *s, const Config *p, int max_seq_len)`: Allocate all scratch buffers + KV cache using **`tn_aligned_calloc()` (Phase 2.6)** with `TN_SIMD_ALIGN` (64-byte) alignment. This guarantees every `float *` buffer is safe for AVX2/AVX-512/NEON SIMD loads without segfaults or performance penalties. Zero-initializes via calloc semantics.
- `free_run_state(RunState *s)`: Free all buffers via **`tn_aligned_free()`**, NULL-out pointers.
- **Dependency:** Requires Phase 2.6 (aligned allocator). During early development before Phase 2.6 exists, a `#define tn_aligned_calloc(...) calloc(...)` shim in `aligned_alloc.h` allows the code to compile with standard malloc (SIMD loads must use unaligned `_mm256_loadu_ps` variants until alignment is guaranteed).

---

## PHASE 2: Memory Subsystem ✅ (core complete, 2.4/2.5 deferred)

### 2.1 — mmap Abstraction Header ✅
- File: `include/memory/mapped_file.h`
- Define `typedef struct { void *data; size_t size; int fd; } MappedFile;`
- Declarations: `TernaryError mapped_file_open(MappedFile *mf, const char *path);`
- `void mapped_file_close(MappedFile *mf);`

### 2.2 — mmap Implementation (POSIX) ✅
- File: `src/memory/mapped_file_posix.c`
- Guarded by `#ifdef TN_POSIX`
- `mapped_file_open()`: `open()` → `fstat()` → `mmap(PROT_READ, MAP_PRIVATE)` → `madvise(MADV_SEQUENTIAL)` hint.
- `mapped_file_close()`: `munmap()` → `close()`.
- All error paths set `MappedFile` to zeroed state.

### 2.3 — mmap Implementation (Windows) ✅
- File: `src/memory/mapped_file_win32.c`
- Guarded by `#ifdef TN_WIN32`
- Uses `CreateFileA()` → `GetFileSizeEx()` → `CreateFileMappingA()` → `MapViewOfFile()`.
- Close path: `UnmapViewOfFile()` → `CloseHandle()` (mapping) → `CloseHandle()` (file).

### 2.4 — Hardware RAM Probe
- File: `include/memory/hw_probe.h` + `src/memory/hw_probe.c`
- `int64_t get_free_ram_bytes()`:
  - Linux: Parse `/proc/meminfo` for `MemAvailable`.
  - macOS: `sysctl(HW_MEMSIZE)` + `vm_statistics64` for free pages.
  - Windows: `GlobalMemoryStatusEx()`.
- `int64_t get_total_ram_bytes()`: Same sources, total RAM.

### 2.5 — Dynamic Memory Budget Calculator
- File: `include/memory/mem_budget.h` + `src/memory/mem_budget.c`
- `MemBudget compute_mem_budget(const Config *cfg, int64_t free_ram)`:
  - Calculates `bytes_per_token = 2 * n_layers * kv_dim * sizeof(float)`.
  - `safe_allowance = free_ram * 0.80`.
  - `max_tokens = min(safe_allowance / bytes_per_token, cfg->seq_len)`.
  - Returns struct: `{ int max_seq_len; int64_t kv_cache_bytes; bool use_kv_compression; bool use_sliding_window; }`.

### 2.6 — SIMD-Aligned Memory Allocator *(Critical Fix: Alignment Trap)* ✅
- File: `include/memory/aligned_alloc.h` + `src/memory/aligned_alloc.c`
- **Why this exists:** Standard `malloc()` returns memory with no alignment guarantee. AVX2 requires 32-byte alignment; AVX-512 requires 64-byte alignment. Feeding an unaligned pointer to `_mm256_load_ps` or `_mm512_load_ps` causes either a massive performance penalty (silent fallback to unaligned loads) or a **Segmentation Fault** (instant crash). Every buffer touched by SIMD math must be allocated through this module.
- Functions:
  - `void *tn_aligned_alloc(size_t size, size_t alignment);`
    - POSIX: `posix_memalign(&ptr, alignment, size);`
    - macOS (also POSIX but older versions): same, with fallback to `aligned_alloc()` on C11.
    - Windows: `_aligned_malloc(size, alignment);`
  - `void tn_aligned_free(void *ptr);`
    - POSIX: standard `free(ptr);` (posix_memalign is free-compatible).
    - Windows: `_aligned_free(ptr);` (**not** standard `free` — will crash).
  - `void *tn_aligned_calloc(size_t count, size_t elem_size, size_t alignment);`
    - Allocates + zero-fills. Convenience wrapper used by RunState init.
  - **Overflow-safe multiplication helpers** (FM-002/FM-005 fix):
    - `int tn_size_mul_overflow(size_t a, size_t b, size_t *result);` — returns 1 on overflow.
    - `int tn_size_mul3(size_t a, size_t b, size_t c, size_t *result);` — 3-factor checked multiply.
    - `int tn_size_mul4(size_t a, size_t b, size_t c, size_t d, size_t *result);` — 4-factor checked multiply.
    - Defined as `static inline` in `aligned_alloc.h` for use by `run_state_alloc` and future allocations.
- Constants:
  - `#define TN_SIMD_ALIGN 64` — 64-byte alignment covers AVX-512 (512-bit = 64 bytes) and is also optimal for CPU cache lines. AVX2 (32-byte) and NEON (16-byte) are automatically satisfied since 64 is a multiple of both.
- **Consumers:** Every `float *` buffer in `RunState` (Phase 1.6), the KV cache buffers (Phase 8), and any temporary SIMD scratch buffers in the matmul kernels.

---

## PHASE 3: Math Primitives ✅ (scalar + AVX2 complete)

### 3.1 — Ternary MatMul (Scalar Reference) ✅
- File: `include/math/ternary_matmul.h`
- Declaration: `void ternary_matmul(float *out, const float *x, const int8_t *w, int n, int d, float scale);`
- File: `src/math/ternary_matmul_scalar.c`
- Pure C scalar implementation: double nested loop, branch on weight == 1 / -1 / 0.
- This is the **reference implementation** all SIMD variants are tested against.
- **Per-group scale factor note:** Currently applies one scale per matrix. Phase 10 packed weights
  may use per-group scales (e.g., every 128 weights). The inner loop will need to accumulate
  per-group and apply `out[i] = sum_g( scale[g] * sum_j_in_g(...) )` when that transition happens.

### 3.2 — Ternary MatMul (Branchless Scalar)
- File: `src/math/ternary_matmul_branchless.c`
- Eliminates `if/else` with: `val += (float)weight * x[j];` — the compiler can optimize since weight ∈ {-1,0,1}, the multiply becomes a conditional negate.
- Alternative: `val += x[j] * (weight > 0) - x[j] * (weight < 0);`

### 3.3 — Ternary MatMul (AVX2 SIMD — x86_64) ✅
- File: `src/math/ternary_matmul_avx2.c`
- Guarded by `#ifdef __AVX2__` (via `TN_HAS_AVX2` macro)
- **Implemented AVX2 kernel** (`ternary_matmul_avx2`):
  ```c
  // Core inner loop — processes 8 floats simultaneously:
  //
  // 1. Zero 8-wide accumulator:  __m256 accum = _mm256_setzero_ps()
  //
  // 2. LUT for 2-bit unpack:     __m256i unpack_lut = _mm256_setr_epi8(
  //                                   0,1,255, 0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
  //                                   0,1,255, 0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
  //    Maps packed 2-bit values:  00→0, 01→1, 10→-1 (255 as unsigned int8)
  //
  // 3. Per-iteration (8 weights at a time):
  //    a. Load 8 floats:          _mm256_loadu_ps(&x[j])
  //    b. Load 2 packed bytes:    *(uint16_t*)(&w_packed[j/4])
  //    c. LUT unpack:             _mm256_shuffle_epi8(lut, raw_ints)
  //    d. Build masks:            mask_pos = _mm256_cmpeq_epi32(weights, 1)
  //                               mask_neg = _mm256_cmpeq_epi32(weights, -1)
  //    e. Masked filter:          to_add = _mm256_and_ps(x_vec, mask_pos)
  //                               to_sub = _mm256_and_ps(x_vec, mask_neg)
  //    f. Accumulate:             accum += to_add; accum -= to_sub
  //       ZERO floating-point multiplications in the entire pipeline.
  //
  // 4. Horizontal sum:            Store 8 accumulators → sum to scalar float
  ```
- **Why this eliminates the FPU multiplier:** Standard matmul requires `_mm256_fmadd_ps` (fused multiply-add, ~5 cycles latency). This kernel uses only `_mm256_and_ps` (1 cycle, bitwise), `_mm256_add_ps` (3 cycles), and `_mm256_sub_ps` (3 cycles). The CPU's floating-point multiplier circuit is never touched.
- **Alignment note:** Input `x` must be 32-byte aligned (guaranteed by Phase 2.6 allocator). Use `_mm256_load_ps` (aligned, faster) instead of `_mm256_loadu_ps` (unaligned, safe fallback).
- **Tail loop:** For `n % 8 != 0`, scalar fallback processes remaining elements.
- **Outer loop:** The full matmul wraps this dot-product in a `for (int i = 0; i < d; i++)` loop, computing one output row per call, with `out[i] = ternary_dot_product_avx2(x, &w_packed[i * (n/4)], n) * scale;`
- **Phase 10 evolution note:** The current implementation takes `tn_i8` weights (1 byte per weight).
  This is the correct intermediate step — it lets us build, test, and validate the full SIMD pipeline
  with clean, debuggable values. When Phase 10 weight packing arrives (4 weights per byte), the
  modification is surgical: a `_mm256_shuffle_epi8` LUT step is inserted at the top of the inner
  loop to unpack 2-bit values into the `int8` registers this kernel already uses. The masking logic
  (`cmpeq_epi32` → `and_ps` → `add_ps`/`sub_ps`) remains **exactly the same**. The `scale`
  parameter will change to `const float *scales` for per-group scale factors. See Phase 10.4–10.5
  for the fused unpack+compute kernel that eliminates the intermediate buffer entirely.

### 3.4 — Ternary MatMul (AVX-512 SIMD — x86_64)
- File: `src/math/ternary_matmul_avx512.c`
- Guarded by `#ifdef __AVX512F__`
- Processes 16 floats per iteration using `_mm512_*` intrinsics.
- Masked tail via `_mm512_maskz_loadu_ps`.

### 3.5 — Ternary MatMul (NEON SIMD — ARM)
- File: `src/math/ternary_matmul_neon.c`
- Guarded by `#ifdef __ARM_NEON`
- Uses `vld1q_f32`, `vaddq_f32`, `vsubq_f32`.
- Processes 4 floats per iteration.

### 3.6 — SIMD Dispatch (Runtime Selection) ✅
- File: `include/math/simd_dispatch.h` + `src/math/simd_dispatch.c`
- **Expanded scope:** Dispatches ALL math operations, not just matmul.
- Global function pointers: `tn_ternary_matmul`, `tn_rmsnorm`, `tn_softmax`, `tn_vec_add`, `tn_vec_mul`, `tn_vec_scale`, `tn_silu`, `tn_vec_dot`
- `const char *tn_simd_init(void)` — initializes dispatch table, returns backend name
- Current: compile-time detection via `TN_HAS_AVX2`
- TODO: Add runtime CPUID probing for heterogeneous deployment
- Fallback: scalar reference implementations

### 3.7 — RMSNorm (Scalar) ✅
- File: `include/math/rmsnorm.h` + `src/math/rmsnorm.c`
- `void rmsnorm(float *out, const float *x, const float *weight, int size);`
- Computes: `ss = sum(x[i]^2) / size`, `out[i] = x[i] * weight[i] / sqrt(ss + 1e-5)`.

### 3.7a — RMSNorm (AVX2) ✅
- File: `src/math/rmsnorm_avx2.c`
- `void rmsnorm_avx2(...)` — two-pass with 8-wide SIMD:
  - Pass 1: `_mm256_fmadd_ps` accumulates sum of squares
  - Pass 2: `_mm256_mul_ps` applies normalization + weight scaling
- Scalar tail for non-multiple-of-8 sizes. Tested against scalar baseline.

### 3.8 — Softmax (Scalar) ✅
- File: `include/math/softmax.h` + `src/math/softmax.c`
- `void softmax(float *x, int size);`
- Numerically stable: find max, subtract, exp, normalize.

### 3.8a — Softmax (AVX2) ✅
- File: `src/math/softmax_avx2.c`
- `void softmax_avx2(...)` — three-pass:
  - Pass 1: `_mm256_max_ps` for 8-wide max reduction
  - Pass 2: scalar `expf()` (no efficient AVX2 exp intrinsic)
  - Pass 3: `_mm256_mul_ps` for 8-wide normalization
- Tested against scalar baseline including extreme-value numerical stability.

### 3.9 — RoPE (Rotary Positional Embedding) ✅
- File: `include/math/rope.h` + `src/math/rope.c`
- `void apply_rope(float *q, float *k, int dim, int kv_dim, int head_dim, int pos, int n_heads, int n_kv_heads, const float *freq_table);`
- **Precomputed frequency table (PR #2):** `rope_precompute_freqs(float *freq, int head_dim)`
  fills a `head_dim/2` table of `1.0 / pow(10000.0, 2i/head_dim)` at init time. This
  eliminates `powf()` from the hot token loop (called once per layer per token) — previously
  the most expensive transcendental call in the critical path. The table is stored in
  `RunState.rope_freq` and reused across all positions and layers.
- Apply rotation per token position: `(q[2i], q[2i+1]) → (q[2i]*cos - q[2i+1]*sin, q[2i]*sin + q[2i+1]*cos)`.
- **Stays scalar — no AVX2 variant:** `cosf`/`sinf` have no AVX2 intrinsics. RoPE operates on
  pairs of floats with per-pair trigonometric computation. The hot path is matmul, not RoPE;
  RoPE is <1% of inference wall-clock time after precomputation.

### 3.10 — Element-wise Operations (Scalar) ✅
- File: `include/math/elementwise.h` + `src/math/elementwise.c`
- `void vec_add(float *out, const float *a, const float *b, int n);`
- `void vec_mul(float *out, const float *a, const float *b, int n);` (for SwiGLU gate)
- `void silu(float *x, int n);` — SiLU activation: `x[i] = x[i] / (1 + exp(-x[i]))`.
- `void vec_scale(float *x, float s, int n);`
- `float vec_dot(const float *a, const float *b, int n);`

### 3.10a — Element-wise Operations (AVX2) ✅
- File: `src/math/elementwise_avx2.c`
- `vec_add_avx2`, `vec_mul_avx2`, `vec_scale_avx2`: 8-wide SIMD load/op/store
- `vec_dot_avx2`: `_mm256_fmadd_ps` accumulation with horizontal sum
- `silu_avx2`: **scalar loop** — `expf()` has no efficient AVX2 intrinsic. A mixed approach
  (vectorize the division, scalar the exp) introduces SIMD↔scalar pipeline stalls that make it
  slower than a clean scalar loop. The `_avx2` suffix exists for dispatch table consistency;
  the implementation is intentionally scalar.
- All variants tested against scalar baselines with odd-sized inputs

---

## PHASE 4: Threading Subsystem ✅

### 4.1 — CPU Core Probe ✅
- File: `include/threading/cpu_probe.h` + `src/threading/cpu_probe.c`
- `int tn_get_optimal_thread_count();`
  - POSIX: `sysconf(_SC_NPROCESSORS_ONLN)`
  - Windows: `GetSystemInfo()` → `dwNumberOfProcessors`
  - Returns `max(1, num_cores - 1)`.

### 4.2 — Thread Pool Header ✅
- File: `include/threading/thread_pool.h`
- Define:
  ```c
  typedef struct {
      pthread_t *threads;
      int num_threads;
      pthread_mutex_t mutex;
      pthread_cond_t cond_work;
      pthread_cond_t cond_done;
      tn_task_fn task_fn;
      void *task_arg;
      int task_total;           /* total work items */
      int task_claimed;         /* next unclaimed slice index */
      int task_remaining;       /* countdown to zero */
      unsigned int dispatch_epoch; /* incremented each dispatch; prevents re-entry */
      bool shutdown;
  } ThreadPool;
  ```
- Declarations: `ThreadPool *threadpool_create(int n);`, `void threadpool_dispatch(ThreadPool *tp, tn_task_fn fn, void *arg, int total);`, `void threadpool_destroy(ThreadPool *tp);`

### 4.3 — Thread Pool Implementation ✅
- File: `src/threading/thread_pool.c`
- **Worker loop:** Each thread waits on `cond_work`. When signaled, claims a slice (`idx =
  task_claimed++`), computes its `[start, end)` row range, and executes. When done, decrements
  `task_remaining` and signals `cond_done` if zero.
- **Dispatch:** Sets `task_fn`, `task_arg`, `task_total`, resets `task_claimed = 0`,
  `task_remaining = num_threads`. Increments `dispatch_epoch`. Broadcasts `cond_work`. Waits on `cond_done`.
- **Destroy:** Sets `shutdown = true`, broadcasts, joins all threads.
- **Race condition fixed (FM-001):** A thread finishing its rows early could loop back,
  claim another idx (≥ num_threads), compute an empty range, still decrement
  `task_remaining`, and cause the dispatcher to wake prematurely. Fix: epoch-based inner
  wait — each worker records `dispatch_epoch` before releasing the mutex; after executing
  it parks while `epoch == my_epoch && task_fn != NULL && !shutdown`. `threadpool_dispatch`
  increments the epoch before broadcasting, so parked threads exit and claim from the new
  dispatch rather than staying trapped. See `SECURITY_FINDINGS.md` FM-001.
- **Deadlock fixed (FM-006):** The epoch-based inner wait also prevents the deadlock that
  occurred when a new dispatch started before threads exited the inner wait. Without the
  epoch check, threads trapped in `while (task_fn != NULL)` would see the new dispatch's
  non-NULL `task_fn` and stay parked forever — never claiming work, never decrementing
  `task_remaining`, causing the dispatcher to block indefinitely on `cond_done`. The epoch
  check ensures threads always exit the inner wait when a new dispatch arrives. Verified
  with 100+ sequential dispatch stress tests. See `SECURITY_FINDINGS.md` FM-006.

### 4.4 — Parallel Ternary MatMul (Thread Pool Integration) ✅
- File: `include/math/parallel_matmul.h` + `src/math/parallel_matmul.c`
- `void parallel_ternary_matmul(float *out, const float *x, const int8_t *w, int n, int d, float scale, ThreadPool *tp);`
- Wraps the SIMD-dispatched `tn_ternary_matmul` into the thread pool's task interface.
- Each thread processes `d / num_threads` rows (remainder rows go to first threads).
- **Portability fix (FM-003):** Weight pointer offset uses `(size_t)start * (size_t)n`; the
  original `(long)start * n` overflows 32-bit `long` on Windows 64-bit for large matrices.
- NULL pool path falls back to single-threaded execution.

---

## PHASE 5: Tokenizer ✅

### 5.1 — Tokenizer Data Structures ✅
- File: `include/tokenizer/tokenizer.h`
- Implemented:
  ```c
  typedef struct {
      char **vocab;               /* token strings */
      float *vocab_scores;        /* BPE merge priority scores */
      int *sorted_vocab_indices;  /* sorted index for binary search */
      int vocab_size;
      int max_token_len;
      int sorted;                 /* 1 after build_sorted_index() */
  } Tokenizer;
  ```
- Declarations: `tokenizer_load`, `tokenizer_free`, `tokenizer_encode`, `tokenizer_decode`.

### 5.2 — Tokenizer File Loader ✅
- File: `src/tokenizer/tokenizer_load.c`
- Binary format: `[vocab_size(int)] [max_token_len(int)] [score(float) len(int) bytes(char*)]...`
- Validates `vocab_size ≤ 1,000,000` and `max_token_len ≤ 4096`.
- Builds sorted index via portable context-aware quicksort for O(log n) binary search during
  encode. Uses median-of-three pivot + insertion sort for small partitions (≤16 elements).
- **Thread safety (NEW-01 fixed):** Original implementation used a non-reentrant global
  `g_vocab_ptr` for the `qsort` comparator. Replaced with a custom reentrant quicksort
  (`vocab_quicksort`) that passes the vocab pointer as an explicit parameter. The global
  has been completely removed — `tokenizer_load` is now safe for concurrent use.

### 5.3 — BPE Encode ✅
- File: `src/tokenizer/tokenizer_encode.c`
- Step 1: Each byte of the input text is looked up as a single-character token; unknown bytes
  fall back to `<unk>` (token 0 as last resort).
- Step 2: Iterative merge loop — scans all adjacent pairs, finds the one with the highest
  `vocab_scores` entry, merges (shift-left in-place), repeats until no merge possible.
- Returns token count, or -1 if `max_tokens` would be exceeded.

### 5.4 — Token Decode ✅
- File: `src/tokenizer/tokenizer_decode.c`
- `tokenizer_decode(t, prev_token, token)`: maps token ID to its string.
- Handles raw-byte tokens (`<0x0A>` → `\n`).
- Strips leading space when decoding the first token after BOS.

---

## PHASE 6: Transformer Forward Pass ✅

### 6.1 — Token Embedding Lookup ✅
- File: `include/transformer/embedding.h` + `src/transformer/embedding.c`
- `void embed_token(float *out, int token, const int8_t *embedding_table, int dim, float scale);`
- Copies `dim` int8_t values from `embedding_table[token * dim]`, dequantizes to float using scale.

### 6.2 — Multi-Head Attention (Single Layer) ✅
- File: `include/transformer/attention.h` + `src/transformer/attention.c`
- `void attention_forward(RunState *s, const TransformerWeights *w, const Config *p, int layer, int pos, ThreadPool *tp);`
- Steps:
  1. RMSNorm the input (`s->x` → `s->xb`).
  2. Compute Q = matmul(`s->xb`, `w->wq[layer]`).
  3. Compute K = matmul(`s->xb`, `w->wk[layer]`).
  4. Compute V = matmul(`s->xb`, `w->wv[layer]`).
  5. Apply RoPE to Q and K.
  6. Store K and V into the **transposed** KV cache at `[layer][kv_head][pos][:]`.
     - Uses `KV_CACHE_IDX(layer, kv_head, pos, 0)` to compute the write offset.
     - Write is a single `memcpy` of `head_dim` floats into a contiguous slot.
  7. For each attention head:
     a. Resolve KV head: `kv_head = head / (n_heads / n_kv_heads)` (GQA mapping).
     b. Compute attention scores: `score[t] = dot(Q_head, K_cached[layer][kv_head][t][:]) / sqrt(head_dim)` for all `t <= pos`.
        - **Performance note:** Because the KV cache is transposed `[layer][head][position][dim]`, all `pos+1` Key vectors for this head are **contiguous in memory**. The dot-product loop is a linear sweep — no cache misses, full SIMD throughput. This is the critical difference vs. naive `[layer][pos][head][dim]` layout where each Key would be `n_heads * head_dim` floats apart.
     c. Softmax over scores.
     d. Weighted sum of V_cached `[layer][kv_head][0..pos][:]` → `s->xb`. Same contiguity benefit.
  8. Output projection: matmul(`s->xb`, `w->wo[layer]`) → `s->xb2`.
  9. Residual add: `s->x += s->xb2`.

### 6.3 — Grouped Query Attention (GQA) Support ✅
- File: part of `src/transformer/attention.c`
- When `n_kv_heads < n_heads`: multiple Q heads share the same K/V head.
- `int kv_head = head / (n_heads / n_kv_heads);` maps Q head index to KV head index.
- Affects KV cache indexing and RoPE application.

### 6.4 — Feed-Forward Network (Single Layer) ✅
- File: `include/transformer/ffn.h` + `src/transformer/ffn.c`
- `void ffn_forward(RunState *s, const TransformerWeights *w, const Config *p, int layer, ThreadPool *tp);`
- Steps (SwiGLU variant):
  1. RMSNorm input (`s->x` → `s->xb`).
  2. Gate: `s->hb = matmul(s->xb, w->w1[layer])` — apply SiLU activation.
  3. Up: `s->hb2 = matmul(s->xb, w->w3[layer])`.
  4. Element-wise multiply: `s->hb = s->hb * s->hb2`.
  5. Down: `s->xb = matmul(s->hb, w->w2[layer])`.
  6. Residual add: `s->x += s->xb`.

### 6.5 — Full Forward Pass (Single Token) ✅
- File: `include/transformer/forward.h` + `src/transformer/forward.c`
- `float *transformer_forward(int token, int pos, Config *p, TransformerWeights *w, RunState *s, ThreadPool *tp);`
- Steps:
  1. Embed token → `s->x`.
  2. For each layer `l` in `[0, n_layers)`:
     a. `attention_forward(s, w, p, l, pos, tp);`
     b. `ffn_forward(s, w, p, l, tp);`
  3. Final RMSNorm on `s->x`.
  4. Classifier matmul: `logits = matmul(s->x, w->wcls)` → `s->logits`.
  5. Return `s->logits`.

---

## PHASE 7: Sampling & Text Generation ✅

### 7.1 — Argmax Sampler ✅
- File: `include/sampling/sampling.h` + `src/sampling/argmax.c`
- `int sample_argmax(const float *logits, int vocab_size);`
- Returns index of maximum logit. Deterministic, greedy decoding.

### 7.2 — Temperature Scaling ✅
- File: `src/sampling/temperature.c`
- `void apply_temperature(float *logits, int vocab_size, float temperature);`
- Divides all logits by temperature before softmax.

### 7.3 — Top-p (Nucleus) Sampling ✅
- File: `src/sampling/top_p.c`
- `int sample_top_p(float *logits, int vocab_size, float top_p, unsigned long long *rng_state);`
- Sort logits descending, accumulate softmax probabilities until sum >= top_p, sample from truncated distribution.
- Uses xorshift128+ RNG for speed.

### 7.4 — Top-k Sampling ✅
- File: `src/sampling/top_k.c`
- `int sample_top_k(float *logits, int vocab_size, int k, unsigned long long *rng_state);`
- Partial sort to find top-k logits, softmax over them, sample.

### 7.5 — RNG Utility ✅
- File: `include/sampling/rng.h` + `src/sampling/rng.c`
- `void rng_seed(unsigned long long *state, unsigned long long seed);`
- `float rng_float(unsigned long long *state);` — Returns uniform [0,1).
- Xorshift128+ implementation. No external dependency.

### 7.6 — Text Generation Loop ✅
- File: `include/transformer/generate.h` + `src/transformer/generate.c`
- `void generate(Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, ThreadPool *tp, const char *prompt, int max_tokens, float temperature, float top_p);`
- Steps:
  1. Encode prompt → token array.
  2. For each position:
     a. `transformer_forward(token, pos, ...)`.
     b. If still in prompt: next token = prompt[pos+1]. Else: sample from logits.
     c. Decode and print token.
     d. Break on EOS token.
  3. Print tokens/second metric.

---

## PHASE 8: KV Cache Optimizations ✅

### 8.1 — KV Cache Quantization (float32 → int8) ✅
- File: `include/kv_cache/kv_compress.h` + `src/kv_cache/kv_compress.c`
- `CompressedKVCache` struct: `{ int8_t *data; float scale; }`.
- `void kv_compress_to_8bit(const float *src, CompressedKVCache *dst, int dim);` — absmax quantization.
- `void kv_decompress_from_8bit(float *dst, const CompressedKVCache *src, int dim);` — multiply by scale.

### 8.2 — KV Cache Quantization (float32 → int4) ✅
- File: `src/kv_cache/kv_compress_4bit.c`
- Packs two 4-bit values per byte.
- `void kv_compress_to_4bit(const float *src, uint8_t *dst, float *scale, int dim);`
- `void kv_decompress_from_4bit(float *dst, const uint8_t *src, float scale, int dim);`
- 4x memory reduction vs. float32.

### 8.3 — Sliding Window Attention Manager ✅
- File: `include/kv_cache/sliding_window.h` + `src/kv_cache/sliding_window.c`
- `typedef struct { int window_size; int system_prompt_len; int write_head; bool wrapped; } SlidingWindow;`
- `void sw_init(SlidingWindow *sw, int window_size, int system_prompt_len);`
- `int sw_map_position(SlidingWindow *sw, int logical_pos);` — maps logical token position to physical KV cache index (circular buffer).
- `void sw_advance(SlidingWindow *sw);` — moves write head, handles wraparound.
- System prompt tokens (indices 0..system_prompt_len-1) are **pinned** and never evicted.

### 8.4 — Adaptive KV Cache Strategy Selector ✅
- File: `include/kv_cache/kv_strategy.h` + `src/kv_cache/kv_strategy.c`
- `typedef enum { KV_FULL_F32, KV_QUANT_I8, KV_QUANT_I4, KV_SLIDING_I8, KV_SLIDING_I4 } KVStrategy;`
- `KVStrategy select_kv_strategy(const Config *cfg, int64_t free_ram);`
  - free_ram > 16GB → `KV_FULL_F32`
  - free_ram > 4GB → `KV_QUANT_I8`
  - free_ram > 1GB → `KV_SLIDING_I8`
  - free_ram <= 1GB → `KV_SLIDING_I4`
- Returns strategy + computed `max_seq_len`.

---

## PHASE 9: Hidden Reasoning Engine ✅

### 9.1 — Think-Tag State Machine ✅
- File: `include/reasoning/thought_filter.h` + `src/reasoning/thought_filter.c`
- `typedef enum { FILTER_PASSTHROUGH, FILTER_BUFFERING_OPEN_TAG, FILTER_THINKING, FILTER_BUFFERING_CLOSE_TAG, FILTER_OUTPUT } FilterState;`
- `typedef struct { FilterState state; char tag_buffer[16]; int tag_pos; int think_token_count; } ThoughtFilter;`
- `void thought_filter_init(ThoughtFilter *f);`
- `bool thought_filter_process(ThoughtFilter *f, const char *token_str, char *output_buf);`
  - Returns `true` if `output_buf` contains text to print to user.
  - Returns `false` if the token was consumed (hidden thinking).
  - Handles partial tag matches (e.g., `<` then `think` then `>` arriving as separate tokens).

### 9.2 — Reasoning Prompt Injector ✅
- File: `include/reasoning/prompt_inject.h` + `src/reasoning/prompt_inject.c`
- `void inject_reasoning_prompt(char *output_prompt, const char *user_prompt, int max_len);`
- Appends `"\nThink step-by-step inside <think> and </think> tags before answering."` to the user's prompt.
- Respects buffer bounds. Does NOT modify original prompt.

### 9.3 — Reasoning-Aware Generation Wrapper ✅
- File: `src/reasoning/reasoning_generate.c`
- `void generate_with_reasoning(Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, ThreadPool *tp, const char *prompt, int max_tokens, float temperature, float top_p);`
- Wraps the standard `generate()` loop with:
  1. Prompt injection.
  2. ThoughtFilter on every decoded token.
  3. Prints "Thinking..." indicator while `FILTER_THINKING`.
  4. Tracks and optionally reports think-token count for diagnostics.

---

## PHASE 10: Weight Packing & Conversion Tools

### 10.1 — Ternary Weight Packer (2-bit Packing)
- File: `tools/pack_ternary.py`
- Input: A directory of per-layer `.npy` or safetensors weight files (values in {-1, 0, 1}).
- Packing scheme: 4 ternary values per byte. Mapping: -1 → 0b00, 0 → 0b01, 1 → 0b10.
- Output: Single `.bin` file with header:
  ```
  [magic_number: 4 bytes "TNRY"]
  [version: 4 bytes]
  [Config struct: sizeof(Config) bytes]
  [scale_mode: 1 byte — 0 = per-matrix, 1 = per-group]
  [group_size: 4 bytes — e.g., 128 (only meaningful if scale_mode == 1)]
  [scale factors: variable — see below]
  [packed weights: variable]
  ```
- **Scale factor granularity** *(Critical for quantization fidelity)*:
  - **Per-matrix scales** (`scale_mode=0`): One `float` per weight matrix. Simple, works for
    models quantized uniformly. Total: `n_matrices * sizeof(float)` bytes.
  - **Per-group scales** (`scale_mode=1`): One `float` per `group_size` weights within each
    matrix. Many aggressively quantized models (e.g., BitNet b1.58 variants, GPTQ-style)
    store scale factors every 64–128 weights to preserve per-channel variance. Total:
    `sum_over_matrices(ceil(matrix_elements / group_size)) * sizeof(float)` bytes.
  - The converter (Phase 10.2) must detect which scheme the source model uses and set
    `scale_mode` accordingly. The C runtime matmul kernels read `scale_mode` at model load
    and select the appropriate accumulation strategy:
    - Per-matrix: `out[i] = scale * sum_j(...)` (current implementation)
    - Per-group: `out[i] = sum_g( scale[g] * sum_{j in group_g}(...) )`
  - The AVX2 kernel (Phase 3.3) and fused packed kernel (Phase 10.5) must both support
    per-group accumulation. The natural boundary is the SIMD register width (8 floats for
    AVX2), so `group_size` should be a multiple of 8 for optimal alignment.

### 10.2 — HuggingFace SafeTensors → Binary Converter
- File: `tools/convert_hf_bitnet.py`
- Reads a HuggingFace BitNet model (safetensors format).
- Extracts `config.json` → writes `Config` struct.
- For each weight tensor: validates values ∈ {-1, 0, 1}, extracts scale, packs ternary.
- Handles layer name mapping: `model.layers.0.self_attn.q_proj.weight` → `wq[0]`.
- CLI: `python convert_hf_bitnet.py --model microsoft/bitnet-b1.58-2B-4T --output model.bin`

### 10.3 — Tokenizer Converter
- File: `tools/convert_tokenizer.py`
- Reads HuggingFace `tokenizer.json` or SentencePiece `.model`.
- Writes binary vocab file matching the format expected by `tokenizer_load()`.
- CLI: `python convert_tokenizer.py --input tokenizer.json --output tokenizer.bin`

### 10.4 — 2-bit Packed Weight Unpacker (C Runtime) *(Critical Fix: Unpack Overhead)*
- File: `include/core/unpack.h` + `src/core/unpack.c`

**The Problem:** If the forward-pass matmul loop calls a per-element `unpack_ternary()` function, the bit-shifting overhead (`>>`, `& 0x3`, subtract-to-sign) per weight dominates the actual add/subtract math. At dim=4096, that's 4096 shift+mask operations per output row — slower than the additions they gate.

**Scalar fallback (for correctness testing only):**
- `int8_t unpack_ternary(const uint8_t *packed, int index);` — Extracts single ternary value. `val = (packed[index/4] >> (2*(index%4))) & 0x3; return val - 1;` (maps 0b00→-1, 0b01→0, 0b10→1).
- `void unpack_ternary_block(int8_t *out, const uint8_t *packed, int count);` — Loop calling above. **Reference only.**

**SIMD Shuffle LUT (the real implementation):**
- File: `src/core/unpack_avx2.c` (guarded by `#ifdef __AVX2__`)
- **Precomputed LUT:** A 16-entry lookup table mapping every possible 2-bit pair pattern within a nibble to the corresponding ternary values. Loaded into an `__m128i` register once at init.
- `void unpack_ternary_block_avx2(int8_t *out, const uint8_t *packed, int count);`
  - Loads 32 packed bytes (= 128 ternary weights) into a `__m256i`.
  - Splits into low and high nibbles using `_mm256_and_si256` / `_mm256_srli_epi16`.
  - Applies `_mm256_shuffle_epi8` with the LUT — this single instruction "explodes" each nibble into 4 separate ternary int8 values **in one clock cycle**.
  - Stores 128 unpacked `int8_t` values.
  - Processes 128 weights per iteration vs. 1 per scalar call — **128x throughput improvement**.
- File: `src/core/unpack_avx512.c` (guarded by `#ifdef __AVX512BW__`)
  - Same approach with `_mm512_shuffle_epi8`, processes 256 weights per iteration.
- File: `src/core/unpack_neon.c` (guarded by `#ifdef __ARM_NEON`)
  - Uses `vtbl1_u8` (NEON table lookup) for the same LUT trick, 64 weights per iteration.

### 10.5 — Packed-Weight Ternary MatMul (Fused Unpack+Compute)
- File: `include/math/ternary_matmul_packed.h` + `src/math/ternary_matmul_packed.c`
- **Why this module exists:** Even the LUT unpack from 10.4 has overhead if you unpack an entire row into a temporary `int8_t[]` buffer, then loop over that buffer for the add/subtract math. The optimal path fuses unpack + accumulate into a single pass.
- `void ternary_matmul_packed(float *out, const float *x, const uint8_t *packed_w, int n, int d, float scale);`
- **AVX2 fused kernel:**
  1. Load 32 packed bytes (128 weights).
  2. Shuffle-LUT unpack into four `__m256i` registers of int8 ternary values.
  3. For each group of 8 weights: use `_mm256_blendv_ps` with sign masks to conditionally add/subtract the corresponding 8 input floats.
  4. Accumulate into `__m256` float accumulator.
  5. Horizontal sum at end of row.
- This eliminates the intermediate `int8_t *` buffer entirely. The packed weights go straight from the mmap'd file into the SIMD pipeline.
- **Fallback:** For non-SIMD paths, unpacks a row into a stack buffer, then calls `ternary_matmul()` scalar.

---

## PHASE 11: Multimodal — Vision Encoder Bridge

### 11.1 — Image Loader (STB)
- File: `src/multimodal/image_load.c`
- Bundles `stb_image.h` (single-header, public domain).
- `TernaryError load_image(const char *path, float **pixels, int *width, int *height);`
- Loads any format (PNG, JPEG, BMP), converts to float RGB [0,1], resizes to encoder's expected resolution (e.g., 384x384) using bilinear interpolation.

### 11.2 — Image Patch Extractor
- File: `include/multimodal/patch_extract.h` + `src/multimodal/patch_extract.c`
- `void extract_patches(const float *image, float *patches, int img_size, int patch_size, int *num_patches);`
- Slices `img_size x img_size x 3` image into `(img_size/patch_size)^2` patches of size `patch_size * patch_size * 3`.
- Flattens each patch into a 1D vector.

### 11.3 — Vision Encoder Weights & Forward
- File: `include/multimodal/vision_encoder.h` + `src/multimodal/vision_encoder.c`
- Minimal ViT (Vision Transformer) forward pass:
  1. Linear projection of flattened patches → patch embeddings.
  2. Add positional embeddings (learned, loaded from weights file).
  3. N transformer blocks (attention + FFN) — reuses our existing math primitives.
  4. Output: `num_patches` vectors of `embed_dim` each.
- Uses the same ternary matmul if vision weights are also quantized; falls back to float matmul if not.

### 11.4 — MLP Vision Projector *(Critical Fix: Dimension Mismatch)*
- File: `include/multimodal/vision_projector.h` + `src/multimodal/vision_projector.c`
- **Why this exists:** A Vision Encoder (SigLIP, CLIP) typically outputs patch embeddings of dimension 768 or 1152. The LLM's text embedding dimension is typically 2048 or 4096. You **cannot** `memcpy` a 768-float vector into a 4096-float slot — the remaining 3278 floats would be uninitialized garbage, causing the attention mechanism to produce nonsensical outputs. Every multimodal model (LLaVA, InternVL, Qwen-VL) trains a dedicated projection layer to bridge this gap.
- **Data structures:**
  ```c
  typedef struct {
      int vision_dim;    // e.g., 768 (SigLIP output)
      int llm_dim;       // e.g., 4096 (LLM input)
      int hidden_dim;    // e.g., 4096 (projector intermediate — typically == llm_dim)
      float *w_down;     // vision_dim × hidden_dim weight matrix
      float *w_up;       // hidden_dim × llm_dim weight matrix
      float *bias_down;  // hidden_dim bias (optional, set NULL if absent)
      float *bias_up;    // llm_dim bias (optional)
  } VisionProjector;
  ```
- **Functions:**
  - `TernaryError vision_projector_load(VisionProjector *proj, const void *mapped_ptr, size_t *offset);` — Reads projector weights from the weight file (stored after the vision encoder weights).
  - `void vision_projector_forward(float *out, const float *patch_embedding, const VisionProjector *proj, ThreadPool *tp);`
    - Two-layer MLP: `hidden = GELU(patch @ w_down + bias_down)` → `out = hidden @ w_up + bias_up`.
    - Uses the standard float matmul (projector weights are typically NOT ternary-quantized — they are small enough that keeping them in float16/float32 adds negligible memory cost while preserving vision-text alignment quality).
  - `void vision_projector_forward_batch(float *out, const float *patches, int num_patches, const VisionProjector *proj, ThreadPool *tp);` — Projects all patches in one call. Output: `num_patches × llm_dim` float array.
- **Note on weight source:** When using `tools/convert_hf_bitnet.py` (Phase 10.2), the converter must also extract `model.mm_projector.*` weights from the HuggingFace model and append them to the `.bin` file. The projector section is identified by a separate offset stored in the file header.

### 11.5 — Vision-to-LLM Injection Bridge
- File: `include/multimodal/vision_bridge.h` + `src/multimodal/vision_bridge.c`
- `VisionContext` struct: `{ int num_patches; int embed_dim; float *patch_embeddings; }` — **`embed_dim` here is `llm_dim`** (post-projection), NOT the raw vision encoder output dimension.
- `void inject_vision_into_kv_cache(RunState *s, const Config *p, const TransformerWeights *w, const VisionContext *v, ThreadPool *tp);`
  - **Precondition:** `v->patch_embeddings` have already been passed through the Vision Projector (Phase 11.4). Each patch vector is now exactly `p->dim` floats — dimension-matched to the LLM.
  - Loops over each projected patch embedding.
  - Copies patch vector into `s->x`.
  - Runs a full `transformer_forward()` pass (populating KV cache).
  - Advances `s->current_pos`.
- After injection, text tokens appended to the same KV cache continue normally.
- **Full multimodal pipeline:** Image → `load_image()` (11.1) → `extract_patches()` (11.2) → `vision_encoder_forward()` (11.3) → **`vision_projector_forward_batch()`** (11.4) → `inject_vision_into_kv_cache()` (11.5) → text generation continues.

---

## PHASE 12: CLI & Main Entry Point

### 12.1 — Argument Parser
- File: `include/cli/args.h` + `src/cli/args.c`
- `typedef struct { char *model_path; char *tokenizer_path; char *prompt; char *image_path; float temperature; float top_p; int max_tokens; int num_threads; bool enable_reasoning; bool verbose; int seed; } CliArgs;`
- `TernaryError parse_args(CliArgs *args, int argc, char **argv);`
- Supports: `--model`, `--tokenizer`, `--prompt`, `--image`, `--temperature`, `--top-p`, `--max-tokens`, `--threads`, `--reasoning`, `--verbose`, `--seed`.
- Defaults: temperature=0.7, top_p=0.9, max_tokens=512, threads=auto, reasoning=off.

### 12.2 — Interactive REPL Mode
- File: `src/cli/repl.c`
- `void run_repl(Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, ThreadPool *tp, CliArgs *args);`
- Infinite loop:
  1. Print `> ` prompt.
  2. Read line from stdin (handles multi-line with backslash continuation).
  3. If line starts with `/quit` → break.
  4. If line starts with `/image <path>` → load and inject image, then continue.
  5. If line starts with `/think` → toggle reasoning mode.
  6. If line starts with `/context` → print current KV cache usage stats.
  7. Generate response, print tokens as they are decoded.
  8. Print `[tokens/sec]` after generation completes.

### 12.3 — Boot Sequence & Main
- File: `src/cli/main.c`
- `int main(int argc, char **argv)`:
  1. Parse CLI args.
  2. Print banner with version.
  3. **Hardware probe:** `get_optimal_thread_count()`, `get_free_ram_bytes()`.
  4. Print detected hardware: `"Detected: X cores, Y MB free RAM"`.
  5. **Create thread pool.**
  6. **Load model:** `mapped_file_open()` → `config_read_from_file()` → `alloc_weight_pointers()` → `memory_map_weights()`.
  7. **Compute memory budget:** `compute_mem_budget()` → `select_kv_strategy()`.
  8. Print strategy: `"KV Strategy: QUANT_I8, max context: 4096 tokens"`.
  9. **Allocate RunState** with computed `max_seq_len`.
  10. **Load tokenizer:** `tokenizer_load()`.
  11. If `--prompt` provided: single-shot generation. Else: enter REPL.
  12. **Cleanup:** free RunState, free weight pointers, close mapped file, destroy thread pool, free tokenizer.
  13. Return 0.

### 12.4 — Performance Timer Utility
- File: `include/cli/timer.h` + `src/cli/timer.c`
- `int64_t timer_now_us();` — `clock_gettime(CLOCK_MONOTONIC)` on POSIX, `QueryPerformanceCounter` on Windows.
- `double timer_tokens_per_sec(int64_t start_us, int64_t end_us, int token_count);`

---

## PHASE 13: Testing & Validation

### 13.1 — Minimal Test Harness
- File: `tests/test_harness.h`
- Macros: `TEST_ASSERT(cond, msg)`, `TEST_ASSERT_FLOAT_EQ(a, b, eps)`, `RUN_TEST(fn)`.
- No external test framework. Pure C.

### 13.2 — Unit Test: Config Read
- File: `tests/test_config.c`
- Creates a synthetic config blob in memory, validates `config_read_from_file()` parses correctly.
- Tests invalid configs (negative dim, zero layers) return `ERR_INVALID_CONFIG`.

### 13.3 — Unit Test: Ternary MatMul Correctness
- File: `tests/test_matmul.c`
- Creates small known weight matrices (e.g., 4x4) with hand-computed expected outputs.
- Runs scalar reference, branchless, and (if available) SIMD variants.
- Asserts all variants produce identical results within float epsilon.

### 13.4 — Unit Test: RMSNorm
- File: `tests/test_rmsnorm.c`
- Known input/output pairs. Validates numerical stability with very small and very large inputs.

### 13.5 — Unit Test: Softmax
- File: `tests/test_softmax.c`
- Validates sum-to-1 property. Tests with extreme values (large positives/negatives).

### 13.6 — Unit Test: RoPE
- File: `tests/test_rope.c`
- Validates rotation is reversible. Validates frequency pattern matches known reference.

### 13.7 — Unit Test: KV Cache Compression
- File: `tests/test_kv_compress.c`
- Round-trip test: compress float32 → int8 → decompress → compare with original.
- Measures max absolute error, asserts < 1% relative error.

### 13.8 — Unit Test: Sliding Window
- File: `tests/test_sliding_window.c`
- Validates circular buffer wrapping. Validates system prompt pinning. Validates position mapping after wraparound.

### 13.9 — Unit Test: Tokenizer
- File: `tests/test_tokenizer.c`
- Encode/decode round-trip on known strings. Tests UTF-8 handling. Tests special tokens.

### 13.10 — Unit Test: Thought Filter
- File: `tests/test_thought_filter.c`
- Feeds token sequences containing `<think>...</think>` blocks.
- Validates only non-thinking tokens are output.
- Tests partial tag arrival (e.g., `<` then `think>` as separate tokens).

### 13.11 — Unit Test: Thread Pool
- File: `tests/test_thread_pool.c`
- Dispatches a simple summation task across threads.
- Validates result matches single-threaded computation.
- Tests with 1 thread, 2 threads, and max threads.

### 13.12 — Unit Test: Memory Mapping
- File: `tests/test_mmap.c`
- Creates a temporary file, writes known data, maps it, reads back, validates.
- Tests cleanup (munmap + close).

### 13.15 — Unit Test: Aligned Allocator
- File: `tests/test_aligned_alloc.c`
- Validates that `tn_aligned_alloc(size, 64)` returns a pointer where `((uintptr_t)ptr % 64) == 0`.
- Tests multiple sizes (small, odd, large). Tests `tn_aligned_free` doesn't crash.
- Validates `tn_aligned_calloc` returns zero-filled memory.

### 13.16 — Unit Test: KV Cache Transposed Layout
- File: `tests/test_kv_layout.c`
- Allocates a small KV cache (2 layers, 4 heads, 8 positions, dim=16).
- Writes known values at specific `[layer][head][pos][dim]` coordinates.
- Reads back via `KV_CACHE_IDX` macro, validates correctness.
- Validates that iterating `pos=0..7` for a fixed `[layer][head]` produces contiguous memory addresses (pointer arithmetic check).

### 13.17 — Unit Test: SIMD Shuffle LUT Unpacker
- File: `tests/test_unpack_lut.c`
- Packs a known sequence of ternary values (-1, 0, 1, 1, 0, -1, ...) into 2-bit packed format.
- Unpacks via scalar `unpack_ternary_block()` and (if available) `unpack_ternary_block_avx2()`.
- Asserts both produce identical output.
- Tests edge cases: all-zeros byte, all-ones byte, all-negative byte.
- Benchmarks: measures throughput of LUT unpack vs. scalar unpack at n=4096.

### 13.18 — Unit Test: Packed-Weight Fused MatMul
- File: `tests/test_packed_matmul.c`
- Creates small matrices in both unpacked `int8_t` and packed `uint8_t` formats.
- Runs `ternary_matmul()` (scalar, unpacked) and `ternary_matmul_packed()` (fused, packed).
- Asserts outputs match within float epsilon.

### 13.19 — Unit Test: Vision Projector
- File: `tests/test_vision_projector.c`
- Creates a tiny projector (vision_dim=8, hidden_dim=16, llm_dim=32) with known weights.
- Runs `vision_projector_forward()` on a known input patch.
- Validates output dimension is `llm_dim` (not `vision_dim`).
- Validates output values against hand-computed expected results.

### 13.20 — Unit Test: Tool Tag Interceptor
- File: `tests/test_tool_interceptor.c`
- Feeds token streams containing `<exec>ls -la</exec>`, `<save_memory>user prefers Python</save_memory>`.
- Validates correct `TagType` returned at tag close. Validates `content_buffer` contains the inner text.
- Tests interleaved tags: `<think>reasoning</think> hello <exec>pwd</exec> world`.
- Tests malformed/partial tags don't crash (graceful fallback to passthrough).

### 13.21 — Unit Test: Command Executor Security
- File: `tests/test_cmd_exec.c`
- Tests `exec_policy_check()` blocks `rm -rf /`, `sudo su`, `shutdown`, fork bombs.
- Tests allowlisted commands (`ls`, `python`, `cat`) pass the check.
- Tests `exec_run_command()` captures stdout from a simple `echo hello` command.
- Tests timeout: runs `sleep 999`, asserts it's killed after `timeout_seconds`.

### 13.22 — Unit Test: Cosine Similarity
- File: `tests/test_cosine.c`
- Tests identical vectors → similarity = 1.0.
- Tests orthogonal vectors → similarity = 0.0.
- Tests opposite vectors → similarity = -1.0.
- Tests SIMD variant matches scalar within float epsilon.

### 13.23 — Unit Test: Vector DB Store & Search
- File: `tests/test_vector_db.c`
- Creates a temporary DB file. Stores 5 known embeddings + texts.
- Searches with a query vector. Validates top-1 result matches expected entry.
- Tests deduplication: stores same embedding twice, asserts DB size doesn't grow.
- Tests persistence: closes and reopens DB, validates entries survive.

### 13.24 — Integration Test: Agentic Loop
- File: `tests/test_agent_e2e.c`
- Uses a mock `exec_run_command()` that returns canned responses.
- Feeds a prompt that triggers `<exec>echo test</exec>`.
- Validates the mock output is injected into the KV cache.
- Validates the generation resumes and produces further tokens after injection.

### 13.13 — Integration Test: End-to-End Inference
- File: `tests/test_e2e.c`
- Uses a tiny synthetic model (2 layers, dim=64, vocab=256) with random ternary weights.
- Runs a full forward pass. Doesn't validate "intelligence" — just validates no crashes, no memory leaks (when run under AddressSanitizer), and output logits sum to ~1 after softmax.

### 13.14 — Benchmark: Tokens Per Second
- File: `tests/bench_matmul.c`
- Benchmarks `ternary_matmul` at realistic sizes (dim=4096, hidden=14336).
- Compares scalar vs. branchless vs. SIMD variants.
- Reports GOPS (Giga-Operations Per Second).

---

## PHASE 14: Agentic Tool Execution ✅

An "Agent" is an LLM hooked up to a `while` loop with permission to trigger external tools. Because we wrote the engine from scratch in C, we extend the State Machine from Phase 9 (the Hidden Thought Loop) to intercept tool-call tags and execute OS commands.

### 14.1 — Tool Tag Interceptor (State Machine Extension)
- File: `include/agent/tool_interceptor.h` + `src/agent/tool_interceptor.c`
- **Why this exists:** The reasoning engine (Phase 9) already intercepts `<think>` tags. The agentic interceptor extends the same state machine to also recognize `<exec>`, `<save_memory>`, and `<search_memory>` tags — routing each to a different handler.
- `typedef enum { TAG_NONE, TAG_THINK, TAG_EXEC, TAG_SAVE_MEMORY, TAG_SEARCH_MEMORY } TagType;`
- `typedef struct { TagType active_tag; char tag_buffer[16]; int tag_pos; char content_buffer[4096]; int content_pos; bool inside_tag; } ToolInterceptor;`
- `void tool_interceptor_init(ToolInterceptor *ti);`
- `TagType tool_interceptor_feed(ToolInterceptor *ti, const char *token_str, char *passthrough_buf);`
  - Returns `TAG_NONE` + fills `passthrough_buf` if token is normal text (print to user).
  - Returns `TAG_THINK` while inside `<think>` (suppress output, same as Phase 9).
  - Returns `TAG_EXEC` when `</exec>` is detected — `ti->content_buffer` contains the full command string.
  - Returns `TAG_SAVE_MEMORY` when `</save_memory>` is detected — content is the fact to store.
  - Returns `TAG_SEARCH_MEMORY` when `</search_memory>` is detected — content is the query.
- Handles partial tag matches, nested content, and malformed tags gracefully.

### 14.2 — Secure Command Executor
- File: `include/agent/cmd_exec.h` + `src/agent/cmd_exec.c`
- **Security model:** The AI can request OS commands, but the engine enforces safety.
- `typedef struct { char **allowed_commands; int num_allowed; char **blocked_patterns; int num_blocked; bool require_user_approval; int timeout_seconds; } ExecPolicy;`
- `TernaryError exec_policy_load(ExecPolicy *pol, const char *config_path);` — Loads allowlist/blocklist from a JSON or plain-text config file.
- `bool exec_policy_check(const ExecPolicy *pol, const char *command);` — Returns `true` only if command passes all checks:
  - Not in blocklist (default blocks: `rm -rf /`, `sudo`, `mkfs`, `dd if=`, `shutdown`, `:(){ :|:& };:`).
  - Optionally: must match a prefix in the allowlist (e.g., only `python`, `ls`, `cat`, `grep`, `curl` allowed).
- `TernaryError exec_run_command(const ExecPolicy *pol, const char *command, char *output_buf, int output_buf_size, int *exit_code);`
  - POSIX: `popen(command, "r")` → reads stdout into `output_buf` → `pclose()` captures exit code.
  - Windows: `_popen()` equivalent.
  - Enforces timeout: uses `alarm()` (POSIX) or thread-based timeout (Windows) to kill runaway processes after `pol->timeout_seconds`.
  - Truncates output to `output_buf_size` if the command produces excessive output.

### 14.3 — Command Output → KV Cache Injection
- File: `include/agent/output_inject.h` + `src/agent/output_inject.c`
- `void inject_exec_result_into_context(RunState *s, const Config *p, const TransformerWeights *w, Tokenizer *t, ThreadPool *tp, const char *result_text, int max_inject_tokens);`
  - Tokenizes `result_text` into token IDs.
  - Truncates to `max_inject_tokens` if the output is too long (prevents a single `cat bigfile.txt` from consuming the entire KV cache).
  - Runs each token through `transformer_forward()` to populate the KV cache.
  - Advances `s->current_pos` accordingly.
  - After injection, the AI's next generated token will see the command output in its context as if it had "read" it.
- **Wrapping format:** Injects as `\n<result>\n{output}\n</result>\n` so the AI knows where OS output starts/ends.

### 14.4 — Agentic Generation Loop
- File: `include/agent/agent_loop.h` + `src/agent/agent_loop.c`
- `void generate_agentic(Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, ThreadPool *tp, const char *prompt, int max_tokens, float temperature, float top_p, ExecPolicy *pol, VectorDB *db);`
- **The core agentic `while` loop:**
  1. Inject system prompt: `"You have tools: <exec>command</exec> to run shell commands, <save_memory>fact</save_memory> to remember facts, <search_memory>query</search_memory> to recall facts."`
  2. Generate tokens one at a time.
  3. Feed each token to `tool_interceptor_feed()`.
  4. **If `TAG_EXEC`:**
     a. Extract command from `content_buffer`.
     b. `exec_policy_check()` — if blocked, inject `"<result>ERROR: Command blocked by security policy.</result>"` into KV cache.
     c. If `require_user_approval`: print command to user, wait for y/n on stdin.
     d. `exec_run_command()` → capture output.
     e. `inject_exec_result_into_context()` → shove result into KV cache.
     f. **Resume generation** — the AI sees the result and continues.
  5. **If `TAG_SAVE_MEMORY`:** Call `vector_db_store()` (Phase 15.3).
  6. **If `TAG_SEARCH_MEMORY`:** Call `vector_db_search()` (Phase 15.4), inject results into context.
  7. **If `TAG_THINK`:** Suppress output (same as Phase 9).
  8. **If `TAG_NONE`:** Print token to user.
  9. Loop until EOS or `max_tokens`.
- **Max iterations safety:** Caps the number of tool calls per generation (default: 10) to prevent infinite loops.
- **Token budget tracking:** Each tool injection consumes KV cache space. If remaining capacity < 256 tokens, stop accepting tool calls and force the AI to output a final answer.

### 14.5 — User Approval Prompt (Interactive Safety)
- File: `src/agent/user_approval.c`
- `bool prompt_user_approval(const char *command);`
  - Prints: `\n[AGENT] Wants to execute: {command}\n  Allow? [y/N]: `
  - Reads single character from stdin (non-buffered where possible).
  - Returns `true` only on explicit `y` or `Y`.
  - Timeout: auto-denies after 30 seconds of no input.
- Integrated into the REPL (Phase 12.2) with a `/auto` command to toggle auto-approve mode for trusted sessions.

---

## PHASE 15: Local RAG — Retrieval-Augmented Generation (Self-Learning Memory) ✅

Since we cannot change the AI's weights at runtime, we make the AI "learn" by giving it a permanent, searchable hard drive. This is Retrieval-Augmented Generation (RAG).

### 15.1 — Embedding Generator (Text → Vector)
- File: `include/rag/embedder.h` + `src/rag/embedder.c`
- **Why this exists:** To search a Vector DB, every piece of stored text must be converted to a fixed-length float array (an "embedding"). We reuse the LLM itself as the embedder — no second model needed.
- `void generate_embedding(float *out_embedding, int embed_dim, const char *text, Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, ThreadPool *tp);`
  - Tokenizes `text`.
  - Runs all tokens through `transformer_forward()`.
  - **Mean-pooling:** Averages the `s->x` vectors across all token positions to produce a single `dim`-length embedding vector.
  - Normalizes the output vector to unit length (L2 norm = 1.0) for cosine similarity.
- **Scratch RunState:** Uses a separate, small RunState with a limited `max_seq_len` (e.g., 512) to avoid polluting the main conversation's KV cache. Allocated once at boot, reused for all embeddings.

### 15.2 — Cosine Similarity Search
- File: `include/rag/similarity.h` + `src/rag/similarity.c`
- `float cosine_similarity(const float *a, const float *b, int dim);`
  - `dot(a, b) / (norm(a) * norm(b))` — but since we pre-normalize in 15.1, this reduces to just `dot(a, b)`.
  - SIMD-accelerated: uses `_mm256_fmadd_ps` (AVX2) or `vfmaq_f32` (NEON) for the dot product.
- `void find_top_k_similar(const float *query, const float **db_embeddings, int db_size, int dim, int k, int *result_indices, float *result_scores);`
  - Brute-force scan: computes cosine similarity against every stored embedding.
  - Maintains a min-heap of size `k` to track the top-k results.
  - For databases < 100K entries, brute-force is faster than approximate methods (no index overhead).

### 15.3 — Vector Database (File-Backed Storage)
- File: `include/rag/vector_db.h` + `src/rag/vector_db.c`
- **Why file-backed:** We don't use SQLite or any external DB. The vector store is a single binary file on disk, mmap'd for instant access — same philosophy as the model weights.
- `typedef struct { int num_entries; int embed_dim; float *embeddings; char **texts; size_t *text_offsets; MappedFile mf; } VectorDB;`
- **File format:**
  ```
  [magic: 4 bytes "VRDB"]
  [version: 4 bytes]
  [num_entries: 4 bytes]
  [embed_dim: 4 bytes]
  [embeddings: num_entries * embed_dim * sizeof(float)]
  [text_lengths: num_entries * sizeof(uint32_t)]
  [text_data: concatenated null-terminated strings]
  ```
- Functions:
  - `TernaryError vector_db_open(VectorDB *db, const char *path, int embed_dim);` — Opens existing DB or creates new empty file.
  - `TernaryError vector_db_store(VectorDB *db, const float *embedding, const char *text);`
    - Appends embedding + text to the file.
    - Uses `ftruncate()` + `mremap()` (Linux) or re-maps (other platforms) to grow the mapped region.
    - Increments `num_entries`, updates header.
  - `void vector_db_close(VectorDB *db);` — Flush + unmap + close.

### 15.4 — Memory Search & Retrieval
- File: `include/rag/memory_search.h` + `src/rag/memory_search.c`
- `int search_memory(VectorDB *db, const char *query_text, Config *p, TransformerWeights *w, RunState *embed_state, Tokenizer *t, ThreadPool *tp, char **results, float *scores, int max_results);`
  - Generates embedding for `query_text` using Phase 15.1.
  - Calls `find_top_k_similar()` from Phase 15.2.
  - Returns up to `max_results` text strings + similarity scores.
  - Filters: discards results with score < 0.5 (too dissimilar to be useful).

### 15.5 — Auto-Retrieve on Every Prompt (Transparent RAG)
- File: `include/rag/auto_retrieve.h` + `src/rag/auto_retrieve.c`
- `void auto_retrieve_and_inject(RunState *s, const Config *p, const TransformerWeights *w, Tokenizer *t, ThreadPool *tp, VectorDB *db, RunState *embed_state, const char *user_prompt, int max_inject_tokens);`
  - Called automatically before every generation cycle.
  - Embeds `user_prompt`, searches vector DB for top-3 relevant memories.
  - If any results found with score > 0.6:
    - Formats as: `\n<memory>\nRelevant context from previous conversations:\n- {result1}\n- {result2}\n</memory>\n`
    - Injects into KV cache via tokenize → forward pass.
  - **Budget-aware:** Checks remaining KV cache capacity. If < 512 tokens remain, skips injection to preserve space for the actual response.
  - The AI sees these injected memories as part of its context, seamlessly "remembering" past conversations.

### 15.6 — Auto-Save Agent Hook
- File: `src/rag/auto_save.c`
- `void check_and_save_memory(ToolInterceptor *ti, VectorDB *db, Config *p, TransformerWeights *w, RunState *embed_state, Tokenizer *t, ThreadPool *tp);`
  - Called when `tool_interceptor_feed()` returns `TAG_SAVE_MEMORY`.
  - Extracts fact text from `ti->content_buffer`.
  - Generates embedding via Phase 15.1.
  - Deduplication: searches DB for existing entries with cosine similarity > 0.95. If found, skips the save (prevents storing the same fact twice).
  - Stores via `vector_db_store()`.
  - Injects confirmation back into KV cache: `"<result>Memory saved.</result>"`.

---

## PHASE 16-S: SIMD Multi-Architecture Performance Upgrade ✅

*Implemented: 2026-03-17. Resolves the Phase 16 numbering collision:
 this plan's Phase 16 = Docs/Packaging is renamed 16-D below.
 Phase 16-S covers the SIMD upgrade work described in CPU_LLM_TERNARY_ENGINE.md § 16.*

### 16-S.1 — Runtime CPU Feature Detection ✅
- **File:** `include/math/cpu_features.h`, `src/math/cpu_features.c`
- `TnCpuFeatures` struct with booleans for every SIMD tier (x86 + ARM)
- x86: CPUID leaf 7 sub-leaf 0/1 for AVX2, AVX-512F/VNNI/BF16/FP16/VBMI, AVX-VNNI
- ARM Linux: `getauxval(AT_HWCAP/AT_HWCAP2)` for DOTPROD, SVE2, BF16
- ARM macOS: `sysctlbyname("hw.optional.arm.FEAT_DotProd")`
- Cached singleton — `tn_cpu_features_detect()` is free after first call
- `tn_cpu_best_backend_name(f)` returns human-readable tier name

### 16-S.2 — float32 → int8 Quantization Kernel ✅
- **File:** `include/math/quantize_i8.h`, `src/math/quantize_i8.c`
- Scalar, AVX-512, and AVX2 variants
- Dynamic per-vector quantization: `scale = max_abs / 127`
- `sum_i8_avx512()`: horizontal sum using `_mm512_cvtepi8_epi32` in 4 quarters

### 16-S.3 — AVX-512 VNNI Kernel (64 MACs/cycle) ✅
- **File:** `src/math/ternary_matmul_packed_vnni.c`
- Guard: `#if TN_HAS_AVX512VNNI`
- `_mm512_dpbusds_epi32` accumulates 64 int8 MACs per cycle
- w_enc bias trick: `dpbusds(w+1, q_x) - sum_qx = true_dot`
- 64-wide inner loop with `unpack16_to_wenc_u8()` helper
- Scalar tail for n % 64 ≠ 0; fallback to AVX-512F for n > 16384

### 16-S.4 — AVX-VNNI 256-bit Kernel (32 MACs/cycle) ✅
- **File:** `src/math/ternary_matmul_packed_avx_vnni.c`
- Guard: `#if TN_HAS_AVXVNNI`
- `_mm256_dpbusds_epi32` (VEX-encoded; Alder Lake, Zen 3, NO AVX-512)
- Compiled with `-mavxvnni` per-file override in Makefile
- Same w_enc bias trick as AVX-512 VNNI

### 16-S.5 — ARM SDOT Kernel (16 MACs/cycle) ✅
- **File:** `src/math/ternary_matmul_packed_dotprod.c`
- Guard: `#if TN_HAS_ARM_DOTPROD`
- `vdotq_s32(acc, w_i8, q_x_i8)` — signed×signed, no bias correction
- Weights decoded from 2-bit packed to int8 {-1, 0, 1} per row

### 16-S.6 — 6-Tier SIMD Dispatch Rewrite ✅
- **File:** `src/math/simd_dispatch.c`
- `tn_simd_init()` runs runtime detection, sets `tn_ternary_matmul_packed`
  function pointer to best available kernel; returns backend name string
- Priority: AVX-512 VNNI → AVX-VNNI → AVX-512F → AVX2 → ARM DOTPROD → NEON → Scalar
- All compile-time `#if` guards prevent SIGILL on unsupported hardware

### 16-S.7 — Phase K-3 RAM Optimization ✅
- **File:** `src/memory/mapped_file.c`
- `MADV_HUGEPAGE` on Linux (requires `-D_GNU_SOURCE`, per-file Makefile rule)
- `MADV_ZERO_WIRED_PAGES` on macOS
- `POSIX_MADV_WILLNEED + POSIX_MADV_SEQUENTIAL` for weight file prefetch
- `src/kv_cache/kv_strategy.c`: corrected thresholds GB_12→GB_8, GB_10→GB_5
  so 16 GB systems select KV_QUANT_I8 instead of KV_SLIDING_I4

### 16-S.8 — Microbenchmark Tool ✅
- **File:** `tools/bench_simd.c`
- `make bench` builds and runs all available SIMD tiers
- Reports ms/call, GMAC/s, and synthetic tok/s for 2048×2048 matmul
- Measured results on Intel Xeon @ 2.80 GHz:
  - AVX2 float32: 2.95 GMAC/s (~4.9 synthetic tok/s)
  - AVX-512F: 10.83 GMAC/s (~17.9 synthetic tok/s)
  - AVX-512 VNNI: 12.57 GMAC/s (~20.8 synthetic tok/s) — **4.3× vs AVX2**

### 16-S.9 — Test Suite ✅
- **File:** `tests/test_simd_vnni.c`
- 50 tests: quantization, sum_i8, VNNI matmul, AVX-VNNI matmul, CPU detection,
  dispatch selection, KV strategy thresholds
- Makefile probes `/proc/cpuinfo` to add only supported VNNI flags to test binary
  (prevents SIGILL when machine has `avx512_vnni` but not `avx_vnni`)
- All 50 tests pass on Intel Xeon (AVX-512 VNNI, no AVX-VNNI)

---

## PHASE K-4: Prefetch + Batch Decode + Barrier + Quantisation Cache ✅

*Implemented: 2026-03-19. Performance references: Addenda P, Q, V, W, X.*

### K-4.1 — Full-Row Prefetch in VNNI Kernel ✅
- **File:** `src/math/ternary_matmul_packed_vnni.c`
- Added `__builtin_prefetch(w_row + 64, 0, 1)` with stride equal to packed row size
- Overlaps next-row weight fetch with current-row arithmetic
- Effect: +1–2 tok/s on Tiger Lake (L3→DRAM latency hiding)

### K-4.2 — unpack64 Inline Helper ✅
- **File:** `src/math/ternary_matmul_packed_vnni.c`
- Replaced 4×`unpack16_to_wenc_u8()` calls in inner loop with single `unpack64()`
- Reduces function-call overhead; enables compiler to maintain register state
- Confirmed no regression vs pre-K-4 baseline

### K-4.3 — Adaptive Dispatcher Spin ✅
- **File:** `src/threading/thread_pool.c`
- Workers spin for `SPIN_ITERS=40000` iterations before parking on `pthread_cond_wait`
- Eliminates wake latency for short matmul tasks at T=4 sweet spot
- Tuned for Tiger Lake L1 cache residency window

### K-4.4 — Barrier Fix (Deadlock on Repeated Dispatches) ✅
- **File:** `src/threading/thread_pool.c`
- Epoch counter distinguishes sequential dispatches; workers compare epoch before re-parking
- Fixed: inner `while (task_fn != NULL)` caused workers to stay trapped if new dispatch
  started before all workers exited — deadlock on second call
- Verified with 100+ sequential dispatch stress tests

### K-4.5 — Quantisation Cache Correctness ✅
- **File:** `src/math/simd_dispatch.c`
- Per-row INT8 scale caching (`TnQuantCache`) invalidated when `x` pointer changes
- Fixed stale cache bug: identical pointer with different content (reused buffers)
  produced silently wrong INT8 outputs
- Validated with test vectors; no tok/s regression

---

## PHASE K-5: Caller-Participates + VNNI-256 + T=8 Fix + Calibration ✅

*Implemented: 2026-03-19. Performance references: Addenda Y, Z, AA, AB, AC.*

### K-5.1 — Caller-Participates Thread Model ✅
- **File:** `src/threading/thread_pool.c`, `src/math/parallel_matmul.c`
- Main thread joins workers as thread N+1 instead of blocking on completion barrier
- Effect: reduces idle synchronisation overhead; effectively T=effective+1 free
- Enabled by `threadpool_dispatch_caller_participates()` entry point
- Addendum Y: +15% at T=4 (BF16), peak **24.43 tok/s** on Tiger Lake

### K-5.2 — AVX-VNNI 256-bit Backend ✅
- **File:** `src/math/ternary_matmul_packed_avx_vnni.c`
- `_mm256_dpbusds_epi32` (VEX-encoded VNNI; Alder Lake, AMD Zen 3, no AVX-512 throttle)
- Guard: `#if TN_HAS_AVXVNNI` — compiled with `-mavxvnni` per-file Makefile rule
- Same `w_enc` bias trick as AVX-512 VNNI (`dpbusds(w+1, qx) - sum_qx = true_dot`)
- Auto-selected by `tn_simd_init()` on hardware with AVX-VNNI but not AVX-512

### K-5.3 — T=8 Adaptive Blocking-Wait Fix ✅
- **File:** `src/threading/thread_pool.c`, `include/threading/thread_pool.h`
- `detect_physical_cores()` reads unique `(physical_id, core_id)` pairs from `/proc/cpuinfo`
- `use_blocking_wait = (n_threads >= physical_cores * 2)`: activates when HT siblings are used
- When active, workers skip 40,000-iteration spin and go directly to `pthread_cond_wait`
- Tiger Lake: triggers at T=8 (4 physical × 2 HT = 8 logical)
- Result: T=8 went from **2.6 tok/s → 21.53 tok/s** (+188%) — spinlock contention eliminated

### K-5.4 — Layer-Level Pre-Quantisation ✅
- **File:** `src/math/parallel_matmul.c`, `include/math/parallel_matmul.h`,
           `src/transformer/attention.c`, `src/transformer/ffn.c`
- `TnPreqActivation` struct: caller owns `int8_t buf[16384]` on their stack; struct holds pointer only
- `tn_preq_prepare()`: quantises `s->xb` once per projection group
- Attention: Q/K/V share one quantisation of `s->xb`
- FFN: gate/up share one quantisation of `s->xb`
- INT8 conversion done exactly once per layer instead of once per matmul projection
- Savings ~0.005% on DRAM-bound workload — primarily an architectural correctness improvement

### K-5.5 — INT8/INT4 Classifier + Bandwidth Probe ✅
- **File:** `src/core/hardware_profile.c`, `src/math/simd_dispatch.c`
- `--classifier int8` flag: selects INT8 weight encoding for LM head (64-col per row)
- INT4 VBMI classifier: 3-instruction unpack with `vpermb` — Xeon Ice Lake / Sapphire Rapids
- BW probe fix: `tn_measure_dram_bandwidth()` now uses uncached accesses — real DRAM latency
- LM head (32K × 2560): BF16 = 131 MB/token, INT8 = 65.5 MB/token → +36% avg speedup
- Addendum T: **36.25 tok/s** on Xeon (95% of measured ceiling)
- Addendum U: **1.80×** faster than bitnet.cpp I2_S on same hardware

### K-5.6 — Robust Calibration System ✅
- **File:** `src/core/calibration.c`, `include/core/calibration.h`
- **Previous bug:** `bench_matmul()` created a `ThreadPool` but called `tn_ternary_matmul_packed`
  (single-threaded fn ptr) — T had zero effect on calibration results
- **Fix:** `bench_robust()` calls `parallel_ternary_matmul_packed()` — pool is actually used
- **Thermal warmup:** Time-based (4s cold / 1.5s hot) instead of 3 fixed iterations (~135ms)
  — avoids Tiger Lake 20–28s PL2 turbo window causing false-positive burst readings
- **Per-rep timing:** Each rep measured independently; median of N reported (not total/N)
- **Background load monitor:** Samples `/proc/stat` for 500ms before benchmark; displays `[!]`
  when background CPU > 30% — warns if results may be contaminated
- **Progress output:** Phase 1 (SIMD compare) and Phase 2 (T=1..N thread sweep) print live
  dots + results so users see activity during 10–20s calibration
- **Thread sweep:** T=1..physical_cores×2 measured; best T written to `TnCalibResult.best_threads`
- **`TN_CALIB_VERSION` 1 → 2:** Bumped to invalidate any stale binary cache
- **POSIX.1-2008 compliance:** `usleep` removed (deprecated) → `nanosleep`
- Addendum AC: T=7 identified as optimal (183.7 tok/s); 18/18 regression tests pass

---

## PHASE 16-D: Documentation & Packaging

### 16.1 — Usage Documentation
- File: `USAGE.md`
- Quick start: download model, convert, run.
- CLI reference for all flags.
- REPL commands reference.

### 16.2 — Binary File Format Specification
- File: `docs/FORMAT.md`
- Byte-level specification of the `.bin` weight file format.
- Magic number, version, config layout, scale factor layout, packed weight layout.
- Enables third-party tool authors to produce compatible files.

### 16.3 — Build Instructions
- Extend existing `USAGE.md` or separate `BUILD.md`.
- Per-platform: Linux (gcc/clang), macOS (clang), Windows (MSVC + CMake or WSL).
- How to verify SIMD support: `gcc -march=native -dM -E - < /dev/null | grep AVX`.

---

## Dependency Graph (Critical Path)

```
Phase 0 (Scaffold)
    └─→ Phase 2.6 (Aligned Allocator) ← MUST EXIST BEFORE ANY BUFFER ALLOC
         └─→ Phase 1 (Structs — RunState uses aligned alloc + transposed KV)
              ├─→ Phase 2.1–2.5 (Memory: mmap, RAM probe, budget) ──────┐
              ├─→ Phase 3.1,3.7–3.10 (Scalar Math Primitives) ─────────┤
              │    └─→ Phase 4 (Threading) ────────────────────────────┤
              ├─→ Phase 5 (Tokenizer) ────────────────────────────────┤
              │                                                        ▼
              └──────────────────────────────────→ Phase 6 (Transformer Forward Pass)
                                                  │ (uses transposed KV layout)
                                       ┌──────────┤
                                       ▼          ▼
                                  Phase 7      Phase 8
                                 (Sampling)  (KV Optimize)
                                       │          │
                                       ▼          ▼
                                  Phase 9 (Reasoning / Think Tags)
                                       │
                            ┌──────────┼──────────┐
                            ▼          │          ▼
                      Phase 10.1–10.3  │    Phase 11.1–11.3
                     (Converters)      │   (Vision Encoder)
                            │          │          │
                            ▼          │          ▼
                      Phase 10.4–10.5  │    Phase 11.4 ← MLP PROJECTOR (dim mismatch fix)
                     (LUT Unpackers +  │          │
                      Fused MatMul)    │          ▼
                            │          │    Phase 11.5
                            │          │   (Vision Bridge)
                            ▼          ▼          │
                      Phase 3.2–3.6 ──────────────┤
                     (SIMD Variants +             │
                      Dispatch)                   │
                            │                     │
                            └──────────┬──────────┘
                                       ▼
                                 Phase 12 (CLI + Main)
                                       │
                    ┌──────────────────┤
                    ▼                  ▼
              Phase 14           Phase 15
            (Agentic Tools)    (RAG / Vector DB)
              14.1 Tag ───────→ 15.1 Embedder
              Interceptor       15.2 Cosine Sim
              14.2 Cmd ────┐    15.3 Vector DB
              Executor     │    15.4 Search
              14.3 Output  │    15.5 Auto-Retrieve
              Inject       │    15.6 Auto-Save
              14.4 Agent ──┴──→ (consumes 15.3–15.5)
              Loop
              14.5 User
              Approval
                    │                  │
                    └──────────┬───────┘
                               ▼
                         Phase 13 (Tests — 13.1–13.24)
                               │
                               ▼
                         Phase 16-D (Docs)
```

### Key Dependencies Introduced by the Four Fixes

| Fix | Source | Depends On | Consumed By |
|-----|--------|-----------|-------------|
| **#1 Aligned Alloc** (2.6) | `src/memory/aligned_alloc.c` | Phase 0 (platform.h) | Phase 1.6 (RunState), Phase 8 (KV buffers), all SIMD kernels |
| **#2 Transposed KV** (1.5) | `include/core/run_state.h` | Phase 2.6 (aligned alloc) | Phase 6.2 (attention), Phase 8 (KV compress), Phase 11.5 (vision inject) |
| **#3 LUT Unpack** (10.4–10.5) | `src/core/unpack_avx2.c`, `src/math/ternary_matmul_packed.c` | Phase 3.1 (scalar ref), Phase 10.1 (pack format) | Phase 3.3–3.5 (SIMD matmul variants), Phase 4.4 (parallel matmul) |
| **#4 Vision Projector** (11.4) | `src/multimodal/vision_projector.c` | Phase 11.3 (encoder output), Phase 3 (matmul) | Phase 11.5 (bridge injection) |

---

## PHASE 17: Mixture of Experts (MoE) Routing

We built our engine assuming a Dense model (like Llama 3 or BitNet), where every word passes through every weight matrix. To run DeepSeek-R1 or Mixtral, we need an MoE Router that activates only a subset of Experts per token.

### 17.1 — MoE Config Extension
- File: `include/core/moe_config.h`
- Extend `Config` with MoE fields:
  ```c
  typedef struct {
      int num_experts;          // e.g., 8
      int num_experts_per_tok;  // e.g., 2 (top-k routing)
      int expert_hidden_dim;    // each expert's FFN hidden dim
      bool is_moe;              // flag read from model header
  } MoEConfig;
  ```
- `TernaryError moe_config_read(MoEConfig *mc, const void *mapped_ptr, size_t *offset);`

### 17.2 — Router Gate
- File: `include/transformer/moe_router.h` + `src/transformer/moe_router.c`
- `void moe_router_forward(float *expert_scores, int *selected_experts, const float *x, const float *gate_weights, int dim, int num_experts, int top_k);`
  - Computes `gate_logits = x @ gate_weights` (dim → num_experts).
  - Softmax over gate_logits.
  - Selects top-k experts by score.
  - Returns selected expert indices + their normalized scores.

### 17.3 — MoE FFN Forward
- File: `include/transformer/moe_ffn.h` + `src/transformer/moe_ffn.c`
- `void moe_ffn_forward(RunState *s, const TransformerWeights *w, const Config *p, const MoEConfig *mc, int layer, ThreadPool *tp);`
  - Calls router gate to select top-k experts.
  - For each selected expert: runs SwiGLU FFN (w1/w2/w3 for that expert).
  - Weighted sum of expert outputs using gate scores.
  - Residual add to `s->x`.
- **Key optimization:** Only 2 of 8 experts run per token — 4x less compute than dense FFN.

### 17.4 — MoE Weight Mapping
- File: `src/core/moe_weights.c`
- `void moe_weights_map(TransformerWeights *w, const MoEConfig *mc, int8_t *ptr, size_t *offset);`
  - Maps `num_experts` sets of (w1, w2, w3) per layer.
  - Per-expert scale factors.
  - Gate weight matrix per layer.

---

### 17.5 — MoE Config MLA Fields *(Phase 17 sub-item — Item B.1 of multi-feature expansion)*
- **Status:** PLANNED
- **File:** `include/core/moe_config.h` (extend), `src/core/moe_config.c` (read)
- **Problem solved:** DeepSeek-V2-Lite uses Multi-head Latent Attention (MLA). The raw projection
  matrices (q_proj, kv_a, kv_b) must be stored in the model file and header. Without them,
  the engine cannot reconstruct correct Q/K/V — producing repetitive garbage output.
- Add 5 MLA fields to `MoEConfig`:
  ```c
  int has_mla;            /* 1 for DeepSeek-style MLA, 0 otherwise */
  int kv_lora_rank;       /* latent KV dimension (e.g. 512) */
  int qk_nope_head_dim;   /* per-head no-positional-encoding dim (e.g. 128) */
  int qk_rope_head_dim;   /* per-head RoPE dim (e.g. 64) */
  int v_head_dim;         /* per-head value dim (e.g. 128) */
  ```
- Extend `moe_config_read()` to read bytes 28–47 of MoE header (was padding).
- `moe_config_init_dense()` sets all new fields to 0.
- **Hot path cost:** Zero — `has_mla == 0` for all non-MLA models.

### 17.6 — MLA Weight Pointers in TransformerWeights *(Item B.2)*
- **Status:** PLANNED
- **Files:** `include/core/weights.h` (+6 fields), `src/core/moe_weights.c` (alloc/free/map),
  `src/core/weights.c` (map routing)
- Per-layer MLA projection arrays (NULL for non-MLA models, zero overhead):
  ```c
  tn_i8 **mla_wq;     /* [n_layers] full Q proj: n_heads*(nope+rope) × dim */
  tn_i8 **mla_wkv_a;  /* [n_layers] KV compress: (kv_lora+rope) × dim      */
  tn_i8 **mla_wkv_b;  /* [n_layers] KV expand:   n_kv_heads*(nope+v) × kv_lora */
  float  *mla_sq, *mla_skv_a, *mla_skv_b;   /* per-layer scales */
  ```
- `weights_map()` when `has_mla=1`: use MLA sizes for wq; map wkv_a/wkv_b; leave wk/wv NULL.

### 17.7 — k_rope_cache in RunState *(Item B.4)*
- **Status:** PLANNED
- **Files:** `include/core/run_state.h` (+1 field), `src/core/run_state.c` (free),
  `src/core/mla_run_state.c` (NEW — alloc)
- Add `float **k_rope_cache` to `RunState`; NULL for non-MLA models.
- `mla_run_state_alloc()` allocates `[n_layers][seq_len × qk_rope_head_dim]` per layer.
- Frees in `run_state_free()` if non-NULL.

### 17.8 — MLA Attention Forward Pass *(Item B.3)*
- **Status:** PLANNED
- **Files:** NEW `include/transformer/mla_attention.h`, NEW `src/transformer/mla_attention.c`
- Algorithm per forward call (6 steps):
  1. `q = mla_wq @ x` → full Q (n_heads × (nope+rope)); split per-head into q_nope + q_rope
  2. `kvc = mla_wkv_a @ x` → split into kv_latent (kv_lora) + k_rope (64-dim shared RoPE)
  3. `kv = mla_wkv_b @ kv_latent` → split into k_nope (n_kv_heads × nope) + v (n_kv_heads × v_dim)
  4. Apply RoPE to q_rope[each head] and k_rope[shared]; store k_nope in `key_cache`, k_rope in
     `k_rope_cache`, v in `value_cache`
  5. Attention scores = `(q_nope·k_nope + q_rope·k_rope) / sqrt(qk_nope + qk_rope)` per head
  6. `out = attn_weights @ v`; project with wo; residual add to `s->x`
- Scratch buffers: reuse `s->hb` (10944 >> 3072 for Q) and `s->hb2` (10944 >> 4096 for KV) —
  **zero new allocations per forward call**
- **Modular guard in `attention.c`:**
  ```c
  if (mc && mc->has_mla) { mla_attention_forward(s, w, cfg, mc, layer, tp); return; }
  ```
  One `#include` + one `if` — nothing else in attention.c changes.

### 17.9 — Converter: Write Raw MLA Weights *(Item B.6)*
- **Status:** PLANNED
- **File:** `tools/convert_hf.py`
- Replace `expand_mla_attention()` (which pre-multiplied kv_b × kv_a, losing positional info)
  with `write_mla_attention()` that writes raw tensors per layer:
  - Full `q_proj` (3072×2048) ternary + scale
  - `kv_a_proj_with_mqa` (576×2048) ternary + scale
  - `kv_b_proj` (4096×512) ternary + scale
  - `o_proj` unchanged
- Extend MoE header bytes 28–47 with `has_mla=1, kv_lora_rank=512, qk_nope_head_dim=128,
  qk_rope_head_dim=64, v_head_dim=128`.
- Weight data offset unchanged (starts at byte 128) — **no format break for existing models**.
- **Root cause of garbage output:** The old approximation truncated q_rope heads and lost the
  shared k_rope (64-dim positional encoding). Both queries and keys had incorrect positional
  context → model couldn't track token positions → repetitive degenerate output.

### 17.10 — Re-convert DeepSeek-V2-Lite *(Item B.7)*
- **Status:** PLANNED
- Run `python3 tools/convert_hf.py --model models/deepseek-ai/DeepSeek-V2-Lite --output models/deepseek-v2-lite.bin --moe`
- Smoke test: "The capital of France is" → expect "Paris" (not "blueprint blueprint...")
- Re-conversion takes ~15–20 minutes (same as original)

### 17.11 — MLA Unit Test *(Item B.8)*
- **Status:** PLANNED
- **File:** `tests/test_moe.c` (extend)
- Add `test_mla_forward_coherent`: 4-expert 1-layer synthetic model with `has_mla=1`;
  verify output is finite and that setting `has_mla=0` on the same weights produces different
  (non-identical) output — confirming routing is actually doing something.

### 17.12 — Full DeepSeek Re-Sweep with Correct Output *(Item H)*
- **Status:** PLANNED (depends on 17.9–17.10)
- Run full T=1..8 × 5 SIMD × 3 classifiers sweep after MLA fix.
- Validate "The capital of France is" → "Paris" **before** benchmarking.
- Document as **Addendum AK** in `docs/PERFORMANCE_CEILING_REPORT.md`:
  - Per-classifier peaks (BF16/INT8/INT4 separately)
  - Thread-wise regression tables for all three classifiers
  - Comparison: pre-MLA-fix (Addendum AJ) vs post-MLA-fix performance
- **MLA is always applied universally** — any model with `has_mla=1` in its header benefits.

---

## PHASE 34.2b: Quantized GGUF Support (Q8_0, Q4_K, I2_S) *(Items C, D)*

**Background:** Currently `tensor_to_f32()` only handles F32/F16/BF16 GGUF tensors. Q8_0 and
Q4_K represent 90%+ of community quantized models in the wild. I2_S is the BitNet-native GGUF
format used by llama.cpp and bitnet.cpp.

### 34.2b.1 — Dequantization Library *(Item C)*
- **Status:** PLANNED
- **Files:** NEW `include/core/gguf_quant.h`, NEW `src/core/gguf_quant.c`
- **Q8_0:** 32-weight blocks, 1×FP16 scale/block. Trivial int8 × f16_scale multiply.
- **Q4_K:** 256-weight super-blocks, 8 sub-blocks of 32. Scale+min per sub-block encoded in
  12-byte header. Most complex k-quant; Q5_K and Q6_K follow same pattern (+~30 lines each).
- Modify `src/core/gguf_loader.c`: add `case GGUF_TYPE_Q8_0` and `case GGUF_TYPE_Q4_K`
  in `tensor_to_f32()` switch. Callers are fully unaware of quantization type.

### 34.2b.2 — I2_S BitNet Native Format *(Item D)*
- **Status:** PLANNED
- 2 bits/weight, groups of 32, FP16 scale per group.
- Decode: `val = (packed >> 2*i) & 0x3` → sign-extend to {-1, 0, +1} → multiply by scale.
- Extends `gguf_quant.c`; adds one case to `gguf_loader.c`.

---

## PHASE 34.5: GGUF Tokenizer Extraction *(Items A, F)*

**Background:** GGUF files embed the full vocabulary in metadata KV pairs under
`tokenizer.ggml.*` keys. Extracting directly from GGUF eliminates the need for external
`.bin` tokenizer files — critical for SmolLM2 and other GGUF-native models.

### 34.5.1 — GGUF Reader ARRAY Metadata Support *(Item F, first step)*
- **Status:** PLANNED
- **Files:** `include/core/gguf_reader.h` (extend GGUFMeta with array union member),
  `src/core/gguf_reader.c` (parse ARRAY-type values in `gguf_read_header()`)
- GGUF ARRAY metadata format: `[elem_type:u32][count:u64][elem_0..elem_n]`
- GGUFMeta union extension: `struct { GGUFValType elem_type; uint64_t count; void *data; } array;`
- Existing callers using scalar KV pairs are unaffected (union extension).

### 34.5.2 — GGUF Tokenizer Loader *(Items A + F combined)*
- **Status:** PLANNED
- **Files:** NEW `include/tokenizer/tokenizer_gguf.h`, NEW `src/tokenizer/tokenizer_gguf.c`
- Reads vocabulary from parsed GGUF metadata:
  - `tokenizer.ggml.tokens` → vocab strings (ARRAY of strings)
  - `tokenizer.ggml.scores` → merge scores (ARRAY of f32)
  - `tokenizer.ggml.token_type` → byte/control/normal/unknown flags (ARRAY of u32)
  - `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id` → special tokens (scalars)
- Populates existing `Tokenizer` struct — no downstream inference changes needed.
- Auto-detection in `src/cli/main.c`: if GGUF model + no `--tokenizer` path given →
  call `tokenizer_load_from_gguf()` (3-line change).

---

## PHASE 16-E: GGUF-Aware Bandwidth Ceiling *(Item E)*

**Background:** `hardware_profile.c` has hardcoded BitNet-specific constants that are wrong
for every other model (wrong vocab, wrong dim, wrong byte count). Ceiling numbers for
non-BitNet models are misleading.

### 16-E.1 — Remove Hardcoded Constants, Add Model-Aware API
- **Status:** PLANNED
- **Files:** `include/core/hardware_profile.h` (+new struct + API), `src/core/hardware_profile.c`
  (remove `MODEL_*` defines, add `tn_hardware_profile_update_model()`), `src/cli/main.c`,
  `src/core/gguf_loader.c`
- New `TnModelParams` struct captures model bytes, vocab, dim, quant_bits, MoE sparsity.
- Five-strategy approach:
  - **S5 (always):** At init: ceiling = 0.0, display "N/A — load a model to calculate"
  - **S3:** After `.bin` load: use Config struct (exact: dim/n_layers/vocab known)
  - **S2:** After GGUF header parse: extract from `llama.embedding_length`, `llama.block_count`
  - **S4 extension:** If `is_moe=1`: `effective_bytes = model_bytes × (top_k/n_experts × expert_fraction + dense_fraction)`
  - **S1 fallback:** Unknown architecture: file-size proxy (conservative, never wrong)
- Users who switch/add/delete models: `tn_hardware_profile_update_model()` called again
  on each model load — always accurate for the currently loaded model.
- Fresh install (no model): S5 applies — hardware BW shown correctly, ceiling shows "N/A".
- User-trained models: Config struct is authoritative (S3).

---

## PHASE 34.3: SmolLM2-1.7B Benchmark *(Item G)*

### 34.3.1 — Download SmolLM2-1.7B-Instruct-F16 GGUF
- **Status:** IN PROGRESS (downloading in background)
- Target: `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` → `smollm2-1.7b-instruct-fp16.gguf` (~3.4 GB)
- Alternative if Q4_K ready: `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (~1.0 GB)
- Tokenizer: extracted directly from GGUF file (Item F) — no external file needed.

### 34.3.2 — SmolLM2-1.7B Benchmark Sweep
- **Status:** PLANNED (depends on 34.5.2 + model download)
- Validate output quality on known prompts before sweep.
- Run full T=1..8 × 5 SIMD × 3 classifiers sweep.
- Document as **Addendum AL** in `docs/PERFORMANCE_CEILING_REPORT.md`.

---

## PHASE 18: Speculative Decoding

The ultimate CPU speed hack. Load two models: a massive "Brain" (verifier) and a tiny "Drafter". The Drafter guesses N tokens rapidly, the Brain verifies them all in one batched forward pass.

### 18.1 — Draft Model Loader
- File: `include/speculative/draft_model.h` + `src/speculative/draft_model.c`
- `typedef struct { Config config; TransformerWeights weights; RunState state; MappedFile mf; } DraftModel;`
- `TernaryError draft_model_load(DraftModel *dm, const char *path);`
- Loads a small model (e.g., 125M params) alongside the main model.
- Uses separate mmap'd file and separate RunState.

### 18.2 — Speculative Generation Loop
- File: `include/speculative/spec_decode.h` + `src/speculative/spec_decode.c`
- `void speculative_generate(Config *p, TransformerWeights *w, RunState *s, DraftModel *draft, Tokenizer *t, ThreadPool *tp, const char *prompt, int max_tokens, float temperature, float top_p, int spec_length);`
  - `spec_length`: number of tokens to draft (default: 5).
  - **Algorithm:**
    1. Draft model generates `spec_length` candidate tokens greedily.
    2. Main model runs a single batched forward pass over all candidates.
    3. Compare draft logits vs. main logits using rejection sampling.
    4. Accept matching tokens (often 3-4 out of 5).
    5. On first mismatch: resample from main model's distribution, discard remaining drafts.
    6. Repeat.
  - **Speedup:** 2-3x on CPU when draft model is 10x smaller than main model.

### 18.3 — KV Cache Rollback
- File: `src/speculative/kv_rollback.c`
- `void kv_cache_rollback(RunState *s, const Config *p, int rollback_pos);`
  - When speculative tokens are rejected, the KV cache must be rewound.
  - Sets `s->current_pos = rollback_pos` — the rejected K/V entries are simply overwritten on next forward pass.

---

## PHASE 19: LoRA Adapters (Hot-Swappable Brains)

Low-Rank Adaptation allows loading tiny 50MB "patch" files that alter the model's behavior without modifying base weights.

### 19.1 — LoRA Weight Structures
- File: `include/core/lora.h`
- ```c
  typedef struct {
      int rank;              // LoRA rank (e.g., 16, 32, 64)
      float alpha;           // Scaling factor
      float **lora_A;        // Per-layer A matrices: [dim × rank]
      float **lora_B;        // Per-layer B matrices: [rank × dim]
      char *target_modules;  // Comma-separated: "q_proj,v_proj,..."
  } LoRAWeights;
  ```
- `TernaryError lora_load(LoRAWeights *lora, const char *path);`
- `void lora_free(LoRAWeights *lora);`

### 19.2 — LoRA File Loader
- File: `src/core/lora_load.c`
- Reads a `.lora.bin` file with header: `[magic "LORA"] [rank] [alpha] [target_mask] [A_matrices] [B_matrices]`.
- A and B matrices stored as float16, dequantized to float32 on load.
- Memory footprint: ~50MB for rank-32 on a 2B model.

### 19.3 — LoRA-Fused MatMul
- File: `include/math/lora_matmul.h` + `src/math/lora_matmul.c`
- `void ternary_matmul_with_lora(float *out, const float *x, const int8_t *w, int n, int d, float scale, const LoRAWeights *lora, int layer, const char *module_name);`
  - Computes: `out = (x @ W) * scale + (x @ A @ B) * (alpha / rank)`.
  - The LoRA term `x @ A @ B` is computed separately: first `x @ A` (dim → rank, very small), then `result @ B` (rank → dim).
  - Added to the base matmul output.
- **Hot-swap:** Calling `lora_free()` + `lora_load()` with a different file switches personality in milliseconds.

### 19.4 — LoRA Converter Tool
- File: `tools/convert_lora.py`
- Reads HuggingFace LoRA adapter (typically `adapter_model.safetensors` + `adapter_config.json`).
- Extracts A/B matrices, writes to `.lora.bin` format.
- CLI: `python convert_lora.py --adapter path/to/lora --output adapter.lora.bin`

---

## PHASE 20: Grammar-Constrained Decoding (JSON Mode)

Physically blocks the AI from outputting invalid syntax by forcing illegal token probabilities to negative infinity.

### 20.1 — Grammar Definition Parser
- File: `include/sampling/grammar.h` + `src/sampling/grammar_parse.c`
- `typedef struct { int num_states; int num_transitions; GrammarTransition *transitions; int start_state; int *accept_states; int num_accept; } Grammar;`
- `TernaryError grammar_load_json_schema(Grammar *g);` — Built-in JSON grammar.
- `TernaryError grammar_load_from_bnf(Grammar *g, const char *bnf_string);` — Parse custom BNF/EBNF grammar string.

### 20.2 — Finite State Machine Engine
- File: `include/sampling/fsm.h` + `src/sampling/fsm.c`
- `typedef struct { int current_state; const Grammar *grammar; } FSMState;`
- `void fsm_init(FSMState *fsm, const Grammar *g);`
- `void fsm_compute_token_mask(const FSMState *fsm, const Tokenizer *t, float *logits, int vocab_size);`
  - For each token in vocabulary: simulates feeding that token's characters into the FSM.
  - If any character causes an invalid transition → set `logits[token_id] = -INFINITY`.
  - Remaining valid tokens retain their original logits.
- `void fsm_advance(FSMState *fsm, int token_id, const Tokenizer *t);`
  - Updates FSM state after a token is accepted.

### 20.3 — Constrained Sampling Integration
- File: `src/sampling/constrained_sample.c`
- `int sample_constrained(float *logits, int vocab_size, FSMState *fsm, const Tokenizer *t, float temperature, float top_p, unsigned long long *rng_state);`
  - Applies FSM mask → temperature → top-p → sample.
  - Advances FSM state with chosen token.
  - Guarantees 100% syntactically valid output.

---

## PHASE 21: OpenAI-Compatible API Layer ✅

> **Implemented 2026-03-27.** See `WALKTHROUGH_PHASE21.md` for full architecture docs.
>
> Implementation notes vs. original plan:
> - Used pure POSIX sockets instead of `mongoose.c` — zero external dependencies maintained.
> - Used a purpose-built JSON parser instead of `cJSON` — consistent with no-external-deps policy.
> - Chat template compiler handles ChatML, Llama-3, DeepSeek, Mistral, and Raw formats.
> - `generate()` refactored: `generate_with_callback()` is the core loop; `generate()` is a
>   thin stdout wrapper. This enables streaming via the callback without any extra buffering.

Turns the C engine into a headless daemon that any frontend (Cline, OpenHands, SillyTavern) can connect to via standard OpenAI API calls.

### 21.1 — Embedded HTTP Server
- File: `src/api/http_server.c`
- Bundles `mongoose.c` (single-file, MIT-licensed embedded web server).
- `TernaryError api_server_start(int port, ApiContext *ctx);`
- Routes:
  - `GET /v1/models` → returns `{"data": [{"id": "local-adaptive-engine"}]}`.
  - `POST /v1/chat/completions` → main inference endpoint.
- Runs on a background thread. Non-blocking.

### 21.2 — JSON Request Parser (cJSON)
- File: `src/api/json_parse.c`
- Bundles `cJSON.h` / `cJSON.c` (single-file, MIT-licensed JSON parser).
- `TernaryError parse_chat_request(const char *json_body, ChatRequest *req);`
  - Extracts: `messages[]`, `temperature`, `top_p`, `max_tokens`, `stream` (bool).
  - `typedef struct { ChatMessage *messages; int num_messages; float temperature; float top_p; int max_tokens; bool stream; } ChatRequest;`

### 21.3 — Chat Template Compiler
- File: `include/api/chat_template.h` + `src/api/chat_template.c`
- `typedef enum { TEMPLATE_LLAMA3, TEMPLATE_CHATML, TEMPLATE_DEEPSEEK, TEMPLATE_MISTRAL } ChatTemplateType;`
- `void compile_chat_template(char *output, int max_len, const ChatMessage *messages, int num_messages, ChatTemplateType template_type);`
  - Llama 3: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
  - ChatML: `<|im_start|>role\ncontent<|im_end|>`
  - Auto-detects from tokenizer vocab if not specified.

### 21.4 — Server-Sent Events (SSE) Streaming
- File: `src/api/sse_stream.c`
- `void sse_stream_token(struct mg_connection *conn, const char *token_text, bool is_done);`
  - Formats: `data: {"choices": [{"delta": {"content": "token"}}]}\n\n`
  - On completion: `data: [DONE]\n\n`
- Hooks directly into the `generate()` loop — each decoded token is immediately flushed over the TCP socket.

### 21.5 — KV Cache Prefix Matching (Context Caching)
- File: `include/api/prefix_cache.h` + `src/api/prefix_cache.c`
- `int find_prefix_match(const int *new_tokens, int new_len, const int *cached_tokens, int cached_len);`
  - Returns the length of the longest common prefix.
  - If prefix match is found, skip re-computing those tokens — reuse existing KV cache entries.
  - Drops Time-To-First-Token from seconds to milliseconds for repeated API calls.

---

## PHASE 22: State Space Models (Mamba/RWKV) — Universal Architecture Router

Abstracts the forward pass so the engine can run both Transformer (Attention) and SSM (Mamba) architectures.

### 22.1 — Architecture Type Detection
- File: `include/core/arch_detect.h` + `src/core/arch_detect.c`
- `typedef enum { ARCH_TRANSFORMER, ARCH_MAMBA, ARCH_RWKV, ARCH_JAMBA_HYBRID } ArchType;`
- `ArchType detect_architecture(const void *mapped_header);`
  - Reads a flag byte from the model header.
  - Determines which forward-pass pathway to use.

### 22.2 — SSM State Vector
- File: `include/core/ssm_state.h` + `src/core/ssm_state.c`
- `typedef struct { float *hidden_state; int state_dim; int num_layers; } SSMState;`
- Fixed-size state — does NOT grow with sequence length (unlike KV cache).
- `TernaryError ssm_state_alloc(SSMState *ss, int state_dim, int num_layers);`
- `void ssm_state_free(SSMState *ss);`

### 22.3 — Mamba Selective Scan
- File: `include/transformer/mamba_scan.h` + `src/transformer/mamba_scan.c`
- `void mamba_selective_scan(float *out, const float *x, SSMState *ss, const MambaWeights *mw, int dim, int state_dim, int layer);`
  - Implements the S6 (Structured State Space for Sequences) selective scan.
  - Computes: discretize A, B → scan → multiply by C.
  - Updates `hidden_state` in-place.
  - Constant memory and time per token regardless of sequence length.

### 22.4 — Universal Forward Pass Router
- File: `src/transformer/forward_router.c`
- `float *universal_forward(int token, int pos, Config *p, TransformerWeights *w, RunState *s, SSMState *ss, ArchType arch, ThreadPool *tp);`
  - If `ARCH_TRANSFORMER`: calls existing `transformer_forward()`.
  - If `ARCH_MAMBA`: calls `mamba_forward()` (no KV cache, no attention).
  - If `ARCH_JAMBA_HYBRID`: alternates between attention and Mamba layers.

---

## PHASE 23: PagedAttention & Continuous Batching

Enables multi-user API serving by breaking KV cache into non-contiguous memory pages.

### 23.1 — Page Table Manager
- File: `include/kv_cache/page_table.h` + `src/kv_cache/page_table.c`
- `#define PAGE_SIZE 16  // tokens per page`
- ```c
  typedef struct {
      float *page_pool;       // Pre-allocated pool of pages
      int num_pages;          // Total pages in pool
      int *free_list;         // Stack of free page indices
      int free_count;
      int **request_tables;   // Per-request page tables (request_id → page indices)
      int max_requests;
  } PageTableManager;
  ```
- `TernaryError page_table_init(PageTableManager *pt, int total_pages, int head_dim, int max_requests);`
- `int page_table_alloc_page(PageTableManager *pt, int request_id);` — Assigns a free page to a request.
- `void page_table_free_request(PageTableManager *pt, int request_id);` — Returns all pages to free list.

### 23.2 — Paged KV Cache Read/Write
- File: `include/kv_cache/paged_kv.h` + `src/kv_cache/paged_kv.c`
- `void paged_kv_write(PageTableManager *pt, int request_id, int layer, int head, int pos, const float *kv_data, int head_dim);`
  - Computes page index: `pos / PAGE_SIZE`, slot within page: `pos % PAGE_SIZE`.
  - Writes to the correct non-contiguous page.
- `void paged_kv_read(const PageTableManager *pt, int request_id, int layer, int head, int pos, float *kv_data, int head_dim);`
- `void paged_attention_forward(...)` — Modified attention that reads K/V from paged memory.

### 23.3 — Continuous Batching Scheduler
- File: `include/api/batch_scheduler.h` + `src/api/batch_scheduler.c`
- `typedef struct { int request_id; int *tokens; int num_tokens; int current_pos; bool is_prefill; bool is_done; } BatchSlot;`
- `void scheduler_add_request(BatchScheduler *bs, const ChatRequest *req);`
- `void scheduler_step(BatchScheduler *bs, ...);`
  - Groups multiple requests into a single batched forward pass.
  - Prefill requests (first pass) and decode requests (subsequent tokens) can run simultaneously.
  - When a request completes, its pages are freed and the slot is reassigned.

### 23.4 — Shared Prefix Deduplication
- File: `src/kv_cache/shared_prefix.c`
- `int find_shared_prefix(PageTableManager *pt, int req_a, int req_b);`
- If two requests share the same system prompt, the engine points both page tables at the same physical pages for the shared prefix — saving up to 50% memory.

---

## PHASE 24: Dynamic Context Scaling (YaRN / NTK)

Stretches a model's context window beyond its training limit by dynamically rescaling RoPE frequencies.

### 24.1 — RoPE Scaling Configuration
- File: `include/math/rope_scaling.h`
- ```c
  typedef enum { ROPE_NONE, ROPE_LINEAR, ROPE_NTK, ROPE_YARN } RoPEScalingType;
  typedef struct {
      RoPEScalingType type;
      float scale_factor;       // e.g., 4.0 for 4x context extension
      int original_max_pos;     // Model's native training limit (e.g., 8192)
      float beta_fast;          // YaRN parameter
      float beta_slow;          // YaRN parameter
  } RoPEScaling;
  ```
- `void rope_scaling_auto_detect(RoPEScaling *rs, int requested_seq_len, int model_max_seq_len);`
  - If `requested_seq_len > model_max_seq_len`: enables YaRN with `scale_factor = requested / model_max`.

### 24.2 — YaRN-Scaled RoPE Implementation
- File: `src/math/rope_yarn.c`
- `void apply_rope_yarn(float *q, float *k, int dim, int head_dim, int pos, int n_heads, int n_kv_heads, const RoPEScaling *rs);`
  - For each frequency dimension `i`:
    - Low frequencies (< beta_slow): scale linearly by `1/scale_factor`.
    - High frequencies (> beta_fast): leave unmodified.
    - Mid frequencies: smooth interpolation (YaRN ramp function).
  - **Result:** An 8K model seamlessly reads 32K, 64K, or 128K context without retraining.

---

## PHASE 25: Native Audio Streaming

Enables real-time voice conversations by streaming audio tokens directly into the LLM.

### 25.1 — Audio Codec Integration (EnCodec/Mimi)
- File: `include/multimodal/audio_codec.h` + `src/multimodal/audio_codec.c`
- `typedef struct { float *codebook_embeddings; int num_codebooks; int codebook_size; int frame_rate; } AudioCodec;`
- `TernaryError audio_codec_load(AudioCodec *ac, const char *codec_path);`
- `void audio_encode(const AudioCodec *ac, const float *pcm_samples, int num_samples, int *audio_tokens, int *num_tokens);`
  - Converts raw PCM audio → discrete audio token IDs at ~50 frames/second.
- `void audio_decode(const AudioCodec *ac, const int *audio_tokens, int num_tokens, float *pcm_samples, int *num_samples);`
  - Converts audio tokens back → PCM waveform for speaker output.

### 25.2 — WebSocket Audio Endpoint
- File: `src/api/ws_audio.c`
- Adds `ws://localhost:8080/audio` WebSocket route to the mongoose server.
- **Streaming protocol:**
  - Client sends raw PCM frames (16kHz, mono, 16-bit).
  - Server encodes to audio tokens, runs through LLM, decodes output tokens to PCM.
  - Server streams PCM response frames back in real-time.
- Target latency: < 300ms end-to-end.

### 25.3 — Audio Token Injection
- File: `src/multimodal/audio_inject.c`
- `void inject_audio_tokens(RunState *s, const Config *p, const TransformerWeights *w, const int *audio_tokens, int num_tokens, ThreadPool *tp);`
  - Embeds audio tokens using a dedicated audio embedding table.
  - Runs through transformer forward pass to populate KV cache.
  - Interleaved with text tokens in the same sequence.

---

## PHASE 26: Prompt Caching (Radix Trees) — Zero-Compute Recall

Eliminates redundant computation when the same prompt prefix appears across multiple API requests.

### 26.1 — Radix Tree Data Structure
- File: `include/kv_cache/radix_tree.h` + `src/kv_cache/radix_tree.c`
- ```c
  typedef struct RadixNode {
      int *token_sequence;       // Tokens at this node
      int seq_len;
      int kv_cache_start_pos;    // Where in KV cache this prefix starts
      int kv_cache_end_pos;      // Where it ends
      struct RadixNode **children;
      int num_children;
      int64_t last_access_time;  // For LRU eviction
  } RadixNode;
  ```
- `void radix_tree_init(RadixTree *rt);`
- `RadixNode *radix_tree_search(RadixTree *rt, const int *tokens, int len, int *match_len);`
  - Returns longest matching prefix node + how many tokens matched.
- `void radix_tree_insert(RadixTree *rt, const int *tokens, int len, int kv_start, int kv_end);`

### 26.2 — Prefix-Aware Inference
- File: `src/kv_cache/prefix_inference.c`
- `int skip_cached_prefix(RadixTree *rt, RunState *s, const int *prompt_tokens, int prompt_len);`
  - Searches radix tree for prompt prefix.
  - If found: copies cached KV entries, sets `s->current_pos` to end of prefix.
  - Returns number of tokens skipped (no forward pass needed for these).
  - **Result:** Time-To-First-Token drops from 15s to 0.1s for repeated system prompts.

### 26.3 — LRU Cache Eviction
- File: `src/kv_cache/radix_evict.c`
- `void radix_tree_evict_lru(RadixTree *rt, int target_free_pages);`
  - When memory pressure is high, evicts least-recently-used prefix entries.
  - Frees corresponding KV cache pages.

---

## PHASE 27: CPU Cache Tiling (FlashAttention for CPU)

Maximizes L1/L2 cache utilization by tiling matrix operations into cache-sized blocks.

### 27.1 — L1 Cache Size Detection
- File: `include/threading/cache_probe.h` + `src/threading/cache_probe.c`
- `int get_l1_cache_size_bytes();`
  - Linux: reads `/sys/devices/system/cpu/cpu0/cache/index0/size`.
  - x86 CPUID: `cpuid(4, 0)` for L1d parameters.
  - Fallback: assume 64KB.
- `int compute_optimal_tile_size(int l1_bytes, int elem_size);`
  - Returns tile dimension (e.g., 64) that ensures a tile × tile block fits in L1.

### 27.2 — Tiled Matrix Multiply
- File: `include/math/tiled_matmul.h` + `src/math/tiled_matmul.c`
- `void ternary_matmul_tiled(float *out, const float *x, const int8_t *w, int n, int d, float scale, int tile_size);`
  - Splits the matmul into `tile_size × tile_size` blocks.
  - Each block fits entirely in L1 cache.
  - Processes: for each output tile → for each input tile → accumulate.
  - **Performance:** Up to 99% L1 cache hit rate vs. ~30% for naive row-major iteration.

### 27.3 — Tiled Attention (FlashAttention-CPU)
- File: `src/transformer/flash_attention_cpu.c`
- `void flash_attention_forward_cpu(RunState *s, const TransformerWeights *w, const Config *p, int layer, int pos, int tile_size, ThreadPool *tp);`
  - Tiles the attention score computation: processes Q×K^T in blocks.
  - Online softmax: computes partial softmax per tile, merges with running statistics.
  - Never materializes the full `seq_len × seq_len` attention matrix.
  - Memory: O(tile_size²) instead of O(seq_len²).

---

## PHASE 28: Distributed Inference (RPC/MPI) — Network Tensor Router

Splits a model across multiple machines on the same network for models too large for a single device.

### 28.1 — Layer Assignment Planner
- File: `include/distributed/layer_plan.h` + `src/distributed/layer_plan.c`
- `typedef struct { int start_layer; int end_layer; char *host; int port; } NodeAssignment;`
- `void plan_layer_distribution(NodeAssignment *assignments, int *num_nodes, const Config *p, int64_t *node_ram_bytes, int num_available_nodes);`
  - Distributes layers proportionally to available RAM on each node.
  - Node with most RAM gets more layers.

### 28.2 — TCP Tensor Transport
- File: `include/distributed/tensor_rpc.h` + `src/distributed/tensor_rpc.c`
- `TernaryError rpc_send_tensor(int sockfd, const float *data, int size);`
- `TernaryError rpc_recv_tensor(int sockfd, float *data, int size);`
  - Simple binary protocol: `[size: 4 bytes][data: size * sizeof(float) bytes]`.
  - Uses TCP sockets for reliable delivery.
- `TernaryError rpc_connect(const char *host, int port, int *sockfd);`
- `TernaryError rpc_listen(int port, int *server_fd);`

### 28.3 — Distributed Forward Pass
- File: `src/distributed/dist_forward.c`
- `float *distributed_forward(int token, int pos, Config *p, TransformerWeights *w, RunState *s, NodeAssignment *assignments, int num_nodes, ThreadPool *tp);`
  - Each node computes its assigned layers locally.
  - After last assigned layer: sends intermediate tensor to next node via TCP.
  - Next node receives, continues computation.
  - Final node returns logits to the coordinator.
- **Auto-discovery:** Nodes broadcast UDP on local network to find peers.

---

## PHASE 29: WASM Secure Execution — Agent Sandbox

Replaces raw `popen()` with an isolated WebAssembly sandbox for safe AI tool execution.

### 29.1 — WASM Runtime Integration
- File: `src/agent/wasm_sandbox.c`
- Embeds `wasm3` (lightweight WASM interpreter, single-file C).
- `TernaryError wasm_sandbox_init(WASMSandbox *ws);`
- `TernaryError wasm_sandbox_exec(WASMSandbox *ws, const char *code, char *output, int output_size, int timeout_ms);`
  - Compiles + runs code inside WASM VM.
  - Cannot access host filesystem, network, or memory.
  - Timeout: kills execution after `timeout_ms`.
- `void wasm_sandbox_destroy(WASMSandbox *ws);`

### 29.2 — Sandboxed Command Executor
- File: `src/agent/sandboxed_exec.c`
- `TernaryError sandboxed_exec_command(WASMSandbox *ws, const ExecPolicy *pol, const char *command, char *output, int output_size, int *exit_code);`
  - Routes low-risk commands (ls, cat, echo) through sandbox.
  - Routes pre-approved commands through `popen()` with policy check.
  - Blocks all other commands.
- Integrates with Phase 14.4 agentic loop as a drop-in replacement for `exec_run_command()`.

---

## PHASE 30: Test-Time Compute (MCTS & PRM) — Advanced Reasoning

Replaces linear Chain-of-Thought with tree-search reasoning for mathematically proven answers.

### 30.1 — Process Reward Model (PRM)
- File: `include/reasoning/prm.h` + `src/reasoning/prm.c`
- `typedef struct { TransformerWeights weights; Config config; MappedFile mf; } RewardModel;`
- `TernaryError prm_load(RewardModel *rm, const char *path);`
- `float prm_score_step(RewardModel *rm, RunState *s, const int *context_tokens, int context_len, const int *step_tokens, int step_len, ThreadPool *tp);`
  - Scores a single reasoning step: returns a float in [0, 1].
  - Higher score = more logically sound step.

### 30.2 — Monte Carlo Tree Search Manager
- File: `include/reasoning/mcts.h` + `src/reasoning/mcts.c`
- ```c
  typedef struct MCTSNode {
      int *tokens;
      int num_tokens;
      float reward_score;
      int visit_count;
      struct MCTSNode **children;
      int num_children;
      RunState *snapshot;  // KV cache snapshot at this node
  } MCTSNode;
  ```
- `void mcts_search(MCTSNode *root, Config *p, TransformerWeights *w, Tokenizer *t, RewardModel *rm, ThreadPool *tp, int num_simulations, int max_depth);`
  - **Selection:** UCB1 formula to pick most promising branch.
  - **Expansion:** Generate N candidate next-steps from current node.
  - **Evaluation:** Score each step with PRM.
  - **Backpropagation:** Update visit counts and reward averages.

### 30.3 — MCTS-Guided Generation
- File: `src/reasoning/mcts_generate.c`
- `void generate_with_mcts(Config *p, TransformerWeights *w, RunState *s, Tokenizer *t, RewardModel *rm, ThreadPool *tp, const char *prompt, int max_tokens, int num_simulations);`
  - Generates step-by-step reasoning inside `<think>` tags.
  - At each step: runs MCTS to explore alternatives.
  - Outputs only the winning path to the user.

---

## PHASE 31: NVMe Storage Tiering (FlexGen) — Infinite Memory

Extends KV cache to SSD when physical RAM is exhausted.

### 31.1 — Asynchronous I/O Pager
- File: `include/memory/nvme_pager.h` + `src/memory/nvme_pager.c`
- **Linux:** Uses `io_uring` for async SSD read/write.
- **Fallback:** `pread()` / `pwrite()` with background threads.
- `TernaryError nvme_pager_init(NVMePager *np, const char *swap_file, size_t max_size);`
- `TernaryError nvme_page_out(NVMePager *np, int page_id, const float *data, int size);` — Async write page to SSD.
- `TernaryError nvme_page_in(NVMePager *np, int page_id, float *data, int size);` — Async read page from SSD.

### 31.2 — Tiered KV Cache Manager
- File: `include/kv_cache/tiered_kv.h` + `src/kv_cache/tiered_kv.c`
- `typedef struct { PageTableManager *hot_pages; NVMePager *cold_pager; int *page_tier; int ram_threshold_pct; } TieredKVCache;`
- **Eviction policy:** When RAM usage > 95%:
  - Identifies least-recently-accessed KV pages (LRU).
  - Async-writes them to SSD via NVMe pager.
  - Marks pages as "cold" (on SSD).
- **Retrieval:** When attention needs a cold page:
  - Async-reads from SSD.
  - Swaps with a hot page if RAM is full.
  - Prefetches adjacent pages to hide latency.

---

## PHASE 32: Test-Time Training (TTT) — Adaptive Internal Networks

Implements the TTT architecture where the model trains a tiny internal network on the input during inference.

### 32.1 — TTT Layer Structures
- File: `include/transformer/ttt_layer.h`
- ```c
  typedef struct {
      float *W_inner;        // Tiny trainable weight matrix (e.g., 64×64)
      float *grad_W;         // Gradient accumulator
      float *optimizer_state; // Adam/SGD state
      int inner_dim;
      float learning_rate;
  } TTTLayer;
  ```
- `TernaryError ttt_layer_init(TTTLayer *ttt, int inner_dim, float lr);`
- `void ttt_layer_free(TTTLayer *ttt);`

### 32.2 — Mini-Backpropagation Engine
- File: `src/transformer/ttt_backprop.c`
- `void ttt_forward_and_train(TTTLayer *ttt, const float *input, float *output, int dim);`
  - Forward: `output = input @ W_inner`.
  - Loss: self-supervised reconstruction or next-token prediction on the input.
  - Backward: compute gradient of loss w.r.t. `W_inner`.
  - Update: `W_inner -= lr * grad_W` (SGD) or Adam step.
- **Key insight:** This runs gradient descent *during inference*, making the internal network adapt to the current context in real-time.

### 32.3 — TTT-Aware Forward Pass
- File: `src/transformer/ttt_forward.c`
- Replaces the attention layer in TTT-marked layers with TTT layer forward+train.
- The model's KV cache equivalent is compressed into `W_inner` — fixed-size regardless of sequence length.

---

## PHASE 33: Cryptographic Watermarking (Toggleable)

Embeds invisible, cryptographically verifiable signatures into generated text.

### 33.1 — Watermark Key Management
- File: `include/sampling/watermark.h`
- ```c
  typedef struct {
      uint8_t private_key[32];  // 256-bit key
      float delta;              // Logit bias magnitude (e.g., 2.0)
      bool enabled;
  } WatermarkConfig;
  ```
- `void watermark_init(WatermarkConfig *wc, const uint8_t *key);`

### 33.2 — Watermark Logit Biasing
- File: `src/sampling/watermark_apply.c`
- `void watermark_bias_logits(float *logits, int vocab_size, const WatermarkConfig *wc, int prev_token);`
  - Hash: `h = HMAC-SHA256(key, prev_token)`.
  - Use hash to partition vocabulary into "green list" and "red list".
  - Add `delta` to all green-list token logits.
  - The model preferentially selects green-list tokens — invisible to humans, detectable by the algorithm.

### 33.3 — Watermark Detector
- File: `src/sampling/watermark_detect.c`
- `float watermark_detect(const int *tokens, int num_tokens, const WatermarkConfig *wc);`
  - For each token: check if it's in the green list given `prev_token`.
  - Compute z-score: proportion of green tokens vs. expected by chance.
  - z-score > 4.0 → statistically certain watermark present.
  - Returns z-score.

---

## PHASE 34: Model Import Pipeline (One-Click Converter) ✅ IMPLEMENTED

The universal Python tool that bridges HuggingFace models to the C engine's binary format.

### 34.1 — Universal Model Importer ✅
- File: `tools/import_model.py`
- CLI: `python tools/import_model.py --repo "moondream-hf/moondream2" --out models/`
- **Auto-detection pipeline (implemented):**
  1. Downloads model from HuggingFace (via huggingface_hub).
  2. Inspects weight values: if uint8 ternary packed → routes to `convert_hf_bitnet.py`.
  3. If multimodal indicators in config → routes to `extract_multimodal.py`.
  4. If standard FP16/BF16 text-only → routes to `convert_hf.py`.

### 34.2 — GGUF Format Reader ✅
- File: `include/core/gguf_reader.h` + `src/core/gguf_reader.c`
- Implemented: `gguf_read_header()`, `gguf_find_tensor()`, `gguf_meta_str/u32/i32()`, `gguf_type_name/bpe()`
- Supports GGUF v1, v2, v3; parses all metadata types; resolves tensor data pointers into mmap.
- Auto-detects multimodal models from vision tensor name prefixes.
- Deferred: full weight loading for float16 LLM inference (requires float transformer path).

### 34.3 — Multimodal Model Extractor ✅
- File: `tools/extract_multimodal.py`
- CLI: `python tools/extract_multimodal.py --repo moondream-hf/moondream2 --out models/`
- Supported models: **Moondream2**, **SmolVLM-256M/500M**, **LLaVA-1.5** (extensible adapter pattern)
- Outputs: `vision.bin` (SigLIP/ViT encoder, float32), `projector.bin` (MLP projector, float32)
- Binary format documented in `include/multimodal/vision_weights_load.h`

### 34.4 — Vision Weight Loader (C) ✅ (new, not in original plan)
- File: `include/multimodal/vision_weights_load.h` + `src/multimodal/vision_weights_load.c`
- `vision_model_load_encoder()` — mmap vision.bin → VisionWeights (zero-copy)
- `vision_model_load_projector()` — mmap projector.bin → VisionProjector (zero-copy)
- `vision_model_print_info()`, `vision_model_free()`

### 34.5 — CLI Integration ✅ (new, not in original plan)
- `--vision <path>` and `--proj <path>` flags added to args.h / args.c
- `main.c` wired: when `--image` + `--vision` + `--proj` all provided:
  - Loads real vision weights → runs full pipeline → injects into KV cache → generates
- When `--vision`/`--proj` absent, `--image` prints actionable error with extraction command

### End-to-End Usage
```bash
# Step 1: Extract weights (one-time)
python tools/extract_multimodal.py --repo moondream-hf/moondream2 --out models/

# Step 2: Run with image
./adaptive_ai_engine \
    --model  models/bitnet-b1.58-2B-4T.bin \
    --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
    --vision models/vision.bin \
    --proj   models/projector.bin \
    --image  strawberry.jpg \
    --prompt "What do you see in this image?"
```

> **Note:** The LLM text backbone (BitNet b1.58) was not trained on image tokens.
> Full multimodal recognition requires a multimodal-trained LLM backbone.
> Phase 34.2 GGUF full weight loading (float16 inference path) is the next step.

---

## Updated Dependency Graph (Full Architecture)

```
Phase 0 (Scaffold)
    └─→ Phase 2.6 (Aligned Allocator) ← MUST EXIST BEFORE ANY BUFFER ALLOC
         └─→ Phase 1 (Structs — RunState uses aligned alloc + transposed KV)
              ├─→ Phase 2.1–2.5 (Memory: mmap, RAM probe, budget) ──────┐
              ├─→ Phase 3.1,3.7–3.10 (Scalar Math Primitives) ─────────┤
              │    └─→ Phase 4 (Threading) ────────────────────────────┤
              ├─→ Phase 5 (Tokenizer) ────────────────────────────────┤
              │                                                        ▼
              └──────────────────────────────────→ Phase 6 (Transformer Forward Pass)
                                                  │ (uses transposed KV layout)
                                       ┌──────────┤
                                       ▼          ▼
                                  Phase 7      Phase 8
                                 (Sampling)  (KV Optimize)
                                       │          │
                                       ▼          ▼
                                  Phase 9 (Reasoning / Think Tags)
                                       │
                            ┌──────────┼──────────┐
                            ▼          │          ▼
                      Phase 10.1–10.3  │    Phase 11.1–11.3
                     (Converters)      │   (Vision Encoder)
                            │          │          │
                            ▼          │          ▼
                      Phase 10.4–10.5  │    Phase 11.4 ← MLP PROJECTOR
                     (LUT Unpackers +  │          │
                      Fused MatMul)    │          ▼
                            │          │    Phase 11.5
                            │          │   (Vision Bridge)
                            ▼          ▼          │
                      Phase 3.2–3.6 ──────────────┤
                     (SIMD Variants +             │
                      Dispatch)                   │
                            │                     │
                            └──────────┬──────────┘
                                       ▼
                                 Phase 12 (CLI + Main)
                                       │
                    ┌──────────────────┤
                    ▼                  ▼
              Phase 14           Phase 15
            (Agentic Tools)    (RAG / Vector DB)
                    │                  │
                    └──────────┬───────┘
                               ▼
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Phase 17         Phase 18         Phase 19
       (MoE Router)   (Speculative)    (LoRA Adapters)
              │                │                │
              └────────┬───────┘                │
                       ▼                        │
                 Phase 20                       │
             (Grammar/FSM) ◄────────────────────┘
                       │
                       ▼
                 Phase 21 (OpenAI API Layer)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   Phase 22       Phase 23      Phase 24
  (SSM/Mamba)  (PagedAttention)  (YaRN/NTK)
         │             │             │
         └──────┬──────┘             │
                ▼                    │
          Phase 25 ◄─────────────────┘
        (Native Audio)
                │
         ┌──────┼──────┐
         ▼      ▼      ▼
   Phase 26  Phase 27  Phase 28
  (Radix    (Cache   (Distributed
   Cache)   Tiling)   Inference)
         │      │      │
         └──────┼──────┘
                ▼
         ┌──────┼──────┐
         ▼      ▼      ▼
   Phase 29  Phase 30  Phase 31
   (WASM    (MCTS/   (NVMe
  Sandbox)   PRM)    Tiering)
         │      │      │
         └──────┼──────┘
                ▼
         ┌──────┼──────┐
         ▼      ▼      ▼
   Phase 32  Phase 33  Phase 34
   (TTT)   (Watermark) (Import
                        Pipeline)
                │
                ▼
          Phase 35 (GGUF MLA+MoE Architecture Router)
                │
                ▼
          Phase 36 (Text-to-Image: Stable Diffusion / Flux)
                │
                ▼
          Phase 13 (Tests — 13.1–13.30+)
                │
                ▼
          Phase 16-D (Docs)
```

---

## Implementation Order (Recommended)

For fastest path to a working text-generation demo:

1. **0.1 → 0.2 → 0.4 → 0.5** — Scaffold + build system
2. **2.6** — Aligned allocator (must exist before any buffer allocation)
3. **1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6** — All structs (RunState now uses aligned alloc + transposed KV layout)
4. **2.1 → 2.2** — mmap (POSIX only, skip Windows initially)
5. **3.1 → 3.7 → 3.8 → 3.9 → 3.10** — Scalar math (no SIMD yet)
6. **5.1 → 5.2 → 5.3 → 5.4** — Tokenizer
7. **6.1 → 6.2 → 6.4 → 6.5** — Forward pass (attention uses transposed KV layout)
8. **7.1 → 7.5 → 7.6** — Basic generation (argmax only)
9. **10.1 → 10.2 → 10.3** — Weight packer + converters
10. **12.3** — Minimal main.c, test with real model (**first working demo**)
11. **4.1 → 4.2 → 4.3 → 4.4** — Thread pool (instant speedup)
12. **10.4** — SIMD shuffle LUT unpackers (AVX2/AVX-512/NEON)
13. **10.5 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6** — Fused packed matmul + all SIMD variants + dispatch
14. **2.4 → 2.5 → 8.4** — Hardware adaptation (RAM probe + budget + strategy)
15. **8.1 → 8.2 → 8.3** — KV compression (int8 + int4) + sliding window
16. **9.1 → 9.2 → 9.3** — Reasoning engine
17. **7.2 → 7.3 → 7.4** — Advanced sampling (temperature, top-p, top-k)
18. **12.1 → 12.2 → 12.4** — Full CLI + REPL + perf timer
19. **11.1 → 11.2 → 11.3 → 11.4 → 11.5** — Multimodal (image load → patches → vision encoder → **projector** → bridge)
20. **14.1 → 14.2 → 14.3 → 14.4 → 14.5** — Agentic tools (tag interceptor → cmd executor → output inject → agent loop → user approval)
21. **15.1 → 15.2 → 15.3 → 15.4 → 15.5 → 15.6** — RAG (embedder → cosine sim → vector DB → search → auto-retrieve → auto-save)
22. **17.1 → 17.2 → 17.3 → 17.4** — MoE routing (config → gate → FFN → weights)
23. **18.1 → 18.2 → 18.3** — Speculative decoding (draft model → spec loop → KV rollback)
24. **19.1 → 19.2 → 19.3 → 19.4** — LoRA adapters (structs → loader → fused matmul → converter)
25. **20.1 → 20.2 → 20.3** — Grammar-constrained decoding (grammar → FSM → constrained sampling)
26. **21.1 → 21.2 → 21.3 → 21.4 → 21.5** — OpenAI API (HTTP server → JSON parse → chat template → SSE → prefix cache)
27. **22.1 → 22.2 → 22.3 → 22.4** — SSM/Mamba (arch detect → state vector → selective scan → universal router)
28. **23.1 → 23.2 → 23.3 → 23.4** — PagedAttention (page table → paged KV → scheduler → shared prefix)
29. **24.1 → 24.2** — YaRN/NTK context scaling
30. **25.1 → 25.2 → 25.3** — Native audio (codec → WebSocket → injection)
31. **26.1 → 26.2 → 26.3** — Radix tree prompt caching
32. **27.1 → 27.2 → 27.3** — CPU cache tiling (L1 probe → tiled matmul → flash attention CPU)
33. **28.1 → 28.2 → 28.3** — Distributed inference (layer plan → TCP transport → dist forward)
34. **29.1 → 29.2** — WASM sandbox (runtime → sandboxed executor)
35. **30.1 → 30.2 → 30.3** — MCTS/PRM reasoning (reward model → tree search → guided generation)
36. **31.1 → 31.2** — NVMe storage tiering (async pager → tiered KV cache)
37. **32.1 → 32.2 → 32.3** — Test-Time Training (TTT layers → backprop → TTT forward)
38. **33.1 → 33.2 → 33.3** — Cryptographic watermarking (keys → biasing → detection)
39. **34.1 → 34.2 → 34.3** — Model import pipeline (importer → GGUF reader → multimodal extractor)
40. **13.1–13.30+** — All tests
41. **16.1 → 16.2 → 16.3** — Docs

---

**Total modules: ~185 discrete, independently testable units.**
**Total source files: ~130 .c files + ~80 .h headers + 5 Python scripts.**
**Zero external C dependencies at the core. Pure C99 + POSIX. Optional: mongoose.c, cJSON.h, wasm3, stb_image.h (all single-file, embeddable).**

---

## ERRATA — Critical Audit Findings

This section tracks bugs and architectural flaws found during code review. Items
are marked **RESOLVED** once the fix is applied in source, or **OPEN** if the fix
is deferred to the phase where the code is first implemented.

### RESOLVED

#### 1. Integer Overflow in KV_CACHE_IDX Macro (Phase 1.5)

**Severity:** Critical — segfault at long context lengths.

The original macro performed all arithmetic in `int` (32-bit signed). For a
32-layer, 8-KV-head, 128-dim model at 128K context:
`32 * 8 * 128000 * 128 = 4,194,304,000` — exceeds INT_MAX (2,147,483,647).

**Fix applied:** Cast leading terms to `size_t` in `include/core/run_state.h`:
```c
#define KV_CACHE_IDX(layer, head, pos, d, n_kv_heads, max_seq, head_dim) \
    ((size_t)(layer) * (n_kv_heads) * (max_seq) * (head_dim) + \
     (size_t)(head) * (max_seq) * (head_dim) + \
     (size_t)(pos) * (head_dim) + (d))
```

#### 2. RoPE Integer Division (Phase 3.9)

**Severity:** Critical — destroys positional encoding.

The plan document showed `2*i/head_dim` using integer division, which truncates
to 0 for all values of `i < head_dim/2`.

**Fix applied during Phase 3 implementation:** `src/math/rope.c` already uses
`(float)i / (float)head_dim`, so the actual code was never affected.

#### 3. CMake ARM/Apple Silicon Build Crash

**Severity:** Critical — fatal compiler error on non-x86 hosts.

`ENABLE_AVX2` defaulted to `ON` unconditionally. Building on M1/M2/M3 Macs or
ARM boards injected `-mavx2 -mfma` into Clang, which ARM does not support.

**Fix applied in CMakeLists.txt:** Architecture auto-detection via
`CMAKE_SYSTEM_PROCESSOR` sets defaults per ISA family (x86 → AVX2, ARM → NEON).

#### 4. CMake GLOB_RECURSE Source Collection

**Severity:** Major — silent link failures when adding new files.

`file(GLOB_RECURSE SOURCES "src/*.c")` does not re-run CMake when files are
added or removed, causing stale build lists.

**Fix applied in CMakeLists.txt:** Replaced with explicit source listing
grouped by subsystem (`CORE_SOURCES`, `MATH_SOURCES`, `MEMORY_SOURCES`,
`THREADING_SOURCES`). Test discovery still uses GLOB with `CONFIGURE_DEPENDS`
(CMake 3.12+) since test files change less frequently.

**Rule:** Every time a new `.c` file is created under `src/`, it MUST be added
to the corresponding `set(...)` block in `CMakeLists.txt`.

#### 5. CMake Threading Portability

**Severity:** Minor — fails on Alpine Linux, minimal containers.

Hardcoded `pthread` link. Replaced with `find_package(Threads REQUIRED)` +
`Threads::Threads` for portable detection.

#### 6. CMake Debug Sanitizer Linking for Main Target

**Severity:** Minor — debug build link failure for main executable.

`-fsanitize=address` was set in `CMAKE_C_FLAGS_DEBUG` (compiler) but not in
the linker flags for the main target. Added `CMAKE_EXE_LINKER_FLAGS_DEBUG`.

#### 6b. FM-002: Integer Overflow in KV Cache Size (Phase 1.6)

**Severity:** Medium — silent truncation on 32-bit hosts, potential OOB writes.

The KV cache allocation `(size_t)cfg->n_layers * cfg->n_kv_heads * max_seq_len * head_dim`
had no overflow detection. While C's left-associative arithmetic promotes to `size_t` on
64-bit (safe for current configs), on 32-bit hosts or with future larger configs, the
intermediate products could overflow silently.

**Fix applied:** Added `tn_size_mul4()` overflow-checked multiplication in
`include/memory/aligned_alloc.h`. `run_state_alloc` now returns `TN_ERR_OOM` on overflow.

#### 6c. FM-005: Overflow in att Buffer Allocation (Phase 1.6)

**Severity:** Low — same pattern as FM-002 for `s->att` allocation.

**Fix applied:** Uses `tn_size_mul_overflow()` for the `n_heads * max_seq_len` multiplication.

#### 6d. FM-006: Deadlock in Thread Pool on Repeated Dispatches (Phase 4.3)

**Severity:** Critical — complete denial of service on second sequential dispatch.

The initial FM-001 fix used `while (task_fn != NULL)` in the inner wait. If a new dispatch
started before parked threads exited, `task_fn` was non-NULL again and threads stayed trapped
forever, causing a deadlock. The epoch-based fix (see 4.3 above) resolves this completely.

**Fix applied in:** `src/threading/thread_pool.c` — epoch counter distinguishes dispatches.
Verified with 100+ sequential dispatch stress tests.

#### 6e. NEW-01: Non-Reentrant Global in Tokenizer (Phase 5.2)

**Severity:** Low — data race if two tokenizers loaded concurrently.

`g_vocab_ptr` file-scope global was used by `qsort` comparator. Replaced with a portable
reentrant quicksort that takes the vocab pointer as an explicit parameter. Global removed.

**Fix applied in:** `src/tokenizer/tokenizer_load.c`.

#### 6f. RoPE Hot Loop powf() Redundancy (Phase 3.9)

**Severity:** Major — O(layers × heads × head_dim) redundant `powf()` calls per token.

The `apply_rope()` function computed `powf(10000.0f, (float)i / (float)head_dim)`
inside the innermost loop. For a 32-layer, 32-head, 128-dim model, this produced
65,536 `powf()` calls per token — despite the frequency values being constant
(they depend only on `i` and `head_dim`, not on position or head index).

**Fix applied:**
- Added `rope_precompute_freqs()` to `src/math/rope.c` — computes the freq
  table once into a flat `float[head_dim/2]` array.
- Added `float *rope_freq` + `int rope_freq_len` fields to `RunState`
  (`include/core/run_state.h`).
- `run_state_alloc()` now precomputes and caches the table at model boot.
- `apply_rope()` signature changed to accept `const float *freq` — the hot
  loop now does only `pos * freq[i]` followed by `cosf/sinf`.

### OPEN — Deferred to Implementation Phase

#### 7. Vector DB Append-Only Layout (Phase 15.3)

**Severity:** Major — O(n) data shift on every insert.

The planned binary layout `[all embeddings][all text_lengths][all text_data]`
cannot support O(1) appends. Inserting a new embedding requires `memmove` of
all subsequent blocks.

**Required fix (Phase 15):** Interleave records:
```
[Header]
Record 1: [embedding][text_length][text_data]
Record 2: [embedding][text_length][text_data]
...
```

#### 8. Agent Sandbox Blocklist Bypass (Phase 14.2)

**Severity:** Critical — trivial shell injection.

A blocklist on `popen()` is bypassable via shell quoting, variable expansion,
and aliasing (e.g., `r"m" -rf /`, `bash -c "rm -rf /"`).

**Required fix (Phase 14):** Use a strict allowlist of permitted commands.
Prefer `execvp()` (no shell) over `popen()` (spawns `/bin/sh`).

#### 9. Watermark Single-Token Entropy (Phase 33.2)

**Severity:** Major — watermark is statistically detectable and trivially
removable.

Hashing only one previous token (`h = HMAC-SHA256(key, prev_token)`) makes the
green list deterministic after any given word, destroying entropy.

**Required fix (Phase 33):** Hash a sliding window of tokens:
`h = HMAC-SHA256(key, concat(token[t-5], ..., token[t-1]))`.

---

## Phase 35 — GGUF MLA + MoE Architecture Router

**Motivation:** DeepSeek-V2-Lite and future MoE models use a fundamentally
different GGUF tensor layout that cannot be handled by the generic Llama weight
loader. Phase 35 adds an architecture-specific dispatch layer inside
`gguf_loader.c` that routes weight loading based on `general.architecture`.

### 35.1 — Architecture Dispatch in gguf_loader.c

Add `gguf_arch_type()` helper that returns an enum:
```c
typedef enum { GGUF_ARCH_LLAMA, GGUF_ARCH_DEEPSEEK2, GGUF_ARCH_UNKNOWN } GGUFArch;
```
`weights_from_gguf()` dispatches to `weights_from_gguf_deepseek2()` for DeepSeek.

### 35.2 — MLA Weight Mapping for deepseek2

DeepSeek-V2-Lite uses Multi-head Latent Attention. Tensor names differ from Llama:

| Standard Llama         | DeepSeek-V2-Lite             | Meaning                        |
|------------------------|------------------------------|--------------------------------|
| `blk.N.attn_q.weight`  | `blk.N.attn_q.weight`        | Q projection (single matrix)   |
| `blk.N.attn_k.weight`  | `blk.N.attn_kv_a_mqa.weight` | KV-A compress (→ kv_lora_rank) |
| `blk.N.attn_v.weight`  | `blk.N.attn_kv_b.weight`     | KV-B expand (→ full K+V)       |
| `blk.N.attn_norm.w`    | `blk.N.attn_kv_a_norm.weight`| Norm after KV-A compress       |
| `blk.N.attn_output.w`  | `blk.N.attn_output.weight`   | Output projection (same name)  |

New fields in `TransformerWeights`:
```c
float **mla_wq_f32;       /* [n_layers] Q projection (dim × dim)              */
float **mla_wkv_a_f32;    /* [n_layers] KV-A compress (dim × kv_lora_rank)    */
float **mla_wkv_b_f32;    /* [n_layers] KV-B expand (kv_lora_rank → full K+V) */
float **mla_wkv_a_norm;   /* [n_layers] norm weight after KV-A                */
```

### 35.3 — Stacked MoE Expert Weights

DeepSeek stores all expert weights for a layer as a single stacked tensor:
- `blk.N.ffn_gate_exps.weight` — shape `[num_experts × expert_hidden, dim]` (Q4_K)
- `blk.N.ffn_up_exps.weight`   — shape `[num_experts × expert_hidden, dim]` (Q4_K)
- `blk.N.ffn_down_exps.weight` — shape `[num_experts × dim, expert_hidden]` (Q5_1)

The loader slices these into `moe_w1[layer][e]`, `moe_w3[layer][e]`, `moe_w2[layer][e]`
using stride arithmetic — no data copy needed (zero-copy slicing into the mmap).

### 35.4 — Additional Quant Types

DeepSeek uses quant types not yet implemented:

| Type  | ggml_type | Block size | Description           |
|-------|-----------|------------|-----------------------|
| Q5_1  | 7         | 24 bytes   | 5-bit + min scale     |
| Q5_K  | 13        | 176 bytes  | 5-bit K-quant         |
| Q6_K  | 14        | 210 bytes  | 6-bit K-quant         |

Implement `gguf_dequant_q5_1()`, `gguf_dequant_q5_k()`, `gguf_dequant_q6_k()`
in `src/core/gguf_quant.c` and expose them in `include/core/gguf_quant.h`.

### 35.5 — Float32 Matmul Path in MLA Attention

`mla_attention.c` currently calls `parallel_ternary_matmul_packed()` which
expects 2-bit packed weights. For GGUF-loaded float32 weights, add:
```c
static void f32_matvec(float *out, const float *x, const float *w, int in_n, int out_n, ThreadPool *tp);
```
`mla_attention_forward()` checks `w->mla_wq_f32 != NULL` and routes accordingly.

### 35.6 — Shared Expert FFN

DeepSeek has 2 shared experts that always fire (in addition to top-6 routed):
- `blk.N.ffn_gate_shexp.weight` / `ffn_up_shexp.weight` / `ffn_down_shexp.weight`
- These go through standard SwiGLU regardless of routing
- Add `moe_shared_w1[layer]`, `moe_shared_w2[layer]`, `moe_shared_w3[layer]`

**Files to create/modify:**
- `src/core/gguf_loader.c` — architecture dispatch + deepseek2 weight loader
- `src/core/gguf_quant.c` — add Q5_1, Q5_K, Q6_K dequant
- `include/core/gguf_quant.h` — declare new dequant functions
- `include/core/weights.h` — add float32 MLA weight pointer arrays
- `src/transformer/mla_attention.c` — add f32_matvec path + shared expert handling

---

## Phase 36 — Text-to-Image: Stable Diffusion / Flux

**Motivation:** Extend the engine beyond text generation into image synthesis.
The architecture document describes two target models and a clear hardware
tiering strategy based on available RAM.

### Architecture Overview

```
Text prompt
    │
    ▼
[CLIP/T5 Text Encoder]  ← already supported (tokenizer + transformer)
    │ text_embeddings [77 × 768]
    ▼
[Diffusion U-Net / DiT]  ← NEW: iterative denoising loop
    │  noise_pred [latent_H × latent_W × 4]
    ▼
[DDIM/DPM Scheduler]  ← NEW: 20-50 denoising steps
    │  clean_latents
    ▼
[VAE Decoder]  ← NEW: latent space → pixel space
    │
    ▼
PNG / JPEG image output
```

### 36.1 — RAM-Tiered Model Selection

| Available RAM | Recommended Model           | Resolution  | Steps | Latency |
|---------------|-----------------------------|-------------|-------|---------|
| ≥ 6 GB        | Flux-Schnell Q4_K (3.5 GB)  | 512 × 512   | 4     | ~45s    |
| ≥ 4 GB        | SD-LCM Q4_K (2.0 GB)        | 512 × 512   | 4     | ~30s    |
| < 4 GB        | SD-Turbo Q8_0 (1.6 GB)      | 256 × 256   | 1     | ~15s    |

### 36.2 — New Source Files

```
src/diffusion/scheduler.c       — DDIM / DPM-Solver++ step functions
src/diffusion/unet.c            — U-Net / DiT forward pass (residual blocks)
src/diffusion/vae_decoder.c     — latent → pixel upsampling
src/diffusion/clip_encoder.c    — text prompt → embeddings
include/diffusion/scheduler.h
include/diffusion/unet.h
include/diffusion/vae_decoder.h
```

### 36.3 — CLI Integration

```
./adaptive_ai_engine --text2img "a red apple on a wooden table" \
                     --model models/flux-schnell-Q4_K.gguf \
                     --output output.png \
                     --steps 4 --width 512 --height 512
```

### 36.4 — Architecture Notes

- Flux-Schnell is a DiT (Diffusion Transformer) — shares transformer block
  infrastructure with Phase 6 but uses timestep conditioning and cross-attention
  on text embeddings, not autoregressive token generation.
- SD-LCM uses a U-Net with residual blocks and skip connections — different from
  transformer; needs dedicated conv2d + upsampling primitives.
- VAE decoder uses 2D convolutions — add `src/math/conv2d.c` with optimised
  3×3 depthwise SIMD kernels.
- No KV cache, no tokenizer (uses separate CLIP encoder vocab).
- Memory layout: latent tensors are small (64×64×4 = 16K floats); U-Net activations
  peak at ~200 MB for 512×512 — easily fits on 16-core machines.

### 36.5 — Test Plan

| Test                    | Assertion                             |
|-------------------------|---------------------------------------|
| `test_scheduler_step`   | DDIM step noise → less noise          |
| `test_vae_range`        | Decoded pixels in [0, 255] range      |
| `test_flux_coherent`    | PSNR vs reference > 20 dB             |
| `test_t2i_no_crash`     | Full pipeline with 2-step schedule    |

**Note:** Phase 36 depends on Phase 34 (GGUF pipeline) and Phase 35 (MLA router)
being complete, as model weights are loaded via the GGUF path.

---

## Phase 37: GGUF Universal Quant Compatibility

> **Status:** Q2_K done. All remaining items are pending.
> **Goal:** Engine accepts any GGUF file without "unsupported quant type" errors.
> **Constraint:** Add modularly — new functions in `gguf_quant.c`/`gguf_quant.h`,
> new cases in `gguf_loader.c` dispatch + `quant_bytes_for_elems`.
> Only touch `gguf_reader.c` if `gguf_block_size()` has a wrong byte count (as was the
> case for Q2_K: returned 256, correct is 84).

### Files changed per task
| File | What changes |
|---|---|
| `include/core/gguf_quant.h` | Add `gguf_dequant_<type>()` declaration |
| `src/core/gguf_quant.c` | Implement dequant function |
| `src/core/gguf_loader.c` | Add `case GGUF_TYPE_<X>:` in dispatch switch + `quant_bytes_for_elems()` |
| `src/core/gguf_reader.c` | Fix `gguf_block_size()` only if the byte count is wrong |

---

### 37.1 — Q2_K ✅ (done 2026-04-03)
- Block: 256 elems, 84 bytes (`scales[16]` + `qs[64]` + `d fp16` + `dmin fp16`)
- Bug fixed: `gguf_block_size(Q2_K)` was returning 256 (elem count), corrected to 84 (byte count)
- Function: `gguf_dequant_q2_k()` — mirrors `dequantize_row_q2_K` from llama.cpp
- `quant_bytes_for_elems`: `(n / 256) * 84`
- Needed for: DeepSeek-V2-Lite-Chat-Q2_K, any model quantized to Q2_K

---

### 37.2 — Q3_K ❌ pending
- Block: 256 elems, 110 bytes
  - `qs[64]`   — 2-bit low values (4/byte)
  - `hmask[32]` — 1 high bit per element (8/byte)
  - `scales[12]` — 16 sub-blocks × 6-bit scale, packed 2 per byte
  - `d fp16`
- Function to add: `gguf_dequant_q3_k()`
- Algorithm: for each elem, `raw3 = (qs_low2 | (hmask_bit << 2))`, then `val = (raw3 - 4) * scale * d`
- `quant_bytes_for_elems`: `(n / 256) * 110`
- `gguf_block_size`: already correct (110) — no reader change needed
- Needed for: any model quantized to Q3_K_S / Q3_K_M / Q3_K_L

---

### 37.3 — Q4_1 ❌ pending
- Block: 32 elems, 20 bytes (`d fp16` + `m fp16` + `qs[16]` 4-bit)
- Function to add: `gguf_dequant_q4_1()`
- Algorithm: `out[i] = (nibble[i] * d) + m` (no zero-point offset, uses additive min instead)
- `quant_bytes_for_elems`: `(n / 32) * 20`
- `gguf_block_size`: already correct (20) — no reader change needed
- Needed for: older Llama/GPT-J Q4_1 GGUF files

---

### 37.4 — Q8_1 ❌ pending
- Block: 32 elems, 36 bytes (`d fp32` + `s fp32` + `qs[32]` int8)
  - Note: `d` and `s` are fp32 here, not fp16 (unlike Q8_0)
  - `s` = sum of all quantised values × d (used for dot-product shortcuts in llama.cpp, not needed for plain dequant)
- Function to add: `gguf_dequant_q8_1()`
- Algorithm: `out[i] = qs[i] * d` (same as Q8_0 but d is fp32 at offset 0, s at offset 4)
- `quant_bytes_for_elems`: `(n / 32) * 36`
- `gguf_block_size`: already correct (36) — no reader change needed
- Needed for: intermediate activation tensors in some models stored as Q8_1

---

### 37.5 — Q8_K ❌ pending
- Block: 256 elems, 292 bytes (`d fp32` + `qs[256]` int8 + `bsums[16]` int16)
  - `bsums` = block sums for GEMM shortcuts; not needed for plain dequant
- Function to add: `gguf_dequant_q8_k()`
- Algorithm: `out[i] = qs[i] * d`
- `quant_bytes_for_elems`: `(n / 256) * 292`
- `gguf_block_size`: already correct (292) — no reader change needed
- Needed for: K-quant intermediate/activation tensors

---

### 37.6 — IQ2_XXS ❌ pending
- Block: 256 elems, 66 bytes (uses 8-entry codebook lookup; ~2.06 bits/weight)
  - Layout: `qs[32]` (packed grid indices) + `scales[4]` + `d fp16`
  - Each 2-byte `qs` chunk encodes 8 elements via a 256-entry lookup table
- Function to add: `gguf_dequant_iq2_xxs()`
- Algorithm: requires static `iq2xxs_grid[256]` lookup table (copy from llama.cpp `ggml-quants.c`)
- Type enum: `GGUF_TYPE_IQ2_XXS = 16` already defined in reader header
- Add `GGUF_TYPE_IQ2_XXS` to `GGUFType` enum in `gguf_reader.h` (or verify it's there)
- `quant_bytes_for_elems`: `(n / 256) * 66`
- `gguf_block_size`: add `case GGUF_TYPE_IQ2_XXS: return 66;` in reader
- Needed for: ultra-low-bit imatrix models

---

### 37.7 — IQ2_XS ❌ pending
- Block: 256 elems, 74 bytes (~2.31 bits/weight)
  - Layout: `qs[32]` + `scales[8]` (4-bit per 32-elem sub-block) + `d fp16`
- Function: `gguf_dequant_iq2_xs()`
- Requires `iq2xs_grid[512]` codebook (copy from llama.cpp)
- `quant_bytes_for_elems`: `(n / 256) * 74`
- Enum value: 17 — add `GGUF_TYPE_IQ2_XS = 17` to enum

---

### 37.8 — IQ3_XXS ❌ pending
- Block: 256 elems, 98 bytes (~3.06 bits/weight)
  - Layout: `qs[96]` + `scales[4]` + `d fp16`
- Function: `gguf_dequant_iq3_xxs()`
- Requires `iq3xxs_grid[256]` codebook
- `quant_bytes_for_elems`: `(n / 256) * 98`
- Enum value: 18 — add `GGUF_TYPE_IQ3_XXS = 18`

---

### 37.9 — IQ4_NL ❌ pending
- Block: 32 elems, 18 bytes (non-linear 4-bit; same byte count as Q4_0)
  - Layout: `d fp16` + `qs[16]` 4-bit, but uses a 16-entry non-linear codebook
- Function: `gguf_dequant_iq4_nl()`
- Requires `kvalues_iq4nl[16]` lookup table (16 signed int8 values, copy from llama.cpp)
- `quant_bytes_for_elems`: `(n / 32) * 18`
- Enum value: 20 — add `GGUF_TYPE_IQ4_NL = 20`
- Note: uses same block size as Q4_0 (32 elems, 18 bytes) but different decode

---

### 37.10 — IQ3_S ❌ pending
- Block: 256 elems, 110 bytes (same byte count as Q3_K)
  - Layout: `qs[96]` (3-bit packed) + `qh[16]` (1 extra bit/elem) + `signs[32]` + `scales[4]` + `d fp16`
- Function: `gguf_dequant_iq3_s()`
- Requires `iq3s_grid[512]` codebook
- `quant_bytes_for_elems`: `(n / 256) * 110`
- Enum value: 21 — add `GGUF_TYPE_IQ3_S = 21`

---

### 37.11 — IQ2_S ❌ pending
- Block: 256 elems, 82 bytes (~2.5 bits/weight)
  - Layout: `qs[32]` + `qh[16]` (high bits) + `signs[16]` + `scales[4]` + `d fp16`
- Function: `gguf_dequant_iq2_s()`
- Requires `iq2s_grid[512]` codebook
- `quant_bytes_for_elems`: `(n / 256) * 82`
- Enum value: 22 — add `GGUF_TYPE_IQ2_S = 22`

---

### 37.12 — IQ4_XS ❌ pending
- Block: 256 elems, 136 bytes (~4.25 bits/weight)
  - Layout: like Q4_K but uses non-linear `kvalues_iq4nl` codebook + 6-bit scales
  - `qs[128]` + `scales[12]` + `d fp16`
- Function: `gguf_dequant_iq4_xs()`
- Requires `kvalues_iq4nl[16]` codebook (same as IQ4_NL)
- `quant_bytes_for_elems`: `(n / 256) * 136`
- Enum value: 23 — add `GGUF_TYPE_IQ4_XS = 23`

---

### 37.13 — IQ1_S ❌ pending
- Block: 256 elems, 26 bytes (~1.56 bits/weight)
- Requires `iq1s_grid[2048]` codebook + `kvalues_iq1b[8]` sign table
- `quant_bytes_for_elems`: `(n / 256) * 26`
- Enum value: 19 — add `GGUF_TYPE_IQ1_S = 19`
- Note: extreme compression; quality is poor but useful for RAM-constrained inference

---

### 37.14 — IQ1_M ❌ pending
- Block: 256 elems, 37 bytes (~1.75 bits/weight)
- Requires `iq1s_grid[2048]` codebook (same grid as IQ1_S) + 4-bit scales
- `quant_bytes_for_elems`: `(n / 256) * 37`
- Enum value: 29 — add `GGUF_TYPE_IQ1_M = 29`

---

### 37.15 — Enum completeness ❌ pending
The `GGUFType` enum in `include/core/gguf_reader.h` is missing:
```c
GGUF_TYPE_IQ2_XS   = 17,
GGUF_TYPE_IQ3_XXS  = 18,
GGUF_TYPE_IQ1_S    = 19,
GGUF_TYPE_IQ4_NL   = 20,
GGUF_TYPE_IQ3_S    = 21,
GGUF_TYPE_IQ2_S    = 22,
GGUF_TYPE_IQ4_XS   = 23,
GGUF_TYPE_IQ1_M    = 29,
```
Add these before the `GGUF_TYPE_COUNT` sentinel.
Also add corresponding `gguf_block_size()`, `gguf_block_elems()`, and `gguf_type_name()`
cases in `src/core/gguf_reader.c` for each new type.

---

### 37.16 — BF16 standalone dequant function ❌ pending (low priority)
BF16 decode is currently inlined in `gguf_loader.c`. For consistency and
testability, extract to `gguf_dequant_bf16()` in `gguf_quant.c`.
No correctness change — purely a refactor.

---

### Implementation order (by impact)
1. 37.2 Q3_K — needed for Q3_K_S/M/L quantized models (very common)
2. 37.3 Q4_1 — legacy format, widely seen in older GGUF files
3. 37.4 Q8_1 / 37.5 Q8_K — needed if model uses these for embedding/norm tensors
4. 37.9 IQ4_NL / 37.12 IQ4_XS — popular imatrix 4-bit (small codebook, easy)
5. 37.6 IQ2_XXS … 37.14 IQ1_M — ultra-low-bit imatrix (need codebook tables from llama.cpp)
6. 37.15 Enum completeness — needed for all IQ types to load at all
7. 37.16 BF16 refactor — last, cosmetic only

### Reference for codebook tables
All lookup tables (`iq2xxs_grid`, `iq2xs_grid`, `iq3xxs_grid`, `iq3s_grid`,
`iq2s_grid`, `kvalues_iq4nl`, `iq1s_grid`) are in
`llama.cpp/src/llama-quants.cpp` (MIT licence). Copy verbatim as static const
arrays into a new file `src/core/gguf_quant_tables.c` with a matching header
`include/core/gguf_quant_tables.h`. Include from `gguf_quant.c` only.
