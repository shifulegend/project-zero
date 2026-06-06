# Fix Plan: QA Audit Bug Fixes

**Date:** 2026-03-13
**Source:** Forensic QA Testing Strategy Report (26-page audit)
**Scope:** 3 latent bugs (CRITICAL/HIGH/MEDIUM) + 2 code quality issues (LOW/INFO)

---

## Bug 1: CRITICAL — Stack Buffer Overflow in `top_k.c`

**File:** `src/sampling/top_k.c`, lines 17-20
**Severity:** CRITICAL

### Problem
`int top_indices[1024]` and `float top_vals[1024]` are fixed-size stack arrays. The variable `k` is clamped to `vocab_size` but NOT clamped to 1024. If `k > 1024`, the loops write beyond the stack buffer, causing a stack buffer overflow.

### Fix
- Define `#define TOP_K_MAX 1024` constant.
- Add `if (k > TOP_K_MAX) k = TOP_K_MAX;` clamp after the `vocab_size` clamp.
- Add a comment explaining the limit.

### Rationale
k > 1024 is never practically useful for sampling. The simple clamp is preferred over dynamic allocation for stack safety and simplicity.

---

## Bug 2: HIGH — Output Buffer Overflow in `thought_filter.c`

**Files:** `src/reasoning/thought_filter.c`, `include/reasoning/thought_filter.h`
**Severity:** HIGH

### Problem
`thought_filter_process()` takes `char *output_buf` with no size parameter. The function appends characters via `strlen(output_buf)` with no bounds check. If a long token is processed, it can write past the buffer.

### Fix
1. Add `size_t output_buf_size` parameter to the function signature in both `.h` and `.c`.
2. Add bounds checks before every write: `if (out_len + 1 < (int)output_buf_size)`.
3. Add bounds check before tag_buffer flush: `if (out_len + f->tag_pos < (int)output_buf_size)`.
4. Update ALL callers to pass the buffer size.

### Affected Callers
- `src/reasoning/reasoning_generate.c:84`
- `tests/test_reasoning.c` (multiple call sites)

---

## Bug 3: MEDIUM — Windows HANDLE Truncation in `mapped_file.c`

**Files:** `src/memory/mapped_file.c`, `include/memory/mapped_file.h`
**Severity:** MEDIUM

### Problem
`mf->fd = (int)(intptr_t)hFile;` — Windows HANDLE is a pointer type (8 bytes on x64). The struct stores `fd` as `int` (4 bytes). Casting truncates the upper 32 bits on 64-bit Windows.

### Fix
1. Use platform-conditional field in `MappedFile` struct:
   - `void *handle` on Windows
   - `int fd` on POSIX
2. Update Windows code to use `mf->handle` directly.
3. POSIX code remains unchanged.

---

## Issue 4: LOW — Static `byte_decode_buf` Thread Safety in `tokenizer_decode.c`

**File:** `src/tokenizer/tokenizer_decode.c`, line 7
**Severity:** LOW

### Problem
`static char byte_decode_buf[2]` is a file-scope static buffer. If two threads call `tokenizer_decode()` concurrently with byte tokens, they race on this buffer.

### Fix
Make the buffer thread-local using platform-specific TLS:
- `__thread` on GCC/Clang (POSIX)
- `__declspec(thread)` on MSVC (Windows)

---

## Issue 5: INFO — SIMD Compile-Time Detection

**File:** `include/math/simd_dispatch.h`
**Severity:** INFO

### Problem
SIMD availability is determined at compile time. On machines without AVX2 but compiled with AVX2 flags, this would crash with SIGILL.

### Fix
Add a TODO comment noting that runtime CPUID detection should be added in a future version. This is an architectural enhancement, not a bug fix for this cycle.
