# Mistake Log — project-zero

> Canonical, append-at-top (newest first). Read this at the start of every session.
> Add an entry **immediately** when a mistake, false assumption, regression, or avoidable
> rework is found. Propagate durable lessons into `engineering-rules.md` and the tool adapters.

## Entry template (copy this)
```
### YYYY-MM-DD — <short title>
- Summary: <what went wrong>
- Root cause: <why>
- Affected files/modules: <paths>
- Detection: <test / CI job / ASan / review>
- Correction: <the fix>
- Prevention rule: <durable rule; also added to engineering-rules.md / adapters if durable>
```

---

### 2026-06-07 — `make debug` never linked the sanitizer runtime / lacked `-march=native`
- Summary: After tests passed, `make debug` failed for both compilers (undefined `__asan_*`,
  then undefined `ternary_matmul_packed_avx2/avx512`).
- Root cause: `debug` compiled objects with `-fsanitize` but the `$(TARGET)` link omitted it;
  and debug had no `-march=native`, so feature-gated fallback TUs compiled empty while VNNI
  dispatch TUs (built with explicit `-mavx512vnni`) referenced them.
- Affected files/modules: `Makefile` (`debug` target, `$(TARGET)` link, `CFLAGS/CXXFLAGS_DEBUG`).
- Detection: GitHub CI `make debug` step (newly reached after `make test` was fixed).
- Correction: debug `LDFLAGS += -fsanitize=address -fsanitize=undefined`; add `-march=native`
  to `CFLAGS_DEBUG`/`CXXFLAGS_DEBUG`; add `-mavx512vl` to the 256-bit VNNI rule.
- Prevention rule: a CI step only validates what it reaches; verify the **full** sequence
  (release+test+debug) for gcc **and** clang locally before declaring CI fixed.

### 2026-06-07 — Uninitialized struct field → nondeterministic ASan stack-overflow
- Summary: `test_vision_projector` crashed (read 64B past `patches`) only on the non-AVX512
  clang runner; passed elsewhere.
- Root cause: `VisionProjector proj;` left `scale_factor` uninitialized; garbage `>1` selected
  the pixel-shuffle path that over-reads.
- Affected files/modules: `tests/test_vision_components.c`, `src/multimodal/vision_projector.c`.
- Detection: GitHub CI ubuntu-22.04 clang ASan.
- Correction: `memset(&proj,0,sizeof(proj))` before use.
- Prevention rule: zero every struct before partial init (same class as the weights bug below).

### 2026-06-07 — Reliance on platform malloc behavior for OOM trapping
- Summary: `rb_mem_02_oom_resistance` (INT_MAX context) got OS `Killed:9` on macOS though it
  returned `TN_ERR_OOM` on Linux.
- Root cause: `run_state_alloc` depended on `calloc` returning NULL for absurd sizes; macOS
  over-commits then the OOM killer fires.
- Affected files/modules: `src/core/run_state.c`, `tests/test_redbox.c`.
- Detection: GitHub CI macOS job.
- Correction: deterministic guard `tn_alloc_too_large()` (overflow check + reject >32× free RAM
  via `tn_get_free_ram`) before the big allocations.
- Prevention rule: trap pathological sizes explicitly; never depend on allocator failure modes.

### 2026-06-07 — Missing `memset` before `weights_alloc_pointers` (caller contract)
- Summary: `test_blackbox` and `audit_sliding_window_crash` aborted under ASan (invalid free /
  wild-pointer memcpy).
- Root cause: tests declared `TransformerWeights` on the stack without zeroing; the documented
  contract requires the caller to `memset` first, else `weights_free_pointers` frees garbage and
  uninitialized `layers_are_ternary`/`layer_weight_type` misroute the forward pass.
- Affected files/modules: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`
  (contract in `src/core/weights.c`).
- Detection: GitHub CI ubuntu-22.04 + macOS ASan.
- Correction: add `memset(...)` (+ `#include <string.h>`) before `weights_alloc_pointers`.
- Prevention rule: honor documented zero-first caller contracts; codified in engineering-rules.md.

### 2026-06-07 — Initial parity confusion: filtered diff hid the real branch delta
- Summary: A `git diff -- src/ include/` suggested only 4 files differed between `master` and the
  unrelated `docs` branch; a full diff showed many more (tests, Makefile, docs, junk logs).
- Root cause: over-narrow diff path filter.
- Affected files/modules: branch-comparison process.
- Detection: cross-checking with `git diff --name-status` (no path filter).
- Correction: always run an unfiltered name-status diff first, then drill down.
- Prevention rule: verify branch parity with a full diff before concluding; recorded in decision-log.
