## Summary
<!-- One-line description of what this PR does -->

## Type of Change
- [ ] Bug fix
- [ ] Performance optimization
- [ ] New feature / phase
- [ ] Documentation
- [ ] Refactor / cleanup

## Related Issue / Discussion
<!-- Closes #<issue> or relates to Discussion #<n> -->

## What Was Changed
<!-- Brief description of the code changes -->

## Testing Done

- [ ] `make release` succeeds
- [ ] `make test` passes (all tests green)
- [ ] Ran inference manually and output is correct:
  ```text
  # paste a sample run here
  ./pz --model ... --prompt "What is the capital of France?" --max-tokens 10
  # Output: ...
  ```
- [ ] For perf changes: benchmarked before/after

## Benchmark Results (for perf PRs)

| Config | Before | After | Delta |
|---|---|---|---|
| T=4, SIMD=auto, CLS=auto | — tok/s | — tok/s | +0% |

## Review aids
- [ ] **One concern.** This PR does one logical thing; mechanical changes (reformat, rename) are in their own PR. CI fails above 400 changed lines (generated/vendored files excluded); a maintainer can add the `size-override` label for a mechanical change.
- [ ] **Tests first.** For a behaviour change or bug fix, a test that fails without the change is included (Red -> Green -> Refactor).
- [ ] **Web UI change:** a screenshot or short recording of the new/changed screen is attached (or "no UI change").
- [ ] **Security / concurrency** (touches `src/api/`, `src/agent/`, `src/update/`, `install.sh`, threads or locks): I checked untrusted input, error paths, locking and shutdown, and `make test-tsan` passes (or "not applicable").
- [ ] **Walkthrough** (large, AI-assisted or architectural PRs): a few lines below explain how the change works and why this design, in my own words; I can answer questions about every file.

### Walkthrough (if checked above)
<!-- how data flows through the change, what invariants it relies on, what could break -->

## Documentation Updated
- [ ] Updated relevant `.md` file (README / BENCHMARK_REPORT / DEBUGGING_JOURNAL / IMPLEMENTATION_PLAN)
- [ ] No docs change needed

## Checklist
- [ ] No hardcoded paths or magic numbers
- [ ] No regressions on BitNet b1.58 (existing working model)
- [ ] Scalar fallback present for any new SIMD code
- [ ] Memory allocated with `tn_aligned_calloc()` for SIMD buffers
