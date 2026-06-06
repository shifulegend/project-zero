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
  ```
  # paste a sample run here
  ./adaptive_ai_engine --model ... --prompt "What is the capital of France?" --max-tokens 10
  # Output: ...
  ```
- [ ] For perf changes: benchmarked before/after

## Benchmark Results (for perf PRs)

| Config | Before | After | Delta |
|---|---|---|---|
| T=4, SIMD=auto, CLS=auto | — tok/s | — tok/s | +0% |

## Documentation Updated
- [ ] Updated relevant `.md` file (README / BENCHMARK_REPORT / DEBUGGING_JOURNAL / IMPLEMENTATION_PLAN)
- [ ] No docs change needed

## Checklist
- [ ] No hardcoded paths or magic numbers
- [ ] No regressions on BitNet b1.58 (existing working model)
- [ ] Scalar fallback present for any new SIMD code
- [ ] Memory allocated with `tn_aligned_calloc()` for SIMD buffers
