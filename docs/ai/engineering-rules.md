# Engineering Rules — project-zero

> Canonical source of truth. Adapters reference this. Last updated: 2026-07-16.

## Modularity
- Smallest sensible units: small functions, single responsibility, explicit interfaces.
- One concern per translation unit; mirror `src/<area>/x.c` with `include/<area>/x.h`.
- Prefer composition and isolated kernels over monoliths. New kernels are separate TUs and
  get a per-file SIMD rule in the `Makefile` when they need feature flags.
- Reuse existing patterns before adding abstractions (see `simd_dispatch`, GGUF reader,
  `tn_size_mul_overflow`/`tn_size_mul4`, `tn_aligned_calloc`, `tn_get_free_ram`).

## No hardcoding / configurability is the default
- Model shape, vocab, special-token IDs, quant types, RoPE/MoE params come from **GGUF
  metadata** (`hdr->arch` prefix), never literals. (Past garbage-output bugs were hardcoded keys.)
- Behavior is selectable via CLI flags / env / calibration: `--threads --simd --classifier
  --temperature --top-p --seed`. Add config, don't branch on magic numbers.
- Any unavoidable constant must carry a timestamped comment explaining why (see code-comment rule).

## Memory & safety (hard rules, learned from real bugs)
- **Zero structs before partial population.** Always `memset(&x, 0, sizeof(x))` before
  `weights_alloc_pointers()` / `run_state_alloc()`-style fillers. `weights_free_pointers()`
  frees every pointer field; un-zeroed fields → invalid free / wild-pointer reads.
- **No reliance on platform malloc behavior.** Absurd sizes must be trapped deterministically
  (overflow-checked multiply + RAM-relative guard via `tn_get_free_ram`), not via "calloc
  returns NULL" (false on macOS overcommit).
- Use the overflow-checked size helpers for every multi-factor allocation.
- All SIMD buffers are 64-byte aligned (`TN_SIMD_ALIGN`, `tn_aligned_*`).

## Code organization
- C99 for engine; C++17 only where already used (chat template). Match existing style
  (`-Wall -Wextra -Wpedantic` clean; no new warnings).
- Headers declare; `.c` defines. Feature-gate SIMD code with `TN_HAS_AVX2/AVX512/...` macros.
- Cross-platform: x86 (AVX2..VNNI) + ARM (NEON/dotprod) + scalar fallback must all compile.
  Portable prefetch via `TN_PREFETCH_T1`; never raw `_mm_prefetch` outside x86 guards.

## Verification (definition of done)
A change is done only when:
1. `make release CC=gcc && make test CC=gcc && make debug CC=gcc` pass, **and** the same for
   `CC=clang` (clang ASan is stricter — e.g. 256-bit VNNI needs `-mavx512vl`).
2. Relevant golden-output check passes (e.g. "capital of France" → contains "Paris").
3. No perf regression for touched kernels (A/B build vs a known-good commit on the same host;
   compare tok/s — see `docs/REGRESSION_VERIFICATION_2026-06-07.md` for method).
4. Docs synchronized (this file set + the right tool adapters) and a commit checkpoint proposed.
- Never claim completion without stating what was verified and what remains UNVERIFIED.
- `make test` runs **every** `tests/*.c` and aborts on the first failure — a green full run is
  the bar. CI historically died early, masking downstream failures; expect a cascade when you
  unblock the first failing test, and keep going until the whole suite passes on all platforms.

## Bug-fix policy (fix, don't just log)
- **Any bug found gets fixed, in the same pass, regardless of whether it's pre-existing or
  unrelated to the task at hand.** "Pre-existing" is not a reason to leave a real defect in
  place — it's a reason to fix it now while it's been found, since it may not be found again
  soon. This applies to anything surfaced by ASan/UBSan/ThreadSanitizer, screenshot/manual
  review, test failures, or incidental code reading — not just bugs in the diff you set out
  to write. (Example: 2026-07-16, a ~1.2MB ASan leak was traced not to the feature being worked
  on but to `tokenizer_free()` being skipped for the common GGUF-auto-load tokenizer path —
  fixed immediately rather than dismissed as "unrelated to this change".)
- Document the fix in `docs/ai/mistakes.md` (root cause + correction) same as any other bug —
  fixing it doesn't remove the obligation to record it.
- The only valid reason to *not* fix a found bug immediately is genuine scope: a fix requiring a
  large, architecturally-significant change (e.g. adding a whole new subsystem, a signal-handling
  rework, a breaking API change) should be flagged explicitly to the user with a clear description
  of the gap and a recommendation, rather than either silently fixed as scope creep or silently
  left unfixed. Small, mechanical, or clearly-scoped fixes (a missing `free()`, a wrong guard
  condition, an off-by-one) never need to wait for permission.
- Re-run the full verification cycle (`## Verification` above) after any bug fix, same as for
  planned work — a fix that isn't verified is just a different unverified change.

## Version control & branch hygiene
- Integrate into `master` via small PRs (don't push directly); keep each PR's branch short-lived.
- **Delete a feature/fix branch as soon as its PR is merged** — keep the remote to `master` plus
  only actively-developed branches. No stale or duplicate branches.
- Enable repo Setting "Automatically delete head branches" so merged PR branches are pruned
  automatically (prevents accumulation; no manual cleanup needed).
- Don't create placeholder/flag branches (e.g. `DELETE-ME-*`) as a habit — they are a last-resort
  workaround only when an environment blocks ref deletion; delete them too once unblocked.
- Don't commit build artifacts, downloaded models, or transient logs (`wget-log`, `*stdout.log`).

## Security / safety constraints
- Public repo: never commit secrets, keys, tokens, `.env`, `*.pem`, real credentials. (History
  is currently clean — keep it so.) Doc placeholders like `<YOUR_SUDO_PASSWORD>` are fine.
- Keep ASan/UBSan green; treat sanitizer aborts as real bugs, not noise.
- Dual-use/security tooling stays within defensive/testing scope.

## Performance
- Hot path is memory-bandwidth bound; prefer fused kernels and avoiding redundant
  quantize/dequant passes. Measure before/after with the `tools/*bench*.sh` harness.
