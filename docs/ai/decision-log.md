# Decision Log — project-zero

> Timestamped architectural / tooling / workflow / process decisions. Newest first.
> Read at session start. Last updated: 2026-06-19.

### 2026-06-19 — Prebuilt x86-64 binary via a tagged GitHub Release
- Decision: Ship a portable prebuilt `adaptive_ai_engine` as a GitHub Release asset, built by a
  new `make dist` target and published by `.github/workflows/release.yml` on `v*` tag push.
  First release tagged `v0.1.0` (pre-1.0 → `prerelease`).
- Portability decision (key): `make release`'s `-march=native` is NOT distributable (the
  AVX-512-VNNI CI runner would bake in AVX-512 → SIGILL on older CPUs). Use **per-TU
  multiversioning**: bulk at `-march=x86-64-v2`, each SIMD kernel TU carries its own ISA flag,
  and `simd_dispatch.c` is compiled at the baseline with `-DTN_FORCE_DISPATCH_ALL` so AVX2/
  AVX-512/VNNI are selected at RUNTIME (the design `simd_dispatch` already implements). Static
  `-static-libstdc++ -static-libgcc` → only libc/libm at runtime.
- Min-CPU envelope: starts + runs BitNet ternary at x86-64-v2; quant/dense GGUF (Q4_K/F16) need
  AVX2 (those kernels are compile-time AVX2-or-scalar, called directly). Documented in
  `docs/RELEASING.md`.
- Supply chain: avoid third-party Actions (cut the release with the `gh` CLI), least-privilege
  `permissions` (only the publish job gets `contents: write`); SHA-pinning noted as a follow-up.
- Status: ACCEPTED; verified gcc release/test(46)/debug/dist + clang release/debug/dist green,
  golden output (France→Paris, Germany→Berlin) correct across scalar/avx2/avx512f/vnni and
  T=1/2/8. (clang `make test` is blocked only by a missing ASan runtime in the local container.)

### 2026-06-07 — Adopt a cross-tool AI development system with one source of truth
- Decision: `docs/ai/**` is canonical; Claude Code, GitHub Copilot, and Google Antigravity
  files are thin adapters that summarize and link here. All such files are dynamic and updated
  proactively every session.
- Rationale: prevent three drifting instruction systems; preserve context across tools.
- Sync rule: on a durable change, update `docs/ai/**` first, then adapters, then record in
  `change-trace.md`. See `tool-sync-policy.md`.

### 2026-06-07 — `master` is the canonical branch; `docs/...` branch archived, not merged
- Decision: Do not merge the unrelated-history `docs/readme-footer-openbenchmarking` branch.
  Verified `master` is a strict superset of its code; archived the branch as a git bundle and
  (pending) remove it from the public remote.
- Rationale: unrelated histories add no code; the branch holds stale tests and messy history.

### 2026-06-07 — Production OOM guard in `run_state_alloc`
- Decision: Add a deterministic size guard (reject buffers >32× available RAM) instead of
  relying on `calloc` returning NULL.
- Rationale: cross-platform determinism (macOS over-commits). 32× headroom never rejects a
  runnable config; all three models re-verified unchanged.
- Status: ACCEPTED, merged via PR #6.

### 2026-06-07 — CI fixes are test/build-only except the one guard above
- Decision: Keep production engine behavior unchanged; fix CI by fixing test harness + Makefile.
- Rationale: "no new development / no regression" constraint; the failures were test-side.

### 2026-06-07 — Regression measured by A/B on the same hardware
- Decision: Because the cloud CPU differs from documented i5-11300H/Xeon baselines, prove "no
  regression" by building HEAD vs a known-good commit on the same host and comparing tok/s +
  golden outputs, rather than comparing to the documented absolute numbers.
- Rationale: absolute tok/s isn't portable across CPUs; relative A/B is.
