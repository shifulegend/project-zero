# Decision Log — project-zero

> Timestamped architectural / tooling / workflow / process decisions. Newest first.
> Read at session start. Last updated: 2026-06-07.

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
