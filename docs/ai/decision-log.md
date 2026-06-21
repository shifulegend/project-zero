# Decision Log — project-zero

> Timestamped architectural / tooling / workflow / process decisions. Newest first.
> Read at session start. Last updated: 2026-06-21.

### 2026-06-21 — Claim corrected to same-SIMD/same-thread/same-precision; hero = beats bitnet.cpp
- Decision: The README headline claims only what holds apples-to-apples. Fresh three-engine
  measurement on one Xeon (same SIMD, per-thread, matched precision) shows: Project Zero
  beats Microsoft `bitnet.cpp` on BitNet at every thread (+19…+37%, BF16 head; 1.80× / 95%
  DRAM ceiling tuned), and on dense SmolLM2 beats `llama.cpp` at 1–3 threads but trails at the
  4-thread peak (−12%). So **drop any blanket "beats llama.cpp"**; hero = beats bitnet.cpp +
  only single no-dep binary running ternary AND dense. DeepSeek MoE 7× gap stays visible.
- Wording: marketing copy/PNGs must **not** use the word "honest" (reads as justifying);
  state facts confidently instead. (Internal docs may still discuss honesty.)
- Methodology: TG steady-state; PZ via `[gen]` over 128 tok; competitors via `llama-bench
  -n 128 -r 5`; warm cache; one test at a time; full results in BENCHMARK_REPORT.md Addendum AP.
- Gotcha (recorded): the container's host CPU migrated mid-session (Xeon 2.10→2.80 GHz; the
  2.80 lacks `avx_vnni`), which SIGILLs native-built competitor binaries and cripples
  bitnet.cpp's i2_s kernel (0.58 tok/s). PZ is unaffected (runtime SIMD dispatch). Comparison
  numbers are from the 2.10 GHz host where both engines had their VNNI kernels; live tty/video
  demos are PZ-only on the current host and labelled as such.
- Assets added: `docs/benchmark_bitnet.png`, `docs/benchmark_smollm2.png` (bar charts),
  `docs/tty_bitnet.png`, `docs/tty_smollm2.png` (live terminal captures), `docs/demo.webm`.

### 2026-06-20 — README repositioned as a "claim + proof" star-conversion page
- Decision: Lead the README with an honest performance claim ("beats `llama.cpp`/`bitnet.cpp`
  in some configs"), an above-the-fold benchmark table, visual proof, a one-command `make demo`,
  and exposed audit/QA links. Detailed technical content (CLI, architecture, DeepSeek, limits)
  is kept below, nothing removed.
- Honesty rule (binding): every headline win states its config; the losses stay visible —
  DeepSeek-V2 MoE ~7× behind llama.cpp (expert scatter) and SmolLM dense losing at T=1–2.
  Audience is systems devs who will scrutinize; an overclaim that doesn't survive `make demo`
  costs more credibility than it gains.
- Added: `make demo` target (downloads SmolLM2-135M GGUF, runs golden prompt; tokenizer is
  embedded so no `--tokenizer`), `docs/GROWTH_STRATEGY.md` (distribution playbook), and three
  committed proof images under `docs/` (fresh-Xeon terminal card + two OpenBenchmarking result
  screenshots, rendered via headless Chromium / Playwright).
- Reproduction note: the cloud host is itself an Intel Xeon 2.10 GHz / AVX-512 VNNI, so the
  fresh run (BitNet 40.42 tok/s, SmolLM2 142.39 tok/s, INT4) is legitimately comparable to the
  documented Xeon numbers. Project Zero reads BitNet only from native `.bin` (no ternary-GGUF
  support), so the BitNet model was converted from HF safetensors; bf16 tensors were re-encoded
  to f32 first because `convert_hf_bitnet.py` uses numpy (no bf16 slicing) — a one-off
  workaround, the committed tool was not modified.

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
