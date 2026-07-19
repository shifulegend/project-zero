# Mistake Log — project-zero

> Canonical, append-at-top (newest first). Read this at the start of every session.
> Add an entry **immediately** when a mistake, false assumption, regression, or avoidable
> rework is found. Propagate durable lessons into `engineering-rules.md` and the tool adapters.
> Last updated: 2026-07-19.

### 2026-07-19 — CI red since 07-17: test_q2_0_matmul's reference compared the wrong kernel's math
- Summary: `test_single_block_multi_row` failed 3/5 rows on every CI run since the 07-17
  Q2_0/VNNI commits (`f9857a2`/`ce8e90d`/`dd295e5`), on ubuntu-22.04, ubuntu-latest, and
  macOS alike — never flaky, always the same 3 rows.
- Root cause: `ref_dot_q2_0_row()` in `tests/test_q2_0_matmul.c` was made to compare against
  `quantize_row_to_i8`-quantized activations, matching what `matmul_q2_0_vnni.c`'s VNNI
  kernel actually computes with (int8 dpbusds is a hardware requirement there). But
  `parallel_matmul_q2_0()` only dispatches to that VNNI kernel `#if TN_HAS_AVX512VNNI` — on
  every CI runner (none of which have AVX-512 VNNI, confirmed: this reproduced locally on a
  2-core host with `avx512bw` but no `avx512vnni`), it falls back to `matmul_q2_0.c`'s
  portable kernel, which FMAs the raw float32 activations directly and never quantizes them
  at all. The test was comparing two genuinely different computations and failing on
  whichever rows this test's synthetic data happened to round unfavorably for.
- Fix: `ref_dot_q2_0_row()` now branches on the same `TN_HAS_AVX512VNNI` macro the kernel
  dispatch itself uses — quantized reference when VNNI will actually run, plain float
  reference otherwise. 35/35 tests pass locally (portable path); full `make test` suite
  passes clean.
- Prevention rule: when a kernel has a compile-time-gated fast path, the test's reference
  implementation must branch on the *same* gate, not assume one path unconditionally — a
  reference tuned for the fast path silently miscompares the fallback path, and CI runners
  are exactly where the fallback path is what actually gets exercised.
- Not fixed here (separate, unrelated, tracked in CMO ops board): `src/math/matmul_q2_0_vnni.c`
  itself was never exercised by this bug or its fix (no VNNI hardware available to test
  against) — worth a dedicated VNNI-host correctness pass before the next release.

### 2026-07-17 — Ceiling push: two optimizations implemented, measured, and REVERTED on evidence; plus a misleading KV-strategy line that had already corrupted a published bound
- Summary: pushing from 59% toward the 6.0 tok/s ceiling, step-timing was first wired into the
  qwen35 hybrid path (steps 4-12 + two new DeltaNet steps) per the "pinpoint before
  optimizing" rule. Attribution: ~91% of per-token time in Q2_0 matmuls, DeltaNet scalar
  recurrence 6.4%, everything else <3%. Two optimizations were then tried and **both failed
  their A/B honestly**:
  1. A3a wide non-VBMI unpack (512-bit SSE-algorithm-across-lanes + 4x4 lane transpose, ~3x
     fewer unpack ops/code): bit-exact, but neutral-to-negative over 3 alternating on/off
     rounds — the A1+A2 kernel is already load-bound at ~30-35 GB/s, so unpack compute was no
     longer the bottleneck. The optimization targeted a regime (compute-bound) that A1+A2 had
     already exited. Reverted.
  2. Sub-64-row inline dispatch (skip the thread pool for DeltaNet's 48-row beta/alpha
     projections, 96 dispatches/token): slightly SLOWER in both interleaved end-to-end rounds
     (3.24/3.31 vs 3.30/3.45 tok/s) — the pool's dispatch is cheap enough that 48-row jobs
     still profit from splitting; the "pool wake+join dominates tiny gemvs" assumption was
     never measured before being coded. Reverted, warning comment left at the site.
- Also fixed: the startup line printed "KV Strategy: Quantized I8" for qwen35 models whose
  K/V caches are actually raw F32 (`q35_key_cache` is `float **`; the quantized-KV machinery
  is not wired into that path). This had already propagated: CEILING_CALCULATION.md §3's KV
  bound was computed from the printed "I8" (32 KB/pos) instead of the real F32 (128 KB/pos)
  — corrected, ~8% of the weight stream at 4K ctx, ~40%+ near max context.
- Detection: both reverts came from the measurement discipline itself (alternating-round
  micro A/B; interleaved same-session end-to-end A/B), not from review; the KV line from the
  hot-path exploration for this push.
- Prevention rules: (1) an optimization designed against a bottleneck model must re-verify
  that model still holds *after* the previous optimization landed — A1+A2 moved the kernel
  from compute-bound to load-bound, silently invalidating A3a's premise; (2) "dispatch
  overhead dominates small jobs" is a measurement, not an axiom — pool implementations vary
  by orders of magnitude in wake cost; (3) a printed runtime-strategy string is a claim like
  any other and needs verification before being used as an input to published arithmetic —
  this file now contains two same-day entries (probe "measured" GB/s, KV "I8") where trusting
  the engine's own output corrupted downstream analysis.

### 2026-07-17 — Independent fresh-context review of the ceiling calculation: 3 new real defects found and fixed; core arithmetic independently confirmed
- Summary: per user request, a clean-slate reviewer (no access to this session's derivations,
  instructed to re-derive everything from source before reading the docs) audited the entire
  ceiling calculation. It independently reproduced the per-token byte accounting to the byte
  (6,827,406,400 B for Ternary-Bonsai-27B default), independently found BOTH pre-fix DRAM-probe
  defects before being told of them, and surfaced defects this session had missed:
  (1) `calibration.c` converted GB/s→MiB/s with ×1024 instead of ×(1e9/2^20)≈953.67 — every
  printed classifier tok/s estimate ~7.4% high (rankings unaffected, common factor); (2) the
  native (non-GGUF) branch never called `tn_hardware_profile_set_model_bytes()`, so any native
  model that isn't exactly BitNet-2B kept the hardcoded 1149 MiB estimate — the same bug class
  just fixed for GGUF, still live one branch over; (3) the fixed probe's DCE-defeat sink was
  `+=`'d from all worker threads unsynchronized (benign result-wise, real data race). Also: a
  main.c comment contradicting its own code (generic-GGUF embedding accounting), `bench_q2_0`
  printing but not enforcing its correctness cross-check, a mixed-unit "1179 MB" arithmetic slip
  in the doc, and the L3-heuristic comment calling a ceiling-raising cache credit a "LOWER
  BOUND". All fixed same pass; full table in `docs/architecture/CEILING_CALCULATION.md` §8.
- Detection: the review itself — specifically its independence (fresh context, derive-first
  ordering), which is what made the ×1024 unit error and the missed native branch visible after
  two authors (the original and this session) had each read past them.
- Verification after fixes: gcc release 0 warnings, test suite 46/46, `make bench-q2` (hard
  correctness gate now active) 1.45-1.57x, `make demo` golden ("Paris", SmolLM2 258 MB →
  165.8 tok/s ceiling at 44.9 GB/s), 27B model run 3.81 tok/s (30 tok) at 40.3 GB/s measured /
  5.9 ceiling; clang standalone compile-checks of all changed TUs clean. Native-branch
  set_model_bytes is compile+logic-verified only (no native .bin model in this container) —
  flagged, not hidden.
- Prevention rule: for any numeric pipeline that survived one author and one later auditor,
  assume unit-conversion factors and branch-coverage symmetry ("was the same fix applied to
  every parallel branch?") are still unchecked until someone re-derives the numbers without
  seeing the originals. A reviewer who must produce the figures independently before comparing
  catches classes of error that diff-reading reviewers structurally cannot.

### 2026-07-17 — DRAM bandwidth probe under-measured ~3.4x (accounting bug + serialized reads); every historical "measured GB/s" and ceiling-utilization claim was wrong
- Summary: while implementing the planned probe improvement (serialized volatile reads →
  multi-accumulator streaming), reading the outer timing loop revealed a second, larger bug:
  `bw_thread_fn` ran its **3 read passes inside one timed create/join round**, while the
  caller's aggregate divided **one** pass worth of bytes by the wall time of all three — a
  systematic ~3x under-measurement. The per-thread bytes/elapsed figures (correctly accounted)
  were computed and then discarded in favor of the broken aggregate.
- Fix: threads now run exactly one pass per round (the caller's existing 3-round best-of loop
  provides repetition), and each pass reads every byte via 8 independent 64-bit accumulator
  chains instead of one volatile char per 64-byte line (volatile scalar chains can't keep line
  fills in flight). Same host, same session: **12.0 → 41.2 GB/s**. Cross-checks that had
  already exposed the discrepancy: memcpy ~22 GB/s effective (≈44 GB/s bus traffic), the Q2_0
  kernel itself streaming 29 GB/s from a >L3 buffer, and — in hindsight the loudest tell —
  SmolLM2 measuring 56.5 tok/s against a "44.1 tok/s at 100% BW" ceiling, a physical
  impossibility that the in-code comment had rationalized as "prefetch effectiveness" and this
  session initially waved through as "probe bias" without quantifying it.
- Consequences: every "DRAM bandwidth (measured)" ever printed (16.0, 14.0-17.3, 12.2, 10.9
  GB/s...) is ~3x low; every ceiling and utilization percentage derived from one is wrong,
  including the BitNet-era "95% of ceiling" claims (true utilization was ~3x lower) and this
  same day's RCA-addendum claim that host A's 2.74 tok/s was "at the bandwidth wall" (it was at
  ~40% of true bandwidth). Corrected in `docs/architecture/CEILING_CALCULATION.md` (spec +
  audit), the RCA addendum, and the README. A/B comparisons and relative claims are unaffected.
- Related result recorded here for cross-reference: with the honest ceiling in place, the Q2_0
  VNNI row-dot restructure (per-row float vector accumulator replacing a per-128-element-block
  `_mm512_reduce_add_epi32`, F16C scale decode; frozen `_ref` variant kept for in-binary A/B via
  new `tools/bench_q2_0`) measured 1.32-1.68x per shape and **2.74/2.80 → 3.56/3.54 tok/s
  (+28%) end-to-end** on the real 27B model, interleaved same-session, token-identical greedy
  output, `TN_STEP_TIMING=1` attribution (Dense FFN 17.7→13.1 s, LM head 1.19→0.83 s per 61
  tokens). Current standing: 3.56 measured vs 6.0 ceiling = 59% utilization.
- Affected files: `src/core/hardware_profile.c` (probe), `src/math/matmul_q2_0_vnni.c` +
  `include/math/matmul_q2_0.h` + `Makefile`/`CMakeLists.txt` (`-mf16c`) + `tools/bench_q2_0.c`
  (kernel + bench), docs as above.
- Detection: code reading during a planned adjacent improvement — not by any test; nothing
  asserts on probe output. The "engine exceeds its own 100%-of-bandwidth ceiling" observation
  had been visible in outputs for weeks.
- Prevention rule: when a benchmark's own reported number is *physically impossible* relative
  to another trusted measurement (throughput above "100% of bandwidth"), stop and reconcile the
  two before publishing either — "the probe is conservative" is a hypothesis to quantify, not a
  caveat to file away. For any timed loop, assert the bytes-counted and the passes-timed come
  from the same code path; best-of-N logic belongs in exactly one layer.

### 2026-07-17 — Hardware profiler's Data/token + tok/s ceiling were hardcoded BitNet-2B constants, wrong for every other model (~6x-overstated ceiling for Ternary-Bonsai-27B)
- Summary: while pinpointing what it would take to "reach the tok/s ceiling" on the current host,
  the ceiling itself failed an arithmetic sanity check: a dense 27B model at 2.125 bits/weight
  must stream ~6.8 GB of weights per token (7.16 GB file minus the embedding, which is read one
  row per token), yet the profiler printed "Data/token: 1149 MB" and "Ceiling: 10.4–17.1 tok/s".
- Root cause: `hardware_profile.c` computes `weight_bytes_per_tok` from compile-time
  `MODEL_TERNARY_BYTES`/`MODEL_VOCAB`/`MODEL_DIM` constants hardcoded for BitNet-b1.58-2B
  (522 MB + a 128256×2560 BF16 classifier ≈ 1149 MB), because `tn_hardware_profile_init()` runs
  before the model file is opened. Nothing ever corrected it post-load — a direct violation of
  the project's own "model shape from GGUF metadata, never hardcoded" rule that survived because
  the constants were right for the one model the profiler was written against. It also *under*-
  stated small models (SmolLM2-135M is ~258 MB/token, not 1149).
- Consequences for published analysis: every "X% of ceiling" claim for Ternary-Bonsai-27B used a
  ~6x-inflated denominator. Corrected: the real bandwidth ceiling at host A's measured 16 GB/s is
  ~2.3 tok/s — the recorded 2.74 tok/s means the engine was already at/above the *calibrated*
  bandwidth (~18.6 GB/s effective; the calibration's streaming probe under-measures real
  achievable bandwidth — same-day cross-check: this session's host calibrated 12.0 GB/s while a
  simple 512 MB memcpy loop measured ~22 GB/s effective). Host B runs (~1.0–1.24 tok/s ≈ 7–8.4
  GB/s) sit at ~60–70% of the calibrated ceiling, so the realistic kernel-side headroom on that
  host class is ~1.5–2x, not the 6–16x the old ceiling implied. The "29x Q2_0 kernel speedup" and
  all A/B comparisons are unaffected (relative measurements).
- Correction: new `tn_hardware_profile_set_model_bytes()` recomputes `weight_bytes_per_tok`,
  `theoretical_ceiling`, `model_fits_l3`, and the summary from the real loaded model; `main.c`
  calls it after GGUF weight load (file size minus embedding/head adjustments for Q2_0-native
  models, honoring a materialized classifier; MoE still overcounts — TODO). The startup box now
  labels its figure "(pre-load est.)" and a `[profile] Data/token (loaded model): ... -> ceiling
  ...` line prints post-load. Native BitNet path keeps the (correct-for-it) constants.
- Affected files: `src/core/hardware_profile.c` (setter + `rebuild_summary()` extraction +
  relabel), `include/core/hardware_profile.h`, `src/cli/main.c`.
- Detection: order-of-magnitude sanity check of the ceiling against model size during an RCA
  follow-up — not by any test; nothing asserts on profiler output.
- Verification: gcc release + full test suite green (46/46); clang release/debug build green
  (clang `make test` remains blocked by this container's missing clang ASan runtime — known
  environment gap, decision-log 2026-06-19). End-to-end: `make demo` (SmolLM2-135M GGUF) prints
  the corrected `258 MB -> ceiling 44.1 tok/s at 12.0 GB/s` line with golden output ("The capital
  of France is Paris.") intact. No test links `hardware_profile.c`'s new path, so the
  sanitizer-instrumented suite adds no additional coverage of this change; the real-binary demo
  run is the meaningful verification here.
- Prevention rule: any printed *derived* performance number (ceiling, efficiency-vs-ceiling,
  data-per-token) must be dimensional-analysis-checked against the actual artifact it describes
  (model bytes × tokens × bandwidth) before being used as an optimization target or published —
  a wrong denominator silently corrupts every percentage built on it. Startup-order constraints
  ("we don't know the model yet") don't justify leaving the estimate uncorrected once the real
  data exists; add a post-load update instead.

### 2026-07-17 — The row-count fix for the banner-scrolling bug introduced a new bug: huge blank space below short output
- Summary: user asked why the freshly-fixed screenshots (banner now visible) had so much blank
  space below the actual content. Root cause: the fix below widened the xterm.js terminal to a
  fixed `rows: 170` so Ternary-Bonsai-27B's long startup output wouldn't scroll the banner out of
  view — but most captures use far fewer than 170 rows, and `page.screenshot({ fullPage: true })`
  was screenshotting the *entire* fixed 170-row terminal regardless of how much of it was actually
  written to, padding every image with a wall of empty rows.
- Investigation: wrote a small debug script to inspect DOM heights before/after calling
  `term.resize()` down to the last used row. Confirmed the terminal element itself (`#term`) and
  `document.body.scrollHeight` both correctly shrink after resize, but
  `document.documentElement.scrollHeight` — what Playwright's `fullPage: true` actually measures —
  stays clamped to the 1200px viewport height, because browsers don't let the document's scroll
  height report below the viewport height when content is shorter than the viewport. So the resize
  worked, but the screenshot mechanism couldn't see it.
- Correction: `capture.mjs` now (1) resizes the xterm.js terminal down to `lastUsedRow + 2` after
  the raw session bytes are written and rendered, using the buffer API to find the last non-blank
  row, and (2) screenshots the `body` element directly (`page.locator('body').screenshot()`)
  instead of `page.screenshot({ fullPage: true })` — an element screenshot uses the element's own
  bounding box, which isn't clamped to the viewport the way the document's scrollHeight is.
  Verified with a synthetic 3-line test (900x1200 padded image → 900x186 correctly-sized image)
  before regenerating all 9 affected screenshots (6 current-HEAD + 3 commit-bisection).
- Consequence for the numbers displayed: since regenerating required rerunning the real model
  again, and this host's performance keeps drifting (documented at length below), the embedded
  tok/s in these screenshots changed *again* on this pass (e.g. classifier auto/bf16/int8/int4:
  0.59/1.02/0.70/1.04) — and this time even the previously-consistent ordering (BF16 slowest,
  INT4 fastest, reproduced across two earlier sweeps) broke, with INT8 landing below BF16. Recorded
  as a data point, not alarming: it reinforces that only "the four formats now genuinely differ in
  code path" is a safe claim on this host, not any specific ranking or magnitude between them.
- Affected files: `tools/screenshots/cli/capture.mjs`, all screenshots under
  `benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/` that used `PZ_CAPTURE_ROWS`.
- Detection: user visually inspected the actual shipped screenshots and asked a direct "why does
  X look wrong" question rather than trusting the prior fix was complete.
- Prevention rule: when sizing a fixed-dimension capture/render buffer generously to avoid clipping
  worst-case content, remember the buffer is now oversized for the common case — either trim to
  actual content before the capture step, or make the capture mechanism content-aware, rather than
  screenshotting the full fixed buffer. A fix for "content gets cut off" and a fix for "there's
  wasted empty space" pull in opposite directions on the same size parameter; sizing generously
  and calling it done leaves the second problem for the same person to be asked about later.

### 2026-07-17 — Banner was printing correctly all along; the screenshot capture tool was scrolling it off before the shot
- Summary: user pointed out the ASCII banner still wasn't visible in the benchmark screenshots
  despite the earlier fix making `main.c` print it unconditionally in a TTY. Checked file mtimes
  first: all 8 screenshots in `benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/`
  are from 20:34, the banner fix (`b953281`) landed at 21:23 — the screenshots simply predate the
  fix and were never regenerated. But regenerating one (`classifier_bf16.png`) with the current
  binary still didn't show the banner — a second, independent bug.
- Root cause: `tools/screenshots/cli/capture.mjs` renders the captured session through an xterm.js
  terminal fixed at `rows: 70` (`capture.html`). Ternary-Bonsai-27B's startup output (calibration
  box + hardware profile + model config dump) is well over 100 lines before generation even starts,
  so by the time the screenshot is taken the banner (printed first) has scrolled out of the
  terminal's own view — same failure mode as a real terminal scrolling. `page.screenshot()` was
  also missing `fullPage: true`, so even a taller terminal would've been clipped to the fixed
  900x1200 Playwright viewport.
- Correction: `capture.html` now reads an optional `?rows=` query param (default stays 70,
  unchanged for existing short-output captures like the BitNet demos); `capture.mjs` accepts
  `PZ_CAPTURE_ROWS` and passes it through, and its final screenshot call now sets
  `fullPage: true`. Regenerated all 6 PZ screenshots (classifier auto/bf16/int8/int4, thread t=1/t4)
  with `PZ_CAPTURE_ROWS=170` — banner confirmed visible in all of them.
- Second finding while regenerating: the 3 classifier screenshots also still had the *old buggy
  no-op numbers* (1.07/1.09/1.07) baked in from before the real classifier fix — already wrong
  independent of the banner issue, since the README table had already been corrected. Would have
  been a real, visible inconsistency between the README table and its own linked screenshots.
- Third finding, incidental but important: the fresh classifier sweep taken during this
  regeneration (auto=1.27, bf16=1.21, int8=1.30, int4=1.37) reproduced the same ordering
  (BF16 slowest, INT4 fastest) as the original post-fix sweep, but with a much smaller spread than
  the original (1.13/1.19/2.60/2.62). Two independent same-direction results across sessions is
  good evidence the fix's *effect direction* is real; the earlier note about magnitude being
  host-dependent (see the memory-subsystem entry below) is reinforced, not contradicted — a ~2.2x
  gap and a ~13% gap are both "INT4 faster than BF16," just by very different amounts depending on
  host state at measurement time.
- Affected files: `tools/screenshots/cli/capture.mjs`, `tools/screenshots/cli/capture.html`,
  6 regenerated PNGs, `README.md` (numbers + explanatory notes).
- Detection: user directly re-checked the actual deliverable (the screenshots) rather than trusting
  my claim that the banner fix was complete — the fix to the printing logic was real and verified,
  but "verified in isolation" (a pty capture) missed that the actual published artifacts used a
  different capture path with its own independent bug.
- Prevention rule: when a fix is meant to change a *visible artifact* (a screenshot, a rendered
  page), verify the actual artifact that ships, not just the underlying mechanism in isolation —
  a pty test proved the banner prints; it didn't prove the screenshot pipeline shows it. Two
  correct components can still compose into a wrong result if a fixed-size capture window is
  smaller than the real output.

### 2026-07-17 — Commit-bisection proof: the 2.74 → ~1.2 tok/s gap is host state, not code, confirmed by literally rerunning the old commit
- Summary: user pushed back on the memory-subsystem explanation below with a direct, falsifiable
  test request: check out the exact commit that produced the 2.74 tok/s screenshot (and the one
  before it), rebuild, and rerun the identical command — rather than accept the `git diff`-based
  argument alone. Did exactly that, in isolated `git worktree`s so the current branch stayed
  untouched.
- Method: the `pz_t4_peak.png` screenshot's banner read `ce8e90d-dirty`, so `ce8e90d` ("Add
  AVX-512 VNNI Q2_0 kernel (~29x)...") is the commit whose build produced 2.74 tok/s. Checked out
  `ce8e90d` and its immediate parent `34d3ac9` (pre-VNNI-kernel, the commit before the fast Q2_0
  kernel existed at all) into separate worktrees, symlinked the model file in (no need to copy
  7.16 GB twice), built each with `make clean && make release CC=gcc`, and ran the identical
  benchmark command against each — capturing real screenshots via the (now-fixed)
  `tools/screenshots/cli/capture.mjs`, pointed at each worktree's binary from the main repo (no
  need for a separate `node_modules`/Playwright install per worktree).
- Results (both real screenshots, not just log text; regenerated once more the same day to fix a
  blank-space capture bug — see the entry above — so the filenames/numbers below are the final,
  retrimmed versions, not the original 1.40/0.12/1.08 first measured):
  `benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/commit_bisect_ce8e90d_1.02toks.png`
  — commit `ce8e90d` (the exact commit behind the 2.74 tok/s number), rebuilt and rerun: **1.02
  tok/s** (first attempt, before the retrim fix, measured 1.40 tok/s — a different number, same
  conclusion). Not 2.74 either way.
  `benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/commit_bisect_34d3ac9_0.08toks.png`
  — commit `34d3ac9` (one commit earlier, before the VNNI Q2_0 kernel existed): **0.08 tok/s**
  (first attempt: 0.12), still matching the historically-documented pre-VNNI baseline order of
  magnitude and still ~13x slower than `ce8e90d`'s same-run number.
  `benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/commit_bisect_HEAD_0.95toks.png`
  — third leg, current HEAD, same command, run immediately after the other two: **0.95 tok/s**
  (first attempt: 1.08). Confirms the same conclusion twice over with two different absolute
  number sets: old commit and new commit are indistinguishable from each other on this host,
  whichever day or hour you happen to measure them, while the pre-VNNI control reliably shows a
  large, real gap both times.
- Why this is conclusive, not just more of the same evidence: the pre-VNNI commit's result is a
  built-in control. If this rebuild-and-rerun methodology were incapable of detecting a real
  performance difference between commits, `34d3ac9` would have measured close to `ce8e90d`'s
  number instead of ~12x slower. It didn't — it reproduced the expected, historically-documented
  gap exactly. That proves the test setup *does* detect genuine code-driven differences when they
  exist. Between `ce8e90d`, HEAD, and the earlier same-day sweeps, no such gap exists (1.08-1.40
  across three separate binaries/builds run over several hours) — the absence of a gap there is
  therefore meaningful, not a failure to look hard enough.
- Affected files: none (still not a code bug) — two new screenshots added as durable evidence
  alongside the explanation below, since a "trust me, I diffed it" argument is weaker than
  literally rerunning the old commit and screenshotting the result.
- Detection: user's direct, specific request for a falsifiable test ("go back to the commit... and
  run using the same command") rather than accepting the diff-based reasoning alone.
- Prevention rule: when a `git diff`-based "this code path is unchanged" argument is available but
  contestable, and the actual old binary can still be built and run, do that instead of resting on
  the diff — a diff proves the code is the same, not that the *old binary, run today* produces the
  same result. Only an actual rerun of the old commit can rule out an environment confound that a
  diff alone cannot address. Screenshot real proof, don't just log-cite it, when the artifact
  format elsewhere in the same body of work is screenshots.

### 2026-07-17 — This host's memory subsystem (not just DRAM bandwidth) is unstable across sessions, and it fully explains the 2.74 → ~1.2 tok/s drop
- Summary: user asked why `auto`'s post-fix 1.19 tok/s was so far below the README's earlier
  2.74 tok/s figure for the identical default configuration (Ternary-Bonsai-27B, 4 threads, no
  `--classifier`). First checked whether my classifier fix regressed the default path — it
  doesn't: `git diff` from the commit behind the 2.74 number to HEAD shows `forward.c`'s only
  change is a branch that's skipped entirely when `classifier_explicit` is false, falling through
  to the identical `parallel_matmul_q2_0(...)` call as before. Code ruled out, so re-ran the exact
  original command (`--max-tokens 60 --threads 4`, no `--classifier`, same file) to get a live,
  apples-to-apples data point instead of trusting the earlier "hardware is noisy" framing again.
- What actually happened: the re-run hung 5+ minutes at "Opening model file" (`mapped_file_open`'s
  `mmap(..., MAP_POPULATE, ...)` on the 7.16 GB file) — a step that historically completes in
  seconds. `ps` showed RSS climbing toward the file size the whole time (not deadlocked, just very
  slow to fault pages in). A direct `dd ... iflag=direct` read moments later measured 1.8 GB/s —
  fine — ruling out "the disk itself is just slow." A second re-run got past file-open but then
  stalled 400+ seconds at "Preparing runtime" (a plain `calloc` for the KV-cache buffers) with RSS
  flat and CPU only partially used. `/proc/vmstat`'s `compact_stall`/`compact_success` were both 0
  and `/proc/buddyinfo` showed plenty of large free blocks, ruling out THP-compaction/fragmentation
  as the mechanism.
- Root cause (inferred from the pattern, not directly observable from inside this guest): both
  stalls are specifically **first-touch of a large fresh memory region** — mmap-populate of the
  model file, then calloc-zeroing of the KV cache — never steady-state compute or a small read of
  already-resident memory. That's the signature of the *underlying host* (not this guest) being
  memory-pressured: the hypervisor slow-paths new physical page allocation for a contended
  co-located host, which this guest's own `free -h`/`/proc/vmstat` can't see (they report this
  guest's own memory as healthy throughout). This extends the already-documented VM-recycling
  finding (`/proc/uptime` resetting mid-session) from "DRAM bandwidth/CPU vary" to "first-touch
  memory allocation of any kind can stall arbitrarily" — a strictly worse and more general
  instability than previously characterized.
- The eventual completed re-run measured **1.24 tok/s (61 tokens)**, DRAM bandwidth 12.5 GB/s,
  calibrated ceiling 10.7 tok/s — consistent with the classifier sweep's `auto` = 1.19 tok/s from
  the same session, and nowhere near the original 2.74 tok/s / 16.0 GB/s / ceiling 17.1 tok/s
  screenshot from 2026-07-16. Achieved-vs-ceiling efficiency is similar both times (~12-16%), so
  most (not all) of the gap is explained by the ceiling itself dropping — the calibrated DRAM
  bandwidth reading is itself a symptom of the same host instability, not an independent variable.
- Affected files: none — this is not a code bug, no fix applies. Documented so this number isn't
  mistaken for a regression in future sessions.
- Detection: user directly questioned an inconsistency between two numbers from different points in
  the session rather than accepting them both at face value.
- Prevention rule: on this specific virtualized host, absolute tok/s is not comparable across
  separate sessions/benchmark runs, even for the identical command against the identical file —
  only back-to-back, same-session comparisons (like the classifier sweep's auto/bf16/int8/int4
  numbers, all measured within minutes of each other) should be treated as relatively trustworthy.
  When a user questions a cross-session number discrepancy, don't default to "explained by prior
  noise documentation" — rule out the code with a real diff, then get a live, current data point
  before concluding it's environmental again.

### 2026-07-16 — `--classifier` silently no-op'd on Q2_0-native models: root-caused, then actually fixed (not just warned about)
- Summary: a classifier-precision benchmark showed BF16, INT8, and INT4 all measuring ~1.07-1.09
  tok/s on Ternary-Bonsai-27B — identical within noise. My first explanation blamed hardware
  measurement noise alone; user pushback ("in the same hardware still does not make sense for bf16
  and int4 to be together") correctly rejected that. Root cause: `forward.c`'s classifier dispatch
  had two branches — `w->q35_is_q2_0_model` (true for this model) took an **unconditional**
  `parallel_matmul_q2_0(...)` path regardless of `--classifier`; only the non-Q2_0 `else` branch
  read `hp->classifier_fmt`. My first fix attempt only added a warning explaining the no-op rather
  than implementing real per-format support, reasoning that materializing a separate BF16/INT8/INT4
  copy of a 248320×5120 vocab was too large an architectural change to take on unprompted. User
  explicitly rejected that as insufficient ("Fix the no-op... No shortcuts, no honest explanations,
  no fooling around") — correctly: a warning documents a broken feature, it doesn't fix one.
- Real fix: `gguf_loader.c` now materializes the classifier for Q2_0-native models — but **only**
  when the user explicitly passes `--classifier` (new `TnHardwareProfile.classifier_explicit` flag,
  set only in `tn_hardware_profile_set_classifier()`), so the default zero-copy path is unaffected
  and the multi-GB cost is opt-in, not forced on every load. When explicit, it dequantizes the raw
  Q2_0 output tensor to F32 (`gguf_dequant_q2_0`), rounds to BF16 (new `f32_to_bf16_round`,
  round-to-nearest-even — not a naive truncation, which would bias the whole tensor one direction),
  and reuses the existing `weights_build_classifier_quant()` (previously only called from the
  non-Q2_0 path) to derive INT8/INT4 from that BF16 base. `forward.c`'s Q2_0 branch now checks
  `classifier_explicit` and dispatches to `parallel_matmul_i4`/`_i8`/`_bf16` against the
  materialized buffers, falling back to the original zero-copy `parallel_matmul_q2_0` only when no
  explicit format was requested. The materialized `w->wcls` is registered via the existing
  `GGUFWeightStore` (`store_add`) — confirmed no double-free risk since `weights_free_pointers`
  never frees `w->wcls` directly (only `wcls_i8`/`wcls_i4`/their scales, which `weights_build_
  classifier_quant` itself owns).
- Verified real, differentiated results (strictly sequential runs, freshly recalibrated each time,
  idle host, 4 threads, Ternary-Bonsai-27B, 40-token generations):
  `auto` (default zero-copy Q2_0) = **1.19 tok/s**, `--classifier bf16` = **1.13 tok/s**,
  `--classifier int8` = **2.60 tok/s**, `--classifier int4` = **2.62 tok/s**. Three genuinely
  different, ordered numbers this time — not three measurements of the same code path.
- Unexpected finding worth recording: INT8/INT4 are ~2.2x *faster* than the zero-copy default, even
  though raw Q2_0 (2.125 bits/weight) is smaller than materialized INT4 (4 bits) or INT8 (8 bits) —
  the opposite of what "smaller footprint = faster" would predict. The general-purpose VNNI
  int8/int4 dot-product kernel (`parallel_matmul_i4`/`_i8`) is apparently more compute-efficient per
  element for this matmul shape than the specialized Q2_0 decode-and-FMA kernel, so here the extra
  bytes read are outweighed by cheaper per-element math. Don't assume a smaller quantization format
  is always faster without measuring — bandwidth and compute-efficiency can trade off either way
  depending on the kernel, not just the byte count.
- Affected files: `include/core/hardware_profile.h` (`classifier_explicit` field),
  `src/core/hardware_profile.c` (sets it), `src/core/gguf_loader.c` (materialization +
  `f32_to_bf16_round`), `src/transformer/forward.c` (dispatch), `src/cli/main.c` (removed the
  now-obsolete no-op warning).
- Detection: user explicitly rejected both the "noise floor" explanation and the warning-only fix,
  in two separate rounds of pushback.
- Prevention rule (unchanged, still holds): when multiple configurations of a flag measure
  *identically*, check whether the flag reaches the code path being measured before blaming
  environment noise. New rule from this round: when a bug is found and a full fix is judged "too
  large" to take on unprompted, flagging it explicitly (as this session did) is the right call *only
  as a proposal* — it is not a substitute for doing the fix once asked, and the flagged concern
  itself may be smaller than estimated (this fix touched 4 files, not a full architecture change).

### 2026-07-16 — Underlying virtualized host was recycled mid-session, invalidating a same-host performance comparison
- Summary: while investigating the classifier-tok/s question above, `/proc/uptime` showed this
  Firecracker microVM had been running only ~5 minutes, despite the conversation's own benchmark
  timeline spanning hours. The thread-sweep numbers (2.74 tok/s @ t=4) and the classifier-sweep
  numbers (1.07-1.09 tok/s) were measured on what the engine's own hardware profiler confirms were
  *different* machines in practice: L2 cache 2048→1024 KiB/core, L3 260→33 MiB, DRAM bandwidth
  14.0-17.3→9.9-12.3 GB/s between the two sweeps. This is real, verifiable hardware drift, not
  measurement error — but (see the entry above) it was NOT the primary explanation for
  BF16≈INT4≈INT8; that was the no-op dispatch bug. Both things are true at once: the classifier
  flag never took effect for this model, *and* the underlying host changed between sweeps.
- Detection: `cat /proc/uptime` compared against the session's own wall-clock timeline.
- Prevention rule: on sandboxed/cloud dev environments, a same-session "before/after" hardware
  comparison is not automatically apples-to-apples — check `/proc/uptime` (or equivalent) when a
  performance number looks surprising, especially after a long gap of unrelated heavy work
  (multiple full compiler rebuilds, ASan test suites, etc.) between the two measurements being
  compared, since that gap is exactly when environment recycling is likely to have happened.

### 2026-07-16 — Startup banner now matches llama.cpp: always shown in a real TTY, not just interactively
- Summary: project-zero's animated ASCII banner was gated on `stdout_is_tty && (server_mode ||
  !prompt)` — shown for the interactive REPL and `--server`, suppressed for one-shot `--prompt`
  runs (deliberately modeled on Claude Code's own CLI convention, per the removed comment). User
  asked why llama.cpp's benchmark screenshots showed its banner even in one-shot mode. Read
  llama.cpp's actual source (`tools/cli/cli.cpp:429`, `console::log("%s\n", LLAMA_ASCII_LOGO)`) —
  it has no such gate at all; the banner is unconditional regardless of `-p`/`--single-turn`.
- Correction: changed the condition to just `stdout_is_tty` (`src/cli/main.c`) — the banner now
  shows for any real-terminal invocation, one-shot or not, matching llama.cpp's actual behavior and
  keeping benchmark screenshots self-identifying. Piped/redirected output (not a TTY) still gets
  plain text, so scripted/automated usage is unaffected.
- Verification: confirmed via a real pty capture (not just reading the source) that (1) the old
  code's interactive-mode banner display actually worked before this change, and (2) the new code
  shows the banner with `--prompt` set too, both by directly running the binary through `script`
  and inspecting the captured raw output.
- Affected files: `src/cli/main.c`.

### 2026-07-16 — Pre-existing compiler warnings across the codebase, fixed on explicit reminder of the bug-fix policy
- Summary: while doing the two fixes above, `make clean` + full gcc/clang rebuilds surfaced a batch
  of pre-existing warnings unrelated to either fix — initially noted as "pre-existing, not a
  regression" and left alone. User explicitly invoked the project's own bug-fix policy ("even if
  something is preexisting, you are not allowed to leave it as it is"), so all were fixed in the
  same pass rather than deferred:
  - `src/cli/main.c`: unused `simd_backend`/`free_ram` locals (the real RAM-aware sizing already
    happens later via a fresh post-model-load `tn_get_free_ram()` call — this local was dead, not a
    missing wire-up); `void *` pointer arithmetic on `mf.data` (cast to `char *` first).
  - `src/core/weights.c`: three `if (a) x=b; if (c) x=d;` pairs on one line (misleading-indentation
    warning) — correct logic, just reformatted onto separate lines.
  - `src/core/unpack_avx2.c`: two dead scratch variables (`shift_amounts`, `shift32`) explicitly
    marked "dummy" / superseded in their own comments, left over from writing the function.
  - `src/math/matmul_q4k.c`: `hsum8_epi32` fully dead (superseded by inline hsum code elsewhere,
    removed); `q4k_decode_scales_raw` only used by the scalar (non-AVX2) fallback, now compiled
    only under `#if !TN_HAS_AVX2` instead of unconditionally.
  - `tests/test_harness.h`: the three test-counter statics aren't used by every test file that
    includes the header (e.g. audit-style files with their own pass/fail output) — marked
    `__attribute__((unused))` since the header provides them unconditionally by design.
  - `tests/test_simd_vnni.c`: a tautological `q[i] > 127` check on an `int8_t` (its own max value —
    can never be true), removed.
  - `tests/test_forward.c`: dead helper `fill_pattern_i8`, never called, removed.
  - `tests/forensic_audit_suite.c` (10), `tests/test_vision_components.c` (5),
    `tests/test_redbox.c` (1): K&R-style `foo()` no-prototype declarations → explicit `foo(void)`.
  - `tests/test_vision_e2e.c`: implicit `int`→`float` narrowing of `RAND_MAX` in `rand() /
    RAND_MAX` → explicit `(float)RAND_MAX` cast (same rounding, no behavior change, just no longer
    silent).
  - `src/multimodal/image_load.c`: one warning from the vendored `stb_image_resize2.h` (third-party
    code) — suppressed via `#pragma GCC diagnostic push/ignore/pop` around its include rather than
    patching the vendor file, to avoid merge friction on future vendor updates.
- Verification: `make clean` + full gcc and clang release/test/debug, plus a from-scratch
  ASan/UBSan run with `$(LIB_OBJS)` genuinely instrumented (not just test files) — all green, zero
  warnings, zero failures, for both compilers.
- Prevention rule: "pre-existing and unrelated to the current task" is never sufficient reason on
  its own to leave a warning or bug in place — the project's own rule requires fixing it in the
  same pass unless it's a genuinely large architectural change (and even then, only after flagging
  it explicitly, not silently skipping).

### 2026-07-16 — CPUID lied about AVX-512VBMI on this virtualized host: SIGILL crash on every first-time calibration (and any `--classifier` use)
- Summary: while running a `--classifier` sweep for benchmark documentation, every invocation
  crashed with `SIGILL` ("illegal instruction"), reproducibly, inside `tn_calibrate()`. Root-caused
  under `gdb` (backtrace + disassembly at the exact faulting `$rip`) to a `vpermi2b`-family
  AVX-512VBMI byte-permute instruction. Both this build's compile-time detection (`-march=native`
  → `__AVX512VBMI__` → `TN_HAS_AVX512VBMI=1`) **and** the engine's own runtime CPUID probe
  (`cpu_features.c`: `f->avx512vbmi = (ecx>>1)&1`, printed "AVX-512 VBMI : YES" in every prior run
  this session) agreed VBMI was available — but actually *executing* a VBMI instruction faults on
  this host. This is a Firecracker microVM (confirmed via `dmesg`); the hypervisor's CPUID leaf
  advertises a feature bit the underlying execution unit cannot actually retire — a known class of
  virtualization CPUID-passthrough bug, not something either compile-time or CPUID-only runtime
  detection can catch.
- Why it wasn't caught earlier: this session's Q2_0/tokenizer work (and the `bitunpack2_vnni.h`
  file this same session added the VBMI fast path to) never happened to exercise the VBMI code
  path — Q2_0 inference apparently hit the SSE fallback path in practice, or got lucky with
  dead-code-adjacent scheduling. The crash only surfaced via `tn_calibrate()`'s Phase-2 classifier
  benchmarking (which profiles bf16/int8/int4 unconditionally, regardless of what `--classifier`
  the user actually requested) and via `parallel_ternary_matmul_packed`'s VNNI pre-quantization
  fast path (itself only reachable once `TN_FORCE_BACKEND=scalar`-style calibration testing forces
  a code path real inference doesn't normally take) — both call chains ultimately reach VBMI-gated
  unpack code with no runtime safety net. Two independent call sites hit the identical class of bug:
  `include/math/bitunpack2_vnni.h` (Q2_0/ternary-packed 2-bit unpack, this session's own new file)
  and `src/math/parallel_matmul.c`'s `matmul_i4_task` (pre-existing INT4 classifier unpack).
- Root cause: both call sites gated their VBMI-vs-fallback code selection with `#if
  TN_HAS_AVX512VBMI` alone — a **compile-time-only** macro — with no accompanying runtime check,
  so there was no way to fall back to the safe SSE/AVX2 path even though the codebase already has
  a proper runtime CPU-feature-detection module (`cpu_features.c`) that could have been consulted.
- Fix: added a real **execution-verified** runtime check in `cpu_features.c`
  (`verify_avx512vbmi_executable()`) — installs a `SIGILL` handler via `sigaction`, executes one
  real `_mm512_permutexvar_epi8` under `sigsetjmp`/`siglongjmp`, and downgrades
  `f->avx512vbmi` to `false` if it faults (or produces a wrong result) — run once, at startup, only
  when the raw CPUID probe already said VBMI was present. Converted both consumer call sites from
  compile-time-only branching to compiling **both** implementations always (whenever baseline
  AVX-512VNNI is available) and dispatching between them via this verified runtime flag: renamed
  `bitunpack2_vnni.h`'s two variants to `tn_unpack64_to_wenc_u8_vbmi`/`_sse` behind a small runtime-
  dispatching wrapper (cached per-TU `static int`, same pattern as `tn_simd_init()`'s one-time
  probe); `matmul_i4_task` now hoists `bool use_vbmi_i4 = tn_cpu_features_detect()->avx512vbmi;`
  once per task and branches on it inside the per-row loop instead of `#if/#else`.
- Detection: `gdb -batch -ex run -ex bt -ex "disassemble $rip-100,$rip+30"` on the crashing binary,
  after confirming reproducibility with `dmesg | grep "trap invalid opcode"` (deterministic fault
  offset across multiple runs, ruling out a rare/racy heisenbug).
- Correction: see fix above; verified the exact crash scenario (`--classifier int4/int8/bf16` with
  a cleared calibration cache, forcing fresh calibration) no longer crashes — calibration now
  completes both phases cleanly end to end.
- Prevention rule: on x86, a CPUID-reported feature bit is a *claim*, not a *guarantee*, especially
  inside a hypervisor/microVM — any code path gated on an advanced ISA extension (VBMI, AMX, etc.)
  that could plausibly run inside a VM should verify real execution once at startup (SIGILL-trapped
  self-test) rather than trusting CPUID alone, and that verified flag — not the raw compile-time
  macro — is what every consumer must branch on. A `#if COMPILE_TIME_MACRO` with no runtime
  fallback is only safe for ISA tiers guaranteed by the OS/ABI baseline; anything gated by dynamic
  CPUID needs a dynamic (and, for advanced extensions, execution-verified) escape hatch.

### 2026-07-16 — Both previously-flagged items fixed on user pushback; `make test` alone never actually ran ASan/UBSan over library code, which hid 2 more real bugs
- Summary: user directly challenged the earlier "flag, don't fix" calls on the Q2_0 batch/MoE VNNI
  path and the byte-level-BPE detokenizer ("Why did u not fix the remaining 2?"). Both are now
  fixed (see below). While re-verifying with a genuinely ASan/UBSan-instrumented full test suite
  (see the build-methodology finding below), 2 more real, independent, pre-existing memory-safety
  bugs surfaced and were fixed in the same pass, per the bug-fix policy.
- **Fix 1 — Q2_0 batch/MoE VNNI path**: added `parallel_matmul_q2_0_batch_vnni()` to
  `src/math/matmul_q2_0_vnni.c` (same w_enc bias trick as the single-matrix path, per-expert
  quantization buffers heap-allocated since expert count `k` is a runtime value), wired into
  `src/math/matmul_q2_0.c`'s `parallel_matmul_q2_0_batch()` with the same VNNI-first/portable-
  fallback dispatch as the single-matrix function. Turned out to be **dead/unreachable code
  today** — `grep` confirmed `parallel_matmul_q2_0_batch` has no real caller anywhere in the
  engine (`moe_ffn.c` only dispatches q4k/q5_1/q5_0/q8_0 batch variants; no Q2_0-MoE model is
  wired up yet) — so the earlier "not on the measured hot path, so left slow" justification was
  true but not actually a reason to defer: there was no real tradeoff being avoided, just
  unfinished work. Verified via direct synthetic unit tests in `tests/test_q2_0_matmul.c`
  (`test_batch_matches_single_matrix_path`, `test_batch_mixed_zero_activation` — batch output
  compared against the already-verified single-matrix path called once per expert, since there's
  no real end-to-end caller to test through).
- **Fix 2 — byte-level BPE detokenizer**: rewrote `src/tokenizer/tokenizer_decode.c` with the
  full 256-entry GPT-2 byte<->unicode reverse table (`g_bpe_cp_to_byte[324]`, codepoints
  0x000-0x143) and a UTF-8 codepoint decoder, replacing the old 2-case `clean_bpe_string()`. The
  originally-assumed blocker ("needs a stateful multi-byte-UTF-8 accumulator across calls") turned
  out to be **wrong** on closer inspection: since the function reverse-maps codepoints back to raw
  *bytes* (not characters) and callers already concatenate each call's output byte-for-byte, a
  multi-byte UTF-8 character's raw bytes reassemble correctly purely from concatenation — no
  cross-call state needed. The real complexity was smaller than the original deferral assumed.
  Also fixed the BOS-leading-space-strip check while in this code: it checked for a literal `' '`
  byte, which only matches this project's native/legacy binary tokenizer format (literal text) —
  GGUF byte-level-BPE vocabs (`tokenizer_gguf.c`) store the same leading space as the 2-byte `Ġ`
  marker (0xC4 0xA0), which the literal check never matched, so BOS-adjacent leading spaces were
  never stripped for GGUF-loaded models. Fixed to check both forms (native format's test coverage
  in `test_tokenizer.c` proves the literal-space path still works unchanged). New test coverage:
  `test_tokenizer_decode_byte_level_space`, `_latin1`, `_multibyte_emoji` (the exact ✅ mojibake
  case, split across 3 synthetic single-byte tokens the way a real tokenizer would), `_after_bos`.
- **Build-methodology finding**: this Makefile's generic `build/%.o: src/%.c` rule uses
  `$(CFLAGS)` (make-variable, not hardcoded), so `debug:`/`release:` targets recursively invoke
  `$(MAKE)` with explicit `CFLAGS="..."` overrides — but `test:` does **not**; it uses whatever
  `CFLAGS` is already in effect (default: `CFLAGS_RELEASE`). Meanwhile the *test-binary* pattern
  rule (`build/tests/%: ... ; $(CC) $(CFLAGS_DEBUG) ...`) is hardcoded to always use debug/ASan
  flags for the test `.c` file itself, regardless of what built `$(LIB_OBJS)`. Net effect: a plain
  `make test` (including the documented canonical sequence `make release && make test && make
  debug`) compiles every test file with ASan/UBSan, but links against whatever `$(LIB_OBJS)`
  already exist on disk — which, after a preceding `make release`, are **release-mode, not
  instrumented**. ASan's malloc/free interposition is process-wide once *anything* links libasan,
  so heap corruption in release-compiled library code is still often caught, but this is weaker
  coverage than it looks and is NOT what `.claude/rules/tests.md` describes ("`make test` ...
  under ASan/UBSan" — true for the test file, not for `$(LIB_OBJS)` in this ordering). Separately,
  because make's staleness check is mtime-based (not flag-aware), running `make debug` right after
  `make release && make test` is **frequently a silent no-op** — the `.o` files are already
  "up to date" relative to unchanged `.c` sources, so the debug/sanitizer flags never actually get
  applied, and `make debug` reports success while leaving the release-mode binary untouched. This
  is not a regression from anything in this session — it's a pre-existing structural gap in the
  Makefile, discovered while trying to get *real* ASan/UBSan coverage of the code touched by this
  entry's two fixes. Verified real coverage instead via `make clean` followed by a single `make`
  invocation with `CFLAGS`/`CXXFLAGS`/`LDFLAGS` all explicitly overridden to the debug/sanitizer
  values before `test`, forcing every `$(LIB_OBJS)` and test binary to be freshly compiled and
  linked with instrumentation in one consistent invocation.
- **Bug found via the above (1) — heap-buffer-overflow in the AVX-512BW LUT ternary kernel**:
  `lookup_and_acc()` in `src/math/ternary_matmul_lut_avx512bw.c` loaded a full 64-byte
  (`_mm512_loadu_si512`) vector of packed weight bytes per 32-column block, but only ever used the
  low 32 bytes (`_mm512_extracti32x8_epi32(abs_v, 0)`) — the upper half was dead computation, not
  just wasted bandwidth. For interior column blocks this silently over-read into the next K-group's
  row of the same buffer (wrong data used nowhere, since only the low 32 bytes are consumed, so
  numerically harmless); for the **last** column block of the **last** K-group, it read 32 bytes
  past the actual end of the packed-weight allocation (`P`, sized exactly `(K/5)*N` bytes, no
  padding) — a real heap-buffer-overflow. Confirmed via ASan on `test_lut_matches_scalar_full_blocks`
  (K5=8, N=64, P=512B exactly — the read window `[480,544)` exceeds the 512-byte allocation by
  32 bytes). Fixed by loading exactly the needed 32 bytes via `_mm256_loadu_si256` +
  `_mm256_abs_epi8` + `_mm256_movepi8_mask` instead of the 512-bit load, removing the dead upper
  half entirely rather than just masking it off.
- **Bug found via the above (2) — test harness under-allocation in `test_moe.c`**: `moe_gate_w[l]`
  is stored as a `tn_i8*` pointer but `moe_ffn.c` casts it to `const float*` before calling
  `moe_router_forward()` (gate weights are raw F32, unlike the ternary-quantized expert FFN
  weights — see that file's own comment: "Gate weight is w->moe_gate_w[layer] — F32
  [num_experts × dim]"). `test_moe.c`'s `test_moe_ffn_output_finite` allocated this buffer with
  `calloc(dim*ne, sizeof(tn_i8))` (1 byte/element) instead of `sizeof(float)` (4 bytes/element) —
  a 4x under-allocation that the router's AVX-512 F32 SIMD load read past. This is a test-harness
  bug, not a source bug; fixed the allocation size in the test.
- Detection: user pushback on the original deferral decision, prompting genuine implementation
  rather than continued justification; the 2 new bugs surfaced only once library object files
  were actually sanitizer-instrumented during a test run (see build-methodology finding above).
- Correction: see each fix above. `make release`/`make test`(release-mode)/full ASan+UBSan test
  run with genuinely instrumented `$(LIB_OBJS)`/`make debug` (from clean) all green for gcc and
  clang.
- Prevention rule: when a deferral decision rests on "this isn't on the hot path" or "this needs
  a bigger rewrite than it's worth right now," verify both claims concretely (grep for real
  callers; sketch the actual minimal fix) before accepting them — both turned out to be wrong
  here in ways that would have been cheap to check upfront. Separately: a Makefile's `test` target
  claiming ASan/UBSan coverage is only as strong as its actual object-file provenance — verify
  what compiled `$(LIB_OBJS)` before trusting a "tests pass under sanitizers" claim, especially in
  a project with a shared `build/` directory across release/debug configurations and no flag-hash
  staleness tracking.

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

### 2026-07-16 — Q2_0 matmul was 1% of this host's own bandwidth ceiling; VNNI kernel closes it (~29x)
- Summary: project-zero measured ~0.11-0.12 tok/s on the real Ternary-Bonsai-27B model, against
  the engine's own hardware profiler's computed ceiling of ~17 tok/s (100% DRAM bandwidth) for
  this host/model — i.e. running at roughly 1% of what this host can actually do. The reference
  engine (PrismML-Eng's llama.cpp fork) was *also* far from that ceiling (~0.9-1.2 tok/s), which
  in hindsight was the tell: this is a genuinely fresh (2-day-old at the time) quantization
  format, so neither engine had a tuned kernel for it yet — not a hardware limit.
- Root cause: `matmul_q2_0.c`'s only kernel decoded each 128-element Q2_0 block into a stack
  float buffer, then ran an AVX2 FMA reduction — correct, but this project already has a proven,
  much faster pattern for exactly this situation (`ternary_matmul_packed_vnni.c`'s AVX-512 VNNI
  "w_enc bias trick": `dpbusds(w_enc_u8, q_x_i8) = dot(w,q_x) + sum(q_x)`, so
  `true_dot = result - sum_qx`) that was never applied to Q2_0, despite GGUF Q2_0's
  code-to-value mapping (0/1/2/(3) -> -1/0/+1/(+2), i.e. `w_enc = w+1`) being bit-for-bit
  identical to this project's own native packed-ternary format that pattern was written for.
- Affected files/modules: new `src/math/matmul_q2_0_vnni.c` (the fast path, guarded
  `#if TN_HAS_AVX512VNNI`), new `include/math/bitunpack2_vnni.h` (the 2-bit unpack primitive,
  extracted out of `ternary_matmul_packed_vnni.c` so both kernels share one implementation
  instead of two copies of the same bit-trick), `src/math/matmul_q2_0.c` (now the portable
  fallback: non-VNNI hosts, or any input shape the VNNI kernel declines), new
  `tests/test_q2_0_matmul.c` (17 cases; a first attempt at this test used *raw* float x as the
  reference, which produced false failures — see below), `Makefile`/`CMakeLists.txt` (new file's
  build rules, mirroring the sibling VNNI kernel's exactly).
- A meta-mistake caught while writing the test: the first test compared the VNNI kernel's output
  against a reference computed with the *raw, unquantized* float activations. VNNI hardware
  fundamentally requires int8 operands, so the kernel necessarily quantizes activations first —
  comparing against the un-quantized reference conflates that expected, by-design quantization
  error with a real kernel bug, and produced 3 "failures" that were actually just wide
  (non-realistic-magnitude) synthetic test inputs making the expected error visible. A one-hot
  activation probe (`x[j]=1, everything else 0`, sweeping every `j`) proved the bit-unpack/decode
  itself was bit-exact correct — the fix was rewriting the reference to quantize x the same way
  the kernel does (`quantize_row_to_i8`) and compare against *that*, isolating "is the VNNI math
  right" from "int8 quantization is lossy by design."
- Detection: `docs/ai/mistakes.md`'s own prior entry (this session's earlier Qwen 3.6 work)
  already flagged this as a known, deliberately-deferred performance limitation; fixed now on
  explicit user request, verified end-to-end on the real model (see decision-log.md for the
  before/after numbers and the exact-sampled-token-sequence correctness check).
- Correction: see Affected files above.
- Prevention rule: when a new SIMD kernel's correctness test uses int8/quantized intermediate
  representations, the reference must replicate that same quantization step — comparing against
  full float precision measures quantization error, not kernel correctness, and produces
  misleading test failures on wide-range synthetic inputs while passing on narrow ones (which is
  exactly backwards from what a test should do). A one-hot/basis-vector probe is a fast, precise
  way to distinguish "the decode is wrong" from "the accumulated numeric approximation is
  (expectedly) different" when a full-vector test fails ambiguously.

### 2026-07-16 — FLAGGED, NOT FIXED: byte-level BPE detokenizer only reverses 2 of ~256 GPT-2 byte remappings
- Summary: found incidentally while reviewing a real-model output screenshot — a single ✅ emoji
  in the model's own "thinking" trace rendered as mojibake (`âľħ`) instead of the correct
  character. `src/tokenizer/tokenizer_decode.c`'s `clean_bpe_string()` only reverses two of
  GPT-2's byte-level-BPE unicode remappings (`Ġ`->space, `Ċ`->newline). The real GPT-2 tokenizer
  scheme remaps *all* 256 possible raw bytes to printable-range unicode codepoints (a fixed,
  well-known permutation: printable ASCII/Latin-1 map to themselves, the ~68 unprintable byte
  values map to codepoints U+0100 upward) — any raw byte in that ~68-value hidden range,
  including every byte of a multi-byte UTF-8 character (emoji, CJK text, box-drawing, etc.) that
  happens to land in it, decodes wrong today.
- Root cause: `clean_bpe_string()` was written to handle the two remappings a plain-ASCII/Latin-1
  chat transcript actually exercises, not the full byte-decoder table — reasonable when only
  ASCII text had been verified, incomplete now that a real run produced non-ASCII output.
- Why flagged instead of fixed in this pass: this is a **pre-existing bug, unrelated to the
  Qwen 3.6 work** (the same `tokenizer_decode()` path is shared by every model this engine
  supports, so it already affects any model emitting emoji/CJK/other non-Latin-1 text — this
  session's testing just happened to be the first to produce a non-ASCII character and notice).
  Fixing it properly needs: (1) the full 256-entry GPT-2 byte<->unicode reverse table, and more
  importantly (2) a stateful multi-byte-UTF-8 accumulator across `tokenizer_decode()` calls,
  since a single multi-byte character's raw bytes can be split across multiple separate BPE
  tokens (multiple calls) — a genuinely stateful change to a `__thread`-buffer-based function
  that's on the hot path for *every token of every model*, not a contained, low-risk fix like the
  ones above. Per engineering-rules.md's bug-fix policy ("only defer for a genuinely large
  architectural change, and flag that to the user explicitly instead of silently skipping it") —
  this qualifies: rushing a stateful rewrite of a shared, heavily-exercised core path without
  dedicated test coverage risks a real regression across every existing model/test, which is a
  worse outcome than a narrow, understood, flagged cosmetic gap.
- Affected files/modules: `src/tokenizer/tokenizer_decode.c` (not yet modified).
- Detection: visual review of a real-model output screenshot (`project-zero-full.png`) — not
  caught by any existing test, since none exercise non-ASCII/emoji output.
- Correction: NOT APPLIED at the time this entry was written. **Superseded**: fixed the same day
  after user pushback — see the 2026-07-16 entry at the top of this file ("Both previously-flagged
  items fixed on user pushback...") for the actual fix and why the "stateful accumulator" blocker
  assumed here turned out not to be necessary.
- Prevention rule: a byte-level-BPE detokenizer needs the *complete* reverse byte table, not just
  the couple of remappings a given test corpus happens to touch — ASCII-only test coverage will
  never catch this class of gap. If/when this gets fixed, verify against real multi-byte UTF-8
  output (emoji, CJK) specifically, and add it as a permanent test case (something this engine's
  test suite currently has zero coverage for).

### 2026-07-16 — `loop.previtem`/`loop.nextitem` documented as supported, never implemented; 2 more real gaps
- Summary: this session's own earlier chat_template.h update (same day, during the Qwen 3.6 work)
  claimed `loop.previtem`/`loop.nextitem` were supported — they were never actually set anywhere
  in the `NT::For` exec() case. The real Qwen 3.6 template uses both
  (`loop.previtem.role != "tool"` / `loop.nextitem.role != "tool"`) to detect tool-response
  message boundaries in multi-turn tool-calling conversations. Found via explicit re-audit
  ("fix all shortcomings") rather than a failing test — nothing had exercised the tool-role path
  yet, so the gap between the doc comment and the code was silent.
- Root cause: documentation written aspirationally (matching real Jinja's `loop` object surface)
  without cross-checking every field was actually implemented.
- Also fixed in the same pass, found while auditing this file's other "not supported" list against
  what the real template's tool-calling section actually needs:
  1. Tuple-unpacking for-loops (`{% for k, v in dict|items %}`) previously discarded the second
     loop variable entirely (`parse_for` only ever captured one name) — no way to bind both.
  2. No `|items` filter existed at all (needed to turn a dict/namespace Object into `[k,v]` pairs
     for the above to iterate over).
  3. Method calls parsed correctly off a variable (`content.split(...)`) but desynced the parser
     when called directly on a literal (`'x'.upper()`) — `parse_primary`'s postfix-chain loop
     (`[key]`/`.attr`/`(args)`) only applied after the `Ident` case, not after string/int/bool/
     none literals. Not exercised by the real template (it only calls methods on variables), but
     a real, fixable gap — parse_primary now builds any primary first, then applies the postfix
     loop uniformly regardless of what kind of primary it was.
- Affected files/modules: `src/tokenizer/chat_template.cpp` (`parse_for`, `parse_primary`,
  `NT::For`'s exec() case, `items` filter in `NT::Filter`'s exec() case),
  `include/tokenizer/chat_template.h` (corrected the doc comment to match reality),
  `tests/test_chat_template.c` (3 new cases: items+tuple-unpacking, previtem/nextitem, method
  call off a variable still works post-refactor).
- Detection: explicit re-audit against the real template's tool-calling section, prompted by the
  user asking whether any shortcomings were still undocumented/pending.
- Correction: see Affected files above.
- Prevention rule: when a doc comment claims a feature is supported, that claim needs the same
  verification bar as the code itself — either a test proves it, or don't claim it. An
  undocumented gap is at least honest; a *mis*documented one is worse, because it looks resolved
  and won't get re-audited until something downstream silently breaks.

### 2026-07-16 — Qwen 3.6 (Ternary-Bonsai-27B) integration: 6 real bugs, found via a real model + a real reference engine
- Context: implementing Qwen 3.5/3.6 hybrid Gated-DeltaNet + Gated-Attention support end-to-end
  (new files: `src/transformer/qwen35_attention.c`, `src/core/qwen35_run_state.c`,
  `src/math/matmul_q2_0.c`) and running the actual downloaded `prism-ml/Ternary-Bonsai-27B-gguf`
  through it. Every bug below was found only because the *real* model was actually run to
  completion (not just compiled/unit-tested) and, for the last one, only because a second,
  independently-implemented reference engine (a from-source build of PrismML-Eng's `llama.cpp`
  fork, `prism` branch) was available to run the identical prompt side-by-side. This is a
  concrete case for the project's "verify with a real end-to-end run, not just tests" principle
  (see the 2026-07-15 API entry below) generalizing beyond the HTTP layer.

  1. **Chat-template parser: keyword-argument call syntax spun forever, allocating AST nodes,
     until OOM.** `namespace(value=0)` and `namespace(multi_step_tool=true, last_query_index=...)`
     (used by Qwen 3.6's real `chat_template.jinja` to seed loop state) hit the call/filter
     argument-list parser's `while (!Rparen) { push_back(parse_expr(p)); ... }` loop. `=` is not a
     valid expression-start token, so `parse_expr` fell through to `parse_primary`'s literal-none
     fallback *without consuming any token* — the loop never terminated, allocating a new AST node
     every iteration: a real, reproducible multi-GB-in-seconds OOM on the actual downloaded GGUF's
     template (confirmed via a standalone test harness against `chat_template.cpp` directly,
     bypassing the ~100s full-engine cycle — RSS grew ~140MB/s with a fixed stack depth, proving a
     non-advancing loop, not runaway recursion). Fixed by wrapping call/filter arguments in an
     `NT::Kwarg` node (`parse_call_arg()`) that consumes `name =` before parsing the value — this
     is also what actually stops the OOM, since it guarantees forward progress — plus a
     forward-progress assertion (`p.pos == before` → throw) in both argument loops as defense in
     depth against the next unanticipated non-advancing construct.
  2. **Chat-template parser: `{% macro %}` bodies were skipped entirely (never executed), so every
     message's content — routed through Qwen 3.6's `render_content` macro — rendered as empty
     string.** This silently produced a syntactically-valid but content-free prompt (`<|im_start|>
     user\n<|im_end|>`) with no crash, which would have made any inference benchmark meaningless.
     Fixed by actually parsing (`parse_macro()`, new `NT::MacroDef`/`NT::Param` node types) and
     invoking macros (positional/default/keyword parameter binding, executed in the macro's
     *definition* scope per real Jinja semantics, output captured into a local string and returned
     as the call's value) instead of skipping them.
  3. **Chat-template parser: `{% set ns.attr = expr %}` (dotted target) overwrote the *whole* `ns`
     namespace object with the RHS value instead of mutating one field**, because `parse_set` only
     ever parsed a single bare identifier as the assignment target. Real chat templates mutate a
     `namespace()` object's fields in exactly this way (`ns.multi_step_tool = false`). Fixed by
     parsing a dotted path into `Node.sval` (`"ns.attr"`) and, in `NT::Set`'s exec() case, walking
     to the target var via a new `Scope::get_ref()` and mutating `target->obj[attr]` in place.
  4. **Chat-template lexer: `-%}`/`-}}` whitespace-control used a single `strip_next` flag
     consumed by whatever the *next* lexed Text token happened to be** — correct only when a tag
     is immediately followed by literal text. When two tags are source-adjacent with nothing
     between them (`{%- endmacro -%}{{- foo }}`), the flag survived past the empty gap and
     stripped leading whitespace off a *later, unrelated* text run instead of the (empty)
     whitespace actually following the first tag. Found via a test comparing exact rendered
     output (`Hello World! Hello Zig?` came out as `Hello World!Hello Zig?`, a real, silent
     one-space corruption). Fixed by consuming the stripped whitespace directly from the source
     cursor at the point the tag closes, instead of deferring to the next Text token.
  5. **`q35_run_state_alloc()`'s own 600MB RAM-budget clamp on `max_seq_len` never propagated back
     to `s->max_seq_len`.** `run_state_alloc_ex()` sets `s->max_seq_len` to the *original,
     unclamped* seq_len earlier in `main.c`; `q35_run_state_alloc()` then computes a *smaller*
     local `max_seq_len` to fit its fixed budget and allocates `q35_key_cache`/`q35_value_cache`
     at that smaller size — but never updated `s->max_seq_len`. `qwen35_attention.c`'s cache-offset
     arithmetic (`(kh * s->max_seq_len + pos) * head_dim`) trusted the stale, larger value as the
     buffer's stride, writing out of bounds for any `kh>0` on literally the first full-attention
     layer of the first generated token. Manifested two different ways depending on what the wild
     write landed on: a reproducible SIGSEGV inside `qwen35_attention_forward`'s `memcpy` (caught
     via `gdb -batch -ex run -ex bt`), and, in one run, a genuine kernel OOM-kill (`anon-rss:
     15.9GB`) when the same out-of-bounds writes happened to fault in a large span of fresh
     anonymous pages before hitting a truly unmapped one. Fixed by assigning
     `s->max_seq_len = max_seq_len;` right after the budget-clamp block, so every consumer sees
     the same (correctly-allocated-against) value.
  6. **Gated DeltaNet's depthwise causal conv1d read `ssm_conv1d.weight` with a transposed
     layout**, indexing it as `[conv_k][conv_dim]` (tap-major: `kernel[i*conv_dim+c]`) when the
     tensor's actual GGUF shape is `[conv_k, conv_dim]` in `ne[]` order — i.e. `ne0=conv_k` is the
     *fast/contiguous* axis, so the real physical layout is channel-major:
     `kernel[c*conv_k+i]`. Confirmed from the real file's own tensor dims
     (`blk.0.ssm_conv1d.weight [4, 10240]`) and cross-checked against llama.cpp's
     `create_tensor(..., {ssm_d_conv, conv_dim}, ...)`, which declares dims in the same `ne0`-first
     order. This silently read the wrong tap weight for every (tap, channel) pair outside the
     accidental overlap on every one of the 48 linear-attention layers, corrupting every
     DeltaNet layer's q/k/v inputs while still producing finite, plausible-looking activations
     (no NaN, no explosion, "max_abs" growing across layers the way transformer residual streams
     normally do) — the *only* externally visible symptom was that greedy decoding immediately
     picked EOS as the very first generated token for every prompt tried. This is the one bug in
     this list that pure code review against the reference source did not catch on its own: the
     reference's math (decay/delta-rule/readout/gates) all matched exactly on inspection: it took
     actually building and running PrismML-Eng's `llama.cpp` fork on the *identical* file and
     prompt, getting coherent output ("Here's a thinking process: 1. **Identify the User's
     Question**...") where project-zero got instant EOS, to prove there *was* a remaining bug
     worth hunting for, and confirming the tensor's physical `ne[]` layout (not just its logical
     shape) was the final, decisive check.
- Affected files/modules: `src/tokenizer/chat_template.cpp` (parser/lexer, items 1-4),
  `src/core/qwen35_run_state.c` (item 5), `src/transformer/qwen35_attention.c` (item 6),
  new test `tests/test_chat_template.c` (18 cases covering items 1-4 and pre-existing behavior).
- Detection: a standalone C++ test harness linking only `chat_template.cpp` (bypassing the ~100s
  full-engine load cycle) for items 1-4; `gdb -batch -ex run -ex bt` plus `dmesg` OOM-kill records
  for item 5; a second, independently-built reference engine run on the identical GGUF file and
  prompt for item 6.
- Correction: see per-bug descriptions above.
- Prevention rule: (a) any recursive-descent parser loop of the shape
  `while (!terminator) { consume_one(); }` needs an explicit forward-progress check
  (`pos == before → throw`), not just trust that every sub-parser advances — cheap insurance
  against the exact non-terminating-fallback class of bug in item 1; (b) a local budget/clamp
  computed against one field (here, a `max_seq_len` parameter) must be written back to every
  struct field other code trusts as ground truth for that same quantity — a clamp that only
  narrows the *allocation* and not the *indexing arithmetic* is worse than no clamp, because it
  turns a would-be large-but-valid allocation into a guaranteed out-of-bounds write; (c) for a
  novel from-scratch architecture port, matching the reference math/control-flow via careful code
  reading is necessary but *not sufficient* — GGUF tensor shapes are declared in `ne[]` order
  (`ne0` = fastest/contiguous), and an easy-to-miss class of bug is indexing a *correctly-shaped*
  tensor with the *wrong axis as the fast dimension*; the only way this was actually caught was
  building a second, independent reference implementation and running the identical input through
  both, then, once a divergence was proven real, checking every weight tensor's actual `ne[]`
  order (via a small `gguf_dump.py`) against every place this code indexes into it — do this
  systematically for *every* new tensor a novel architecture introduces, not just the ones that
  happen to look suspicious on a first pass.

### 2026-07-16 — --server mode's Ctrl+C never ran cleanup; fixing it exposed 2 more real bugs
- Summary: `--server` mode's shutdown path (`api_server_stop()` and all of `main()`'s later
  cleanup — `tokenizer_free`, `gguf_header_free`, `mapped_file_close`, etc.) had been dead code
  since forever: `main.c` called a bare `pause()` with no signal handler installed, so
  SIGINT/SIGTERM used the default disposition (immediate process termination) — `pause()` never
  actually returns under default disposition. Fixed by installing a `sigaction` handler
  (SIGINT + SIGTERM) that sets a `volatile sig_atomic_t` flag, and replacing the bare `pause()`
  with `while (!g_shutdown_requested) pause();`, scoped to only the `--server` block so
  REPL/one-shot Ctrl+C behavior (immediate kill) is unchanged. Verified: server now exits 0 and
  prints "Shutting down..." on Ctrl+C, under gcc release, gcc debug (ASan/UBSan), and clang debug
  (ASan/UBSan) alike.
- **Making that dead code reachable for the first time immediately exposed two more real,
  previously-latent bugs** (this is exactly why "add the missing handler" isn't a one-line
  change — code that's never executed can hide anything):
  1. `api_server_stop()`'s `close(ctx->server_fd)` alone does not reliably wake the listener
     thread's blocking `accept()` call on Linux (POSIX explicitly leaves this unspecified; in
     practice it just hangs). Confirmed directly: sending SIGINT hung the process indefinitely
     (still running per `ps` well past any reasonable timeout) until fixed by calling
     `shutdown(ctx->server_fd, SHUT_RDWR)` *before* `close()` — the standard, reliable way to
     unblock a concurrent `accept()`.
  2. Testing the debug/ASan/UBSan build's server path (never previously done, since there was
     no reachable clean-shutdown path to test through) surfaced a genuine, unrelated UBSan
     finding: `src/tokenizer/tokenizer_gguf.c` read GGUF metadata numeric arrays (`scores`,
     `token_type`) via a raw `(const float *)`/`(const int32_t *)` pointer cast into the mmap'd
     file and direct array indexing — an unaligned load, since GGUF packs metadata byte-tight
     with no alignment guarantee for array start offsets. UBSan: "misaligned address ... requires
     4 byte alignment". Fixed both sites with `memcpy` into a local before use — the exact same
     idiom `gguf_reader.c`'s own `read_u32`/`read_u64` already use for scalar header fields, and
     that `str_cursor_next` already uses for string arrays; only these two numeric-array reads
     had skipped it.
- Affected files/modules: `src/cli/main.c`, `src/api/http_server.c`,
  `src/tokenizer/tokenizer_gguf.c`.
- Detection: (1) manual `kill -INT` test against a running `--server` instance — exit code and
  log output. (2) direct observation via `ps` that the process hung after SIGINT despite the
  signal handler firing. (3) UBSan output during a debug-build server test.
- Correction: see summary above for each of the three fixes.
- Prevention rule: **exercising a previously-dead code path for the first time is a real test,
  not a formality** — budget for finding and fixing whatever it turns up, not just confirming the
  originally-requested change compiles. This is a direct instance of the bug-fix policy: don't
  stop at the first fix found: keep testing until the whole newly-reachable path is clean.

### 2026-07-16 — Tokenizer + GGUF header cleanup skipped on the common code path (real leaks)
- Summary: a ~1.2MB ASan leak (49,186 allocations) appeared during Phase 22.5 verification.
  Initially, incorrectly, treated as "pre-existing and unrelated to this change" and merely
  documented rather than fixed — corrected after the user pushed back that pre-existing bugs
  found during any task must be fixed, not just logged (see decision-log.md, same date). Tracing
  it surfaced two distinct real bugs, both present since long before this session:
  1. `src/cli/main.c`'s final cleanup called `tokenizer_free(&t)` only `if
     (args.tokenizer_path)` — but `args.tokenizer_path` is only set when an external
     `--tokenizer <file>` is passed. The far more common path, `tokenizer_load_from_gguf()`
     auto-loading the tokenizer straight from the model's own GGUF metadata (used by every run
     in this session with no `--tokenizer` flag), populated the exact same `Tokenizer` fields
     but the guard skipped freeing them — leaking the entire vocab (49,152 strings), scores,
     sorted index, chat template, and special-token list on every single invocation.
  2. `GGUFHeader` (`src/core/gguf_reader.c`) heap-allocates a NUL-terminated copy for every
     `GGUF_VAL_STRING` metadata entry (`parse_meta_entry`, since on-disk GGUF strings aren't
     NUL-terminated) — but no free function for `GGUFHeader` existed anywhere in the codebase,
     so every string-typed metadata value (11 of them for this model) leaked too.
- Root cause (both): cleanup code was written for one loading path and not audited against the
  other; a data structure gained a heap-allocating field without a matching destructor ever
  being added.
- Affected files/modules: `src/cli/main.c`, `src/core/gguf_reader.c`,
  `include/core/gguf_reader.h`.
- Detection: `make debug` (ASan/UBSan) on a real model load — LeakSanitizer's exit-time report;
  confirmed both were present on the prior commit too (identical byte/allocation counts with and
  without the Phase 22.5 changes), so genuinely pre-existing, not introduced by this session.
- Correction: (1) call `tokenizer_free(&t)` unconditionally in the cleanup section — safe
  because `t` is `memset` to zero before either load path, and `tokenizer_free` already no-ops
  cleanly on a zeroed/already-freed struct (`free(NULL)`, `vocab_size == 0` loop). (2) added
  `gguf_header_free(GGUFHeader *hdr)` (frees only the `GGUF_VAL_STRING` entries' heap copies;
  array/tensor data are zero-copy mmap pointers, untouched), called from `main.c`'s cleanup path
  and both early-return GGUF-parse-failure branches.
- Prevention rule: when a struct gains a heap-allocating field, its free function must be
  added/updated in the same change, and audited against **every** code path that populates the
  struct, not just the one path being actively tested. See the new "Bug-fix policy" in
  `engineering-rules.md`: a leak found while working on something else still gets fixed, not
  just logged as a known issue.

### 2026-07-16 — Banner glyph separator rendered as a solid block, not a gap
- Summary: the new CLI startup banner (`src/cli/banner.c`, "PROJECT ZERO" block-font splash)
  rendered as an unreadable wall of `#` characters instead of legible letters when first
  captured via a raw `script(1)` terminal session.
- Root cause: `compose_banner()` joined each letter's glyph rows with a literal `' '` (space)
  separator character, but `print_glyph_row()` only treats the glyph data's `'.'` character as
  blank — any other character, including that literal space, is rendered as `'#'`. Every
  inter-letter gap therefore printed as filled instead of blank.
- Affected files/modules: `src/cli/banner.c` (`compose_banner`, `print_glyph_row`).
- Detection: manual raw `script(1)` capture + hand-tracing the composed row string against the
  expected per-letter glyph layout (not caught by compilation or any automated test, since the
  file has no dedicated unit test — this is terminal-rendering-only code, same class as
  `live_stats.c`).
- Correction: changed the separator from `' '` to `'.'` so it's treated as blank consistently
  with the rest of the glyph data.
- Prevention rule: when a rendering function has a single "this character means blank" rule,
  every code path that builds input for it (including separators/padding, not just the "real"
  content) must emit that same blank sentinel — never a different character that happens to look
  blank in source but isn't recognized by the renderer.

### 2026-07-15 — Live tok/s indicator overwrote the streamed response text
- Summary: the REPL's live tok/s status line (`\r` + overwrite + `\x1b[K`) clobbered the actual
  response text every single token, because both were written to the same terminal row with no
  forced newline between them — a screenshot taken for design QA showed the response reduced to
  a couple of stray characters, with the stats line sitting where the reply should have been.
- Root cause: `\r` moves the cursor to the start of whatever line it currently occupies; since
  streamed response tokens don't force their own line, the live-stats update and the response
  text shared a cursor position, so each stats redraw erased the response printed so far on that
  row.
- Affected files/modules: `src/cli/live_stats.c` (`tn_live_stats_render`).
- Detection: a real terminal screenshot taken for the Phase 22.4 design-QA pass — not caught by
  `tests/test_progress.c`/`test_color.c`/`test_md_render.c`, which test formatting logic in
  isolation, not interaction between two features writing to the same terminal row. A second,
  smaller mistake compounded this while fixing it: the first attempt used `"\x1b7"` as a single
  string-literal escape, which C parses greedily as one (invalid, out-of-range) hex escape
  because `7` is a valid hex digit — had to split it into `"\x1b" "7"`.
- Correction: save the cursor (`\x1b7`), jump to the terminal's last row (`\x1b[999;1H` —
  terminals clamp an out-of-range row to their actual height, no need to query real size), print
  the stats there, restore the cursor (`\x1b8`) — the standard "status line" technique, and it
  never touches the response's own cursor position.
- Prevention rule: two features that both write control sequences to the same stream/terminal
  must be verified together with a real render (screenshot or manual terminal check), not just
  unit-tested independently — formatting-logic tests cannot catch cross-feature terminal-state
  interaction. Also: never write a bare `\xNN` immediately followed by another hex digit in a C
  string literal — split into separate literal segments.

### 2026-07-15 — "Untested socket layer" hid four real HTTP protocol bugs
- Summary: `docs/ai/project-overview.md` had long flagged the Phase 21 HTTP server's "socket
  layer" as untested/not-in-CI — taken at face value rather than verified. Doing the first real
  end-to-end `curl` testing (Phase 22 API hardening) surfaced four separate, real bugs, none of
  which any existing test caught because `tests/test_api_server.c` only exercises pure-logic
  units (JSON parse, chat compile, SSE format) via pipes — never a real socket:
  1. `handle_connection`'s receive loop never broke out when a request had no `Content-Length`
     (the normal case for `GET`/`HEAD`/`OPTIONS`) — it kept calling `recv()` forever waiting for
     a body that would never arrive, hanging indefinitely on every plain `GET /health`.
  2. The non-streaming chat-completion path sent `Content-Length: 0` (from a `NULL` body) and
     only *then* wrote the real JSON via a second `write()` — compliant HTTP clients stop reading
     at 0 bytes and never see the actual response.
  3. Streaming responses claimed `Transfer-Encoding: chunked` but the SSE writer emitted raw,
     unframed bytes (no hex chunk-size prefixes) — a real protocol violation that curl tolerated
     by accident but a strict client (browser `fetch`/`EventSource`) would not.
  4. The server never called `signal(SIGPIPE, SIG_IGN)`, so any client disconnecting mid-response
     delivered `SIGPIPE` to `write()`, whose default disposition kills the *entire* process, not
     just that connection — trivially triggered by any short `curl --max-time`.
- Root cause: all four are consequences of never having driven the server with a real HTTP
  client. Logic-level pipe tests validate formatting functions in isolation but cannot catch bugs
  in the receive-loop state machine, header/body sequencing, or process-wide signal disposition.
- Affected files/modules: `src/api/http_server.c` (recv loop, `send_response_ex`, `SIGPIPE`),
  `src/api/sse_stream.c`/`include/api/sse_stream.h` (new `sse_format_full_response()` so the
  caller can compute `Content-Length` before sending headers).
- Detection: manual `curl`/`curl --raw` end-to-end smoke testing against a real running server
  with a real model — not caught by `make test`, ASan, UBSan, or code review of the diff alone.
- Correction: stop reading once headers are complete and no `Content-Length` is present; build
  the JSON body before sending headers so `Content-Length` is correct; drop the false
  `Transfer-Encoding: chunked` claim (the response already closes the connection, so EOF
  delimits the body per RFC 7230 §3.3.3 case 7); ignore `SIGPIPE` at server start.
- Prevention rule: a component documented as "untested" is not verified — before extending or
  hardening it, exercise it for real (a live server + `curl`/`curl --raw`, not just unit tests)
  rather than trusting the label. Any new HTTP/socket work in this codebase must include a real
  end-to-end request/response check, not only logic-level tests.

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
