# RCA — Ternary-Bonsai-27B throughput drop: ~2.74 → ~1 tok/s (2026-07-16/17)

**Verdict: not a code regression.** The drop is a change in the underlying (virtualized) host's
performance profile, which happened to coincide in time with the classifier-compatibility work.
The exact commit that produced 2.74 tok/s, rebuilt and rerun unmodified on the current host,
measures ~1.0 tok/s — the same as current HEAD. A pre-VNNI control commit still measures ~13x
slower than both, proving the test methodology detects real code-driven gaps when they exist.

## 1. Symptom timeline

All runs: same `Ternary-Bonsai-27B-Q2_0.gguf` (7.16 GB), same prompt, `--max-tokens 60
--temperature 0 --threads 4`, greedy decoding. "Host profile" is the engine's own auto-detected
Hardware Profile printed at startup (visible in every screenshot).

| When | Build | Host profile (L2/core · L3 · DRAM bw · ceiling) | Result |
|---|---|---|---|
| 07-16 17:40 | `ce8e90d-dirty` (VNNI Q2_0 kernel) | 2048 KiB · **260 MiB** · **16.0 GB/s** · 17.1 tok/s | **2.74 tok/s** (t=4 peak of the 0.86/1.62/2.31/2.74 thread sweep) |
| 07-16 evening | host recycled mid-session (`/proc/uptime` ≈ 5 min) | 1024 KiB · **33 MiB** · 9.9–12.3 GB/s | classifier sweeps land at ~1.07–1.19 tok/s |
| 07-17 11:08 | HEAD `85d3b36` | 1024 KiB · 33 MiB · 12.2 GB/s · 10.4 tok/s | 0.59 tok/s (`classifier_auto.png`) |
| 07-17 11:51 | **`ce8e90d` rebuilt from that exact commit** | 1024 KiB · 33 MiB · 10.9 GB/s · 9.4 tok/s | **1.02 tok/s** (1.40 on an earlier same-day rerun) |
| 07-17 same session | `34d3ac9` (control: parent of `ce8e90d`, pre-VNNI kernel) | same host | **0.08 tok/s** (0.12 earlier) — the expected ~13x code gap still shows |
| 07-17 same session | HEAD, third leg | same host | 0.95 tok/s (1.08 earlier) |

Screenshots (primary evidence, hardware box + `[gen]` line embedded in each):
`benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/commit_bisect_{ce8e90d_1.02toks,34d3ac9_0.08toks,HEAD_0.95toks}.png`.
The original 2.74 capture with the host-A hardware box is recoverable from git history:
`git show 591333d:benchmark_results/qwen35_ternary_bonsai_2026-07-16/screenshots/pz_t4_peak.png`
(the in-tree copy was later recaptured on host B to fix display bugs and shows 1.07).

Note: the sweep between the two states was a **thread** sweep (t=1..4 = `--threads`), not a
temperature sweep — `--temperature 0` was constant throughout.

## 2. Suspects examined and ruled out

Every commit between the 2.74 measurement (`ce8e90d`) and HEAD was audited
(`git diff ce8e90d..HEAD -- src/ include/`: 16 files). The only ones touching engine code:

1. **`85d3b36` classifier materialization (the prime suspect — the drop was noticed right after
   this work).** `forward.c`'s LM-head dispatch gained a branch, but it is guarded by
   `hp->classifier_explicit`, which is set **only** by `tn_hardware_profile_set_classifier()`
   when `--classifier` is passed on the CLI. Without the flag, execution falls through to the
   byte-identical `parallel_matmul_q2_0(...)` call as before, plus one `tn_hardware_profile_get()`
   and two predictable branches **per token** (not per row) — nanoseconds against a ~1 s token.
   The loader-side materialization in `gguf_loader.c` is behind the same flag: default loads
   allocate nothing extra.
2. **`921e223` VBMI SIGILL fix (runtime dispatch instead of `#if`).** `bitunpack2_vnni.h`'s
   unpack now branches on a cached per-TU `static int` (one load+compare per 64 weights,
   fully predicted) and `matmul_i4_task` hoists the check once per task. Cannot cost 2.7x on a
   bandwidth-bound kernel.
3. **`dd295e5` detokenizer/batch-path fixes.** The Q2_0 batch path has no caller for dense
   models; the detokenizer is per-emitted-token string handling, off the matmul hot path.

Ruling out by diff was then made falsifiable by the commit bisection above: even if some audit
argument were wrong, **the pre-fix binary itself runs at ~1 tok/s today**. Conversely, the
`34d3ac9` control run proves the harness reliably resolves a genuine code-level performance gap
on this same host, so the absence of a `ce8e90d`-vs-HEAD gap is signal, not insensitivity.

Two real bugs *were* found during the classifier work — the `--classifier` no-op on Q2_0-native
models and the CPUID-advertised-but-faulting AVX-512VBMI SIGILL — but both affected only
explicit `--classifier` runs, not the default path that regressed from 2.74 to ~1.

## 3. Root cause

The sessions run on a Firecracker microVM whose **underlying physical host changed and remains
unstable**:

- **Different hardware between the two sweeps**: the engine's own profiler recorded L2
  2048 → 1024 KiB/core, L3 **260 → 33 MiB**, measured DRAM bandwidth 16.0 → 9.9–12.3 GB/s.
  A ~27B dense model is bandwidth-bound, so the ~30% bandwidth drop directly moves the
  ceiling. (The profiler's printed figures — "1149 MB/token", ceilings of 9.4–17.1 tok/s —
  were later found to be computed from hardcoded wrong constants; see the §5 addendum for
  the corrected ceilings. The *relative* bandwidth drop between hosts stands regardless.)
- **Co-tenant contention on the new host**: effective weight-streaming rate fell more than
  bandwidth alone explains (~18.6 GB/s on host A vs 7–8.4 GB/s on host B — a ~55–60% drop
  against a ~30% calibrated-bandwidth drop), and re-runs stalled minutes in first-touch of
  fresh memory (`mmap(MAP_POPULATE)` of the model, KV-cache `calloc`) while guest-side memory
  stats stayed healthy — the signature of hypervisor-level memory pressure invisible to the
  guest. Direct live evidence of the noise floor: within single calibration runs minutes apart,
  the same T=4 microbench measured 3.5 tok/s and 35.6 tok/s.
- Even the VBMI-faults-on-execute behavior appears host-state-dependent: `ce8e90d` (which
  executes VBMI unconditionally) crashed with SIGILL on the 07-16-evening host but completed
  calibration cleanly during the 07-17 bisection — consistent with continued host churn.

Full evidence chain: `docs/ai/mistakes.md` (2026-07-17 entries) and `docs/ai/decision-log.md`.

## 4. Solution

**Restore.** There is nothing to revert or patch: the current binary already produces
~2.7 tok/s-class results whenever it runs on host-A-class hardware. To confirm which class a
session landed on, read the startup Hardware Profile box (L3 size and measured DRAM bandwidth
are an effective host fingerprint) before trusting any absolute number. On this environment:

- Treat cross-session absolute tok/s as non-comparable; only same-session, back-to-back A/B
  sweeps are valid (already codified in `mistakes.md` and the README caveat).
- For regression checks, always include a known-slow control leg (as the bisection did) so
  "no gap" is distinguishable from "test can't see gaps."

**Improve (available today, no code change).** On the one clean post-fix same-session sweep,
`--classifier int8`/`int4` measured **2.60/2.62 tok/s vs 1.19 for the zero-copy default** —
the materialized INT8/INT4 LM head avoids the Q2_0 per-block unpack entirely and the VNNI
int8/int4 kernels are more compute-efficient per element for the 248320×5120 head, at the cost
of 0.6–1.2 GB extra RAM. Caveat: a later contended-host sweep did not reproduce the magnitude
or ordering, so verify with a same-session A/B on the target host before relying on it.

**Improve (code-side follow-ups — require a stable host to A/B first, per engineering rules).**

1. Profile the zero-copy Q2_0 LM head (`n=5120, d=248320` — confirmed to take the VNNI path,
   `TN_Q2_0V_MAX_N=20480`). The int8-vs-Q2_0 result implies the per-64-element unpack
   (~15-instruction SSE variant whenever verified VBMI is unavailable) dominates on weak cores;
   options include a row-blocked unpack-once-reuse scheme or auto-selecting the materialized
   INT8 head when `tn_get_free_ram()` shows headroom (would need a deliberate reversal of the
   opt-in-only decision in `decision-log.md` 2026-07-17).
2. Kernel efficiency headroom — see the addendum below for the corrected size of this
   opportunity (~1.5–2x on host-B-class hardware, not the 6–16x the profiler's ceiling
   implied) and the specific `dot_q2_0_row_vnni` defects to fix.

## 5. Addendum (2026-07-17, same day): the "ceiling" itself was wrong — corrected targets

A follow-up question ("what would it take to reach the ceiling?") exposed that the profiler's
ceiling was fiction: `hardware_profile.c` hardcoded BitNet-2B's geometry (Data/token
"1149 MB") for **every** model. A dense 27B Q2_0 model really streams ~6.8 GB of weights per
token (7.16 GB minus the embedding), so the honest bandwidth ceilings are:

**Second correction (same day, later):** the "calibrated BW" itself was then found to be
~3x under-measured — `probe_dram_bandwidth()` divided one pass worth of bytes by three
passes worth of wall time, on top of a serialized volatile read loop (both fixed; same host
re-measured 12.0 → **41.2 GB/s**). The table below shows both stages of correction:

| Host | Probe-reported BW | True BW (post-fix) | Real ceiling (~6.5 GiB/tok) | Measured | True utilization |
|---|---|---|---|---|---|
| A (2.74 tok/s era) | 16.0 GB/s | ~48 GB/s (inferred, ×3) | ~7 tok/s | 2.74 | ~40% |
| B (current class) | 10.9–12.2 GB/s | 41.2 GB/s (measured) | **6.0 tok/s** | 0.95–1.24 then; **3.56 after the kernel fix below** | 20% → **59%** |

Consequences, replacing the earlier "6–16% of ceiling" framing in this report (and the
first addendum draft's "host A was at the bandwidth wall" claim — it was at ~40% of *true*
bandwidth; it only looked wall-limited against the broken probe number):

- The kernel-side headroom was real and larger than the first estimate. Implemented on this
  branch: the `dot_q2_0_row_vnni` restructure below, verified end-to-end at **2.74/2.80 →
  3.56/3.54 tok/s (+28%, interleaved same-session A/B, token-identical output)** — the
  optimized kernel on a host-B-class VM now beats the original host-A 2.74 headline.
- The two defects behind that +28% (both now fixed): a full `_mm512_reduce_add_epi32`
  horizontal reduction **per 128-element block** — replaced by a per-row float vector
  accumulator with one reduce per row — and a branchy scalar fp16→f32 scale conversion per
  block, replaced by F16C. Remaining headroom to the honest 6.0 ceiling: the
  ~15-instruction SSE 2-bit unpack on hosts without working VBMI (this host has no VBMI at
  all — a wider AVX-512BW unpack is the specified follow-up), multi-row blocking, and the
  uninstrumented attention/DeltaNet stretch of the per-token time.
- Going **above** the ~6 tok/s single-stream ceiling on this hardware is not a kernel
  problem: at 2.125 bits/weight there is no smaller format to stream, so the only levers are
  faster memory (hardware), or amortizing the weight stream across multiple sequences/tokens
  per pass (batched or speculative decoding — an architectural feature, not an optimization).

The profiler bug is fixed in this branch (`tn_hardware_profile_set_model_bytes()`, called
post-load from `main.c`; startup box now labeled "(pre-load est.)"), verified end-to-end via
`make demo` (SmolLM2 now reports its true 258 MB/token instead of 1149). The probe fix,
kernel optimization, and their verification records live in
`docs/architecture/CEILING_CALCULATION.md` (the full ceiling-calculation spec). Full
writeups: `docs/ai/mistakes.md` (2026-07-17 entries).
