# Plan: fix the ternary activation quantiser, add a correctness gate, decide the bitnet.cpp treatment

Follows `2026-11-18-ternary-output-correctness.md` (the RCA).

Status: implemented — F1 quantiser specification and exactness tests (#75), the correctness gate (§ 2, #74/#77) and the faster LM-head kernels (#76). Not implemented, deferred as a larger change: F2 `act-quant` setting, F6 in-engine accuracy probe, `pz --self-check`, `/v1/selfcheck` and web UI controls (§ 6). Today `--classifier int4` is a documented lossy option; on Mamba-130M it gives wrong output (perplexity 84.8 against 12.6), which the benchmark gate reports as INVALID.
Your decisions (2026-11-18) are recorded in § 5 and applied below.

## 1. Fix pz's activation quantisation (separate PR from master)

### 1.1 Why a one-line change is not the fix
The RCA found one wrong instruction, but the defect is that **nothing defined or enforced what
the quantiser must compute**:
1. Three independent copies of the same function (scalar, AVX2, AVX-512; ARM reuses the scalar one),
   each free to choose its own rounding.
2. The only written spec was a wrong comment ("truncation ... same as llama.cpp"; llama.cpp and
   bitnet.cpp both round to nearest).
3. The unit test encodes the wrong behaviour (`tests/test_simd_vnni.c:125`: error `<= scale`, where
   correct rounding gives `<= scale/2`).
4. Tier-vs-tier tests cannot catch a shared bug (all tiers truncated identically).
5. No test compares a layer against an independent implementation, and no model-level quality
   check exists (perplexity) in CI or in the benchmark scripts.

### 1.2 Code audit (all float to integer conversions in pz)
| quantiser | rounding | verdict |
|---|---|---|
| `quantize_i8.c` scalar `:29`, AVX2 `:96,111`, `quantize_i8_avx512.c` `:60,72` | truncate toward zero | **wrong** (feeds ternary VNNI / VNNI256 / ARM dotprod / Q2_0 VNNI / preq path) |
| `q8_0_act.c` (AVX2 and scalar), `matmul_q4k.c:205` (Q8_K), `kv_quant_ops.c` (I8 and I4), `kv_compress.c` | round to nearest | correct |
There is no NEON quantiser (ARM goes through the scalar one), so ARM is wrong too; a NEON version is
part of the fix.

### 1.3 The fix
**F1. One specification, one oracle.** In `include/math/quantize_i8.h`:
`scale = absmax/127`; `q = clamp(rint_nearest_even(x * 127/absmax), -127, 127)`; `absmax == 0` gives
`q = 0` and scale 0. The scalar function is the oracle; every SIMD tier must equal it **bit for bit**
(integer outputs, no tolerance). Rounding: `_mm512_cvtps_epi32` / `_mm256_cvtps_epi32` (hardware
nearest-even), new NEON `vcvtnq_s32_f32`, scalar `nearbyintf` (identical in the default FP mode;
`tn_simd_init` asserts `fegetround() == FE_TONEAREST`). Same convention as the rounding already used in
`q8_0_act.c`. It differs from bitnet.cpp's `roundf` (ties away from zero) only on exact `.5` products.

**F2. Activation numerics are a setting, decided by the system, not hard-coded (your decision).**
There are two legitimate ways to run a ternary matmul: int8 activations (the model's W1.58A8
definition, used by bitnet.cpp and the HF reference, fastest on VNNI) and float activations (slightly
more precise, slower). Today the choice is accidental: the dispatcher's tier decides it, and
`--simd` therefore changes numerics as a side effect. Design:
- New setting `act-quant = auto | int8 | float` (default `auto`), independent of `--simd`.
- `auto` is resolved from the hardware, on first run where it applies, by the existing calibration:
  candidates are timed as today, but a candidate is eligible only if its **in-engine accuracy probe**
  (F6) is within `quality-tolerance` of the float path. The fastest eligible one wins. A path that is
  fast and wrong can never be chosen. The result is cached under the existing calibration key,
  extended with the act-quant mode and the tolerance.
- Users override it in config, CLI and web UI (§ 6). `--no-calibrate` uses the static default for the
  hardware (int8 on VNNI/dotprod, float otherwise) without a probe.
- Every int8 path uses the one quantiser defined in F1; every tier that runs int8 is bit-identical to
  the others (exact integer accumulation), so `tests/golden_regression.sh` stays valid per mode.
  `--simd` selects speed only; the numerics follow `act-quant`.

**F3. Tests that enforce it.**
- `test_quantize_i8_exact`: every tier equals the oracle bit-exactly on random data, exact `.5`
  ties, `+-absmax`, zeros, denormals, all lengths 1..300 (tails), unaligned pointers.
- Replace the `<= scale` test by `<= scale/2 + eps` (the spec), as llama.cpp's
  `test-quantize-fns` bounds round-trip error per type.
- Ternary matmul error bound derived analytically (`<= 0.5 * act_scale * sum|w|` per output) vs a
  float64 reference, for every tier. A truncating quantiser exceeds it; an ad-hoc tolerance would not.
- `tools/check_rounding.py` (same pattern as `check_isa_guards.py`, CI step): fails on `cvtt*` and
  `(int8_t)(int)` inside quantiser files.
- Golden: `tests/golden_regression.sh` with a ternary model where one is available.

**F4. Model-level check** (not in CI, no model there): the harness gate in § 2 (perplexity).
Acceptance for BitNet 2B on the RCA text: PPL within 0.5 % of the float-activation value (31.3) and of
bitnet.cpp-with-fix (31.4); today 34.39.

**F6. In-engine accuracy probe** (what lets pz judge itself without another engine): run a short
fixed built-in text (about 64 tokens, a few seconds) through the candidate path and through the
float path of the same engine, compare logits (relative L2 error, top-1 agreement, KL). Used by
calibration (F2), by `pz --self-check`, and surfaced in the API and web UI (§ 6). Would have flagged the
truncating quantiser (3.7 % vs 0.4 % layer error) on day one.

**F5. Rest of the PR.** `make release/test/debug` green on gcc and clang; ASan/UBSan; native-vs-portable
A/B and tier A/B for speed (cvtps costs the same as cvttps; the NEON path is new work, so ARM CI must
build); `docs/ai/mistakes.md`, `decision-log.md`, `change-trace.md`; CHANGELOG `Fixed` with the
measured perplexity change; README numbers re-read; golden outputs that move are regenerated and
listed in the PR.

### 1.4 Related items found, tracked separately (not folded into this PR)
- SmolLM2-1.7B Q4_K_M: pz PPL 5.9221 vs llama.cpp 5.9495 (0.46 %, pz lower). Plan: per-tensor logit
  comparison to find which path differs before calling it benign.
- Embedding is held as BF16 (about 0.2 % error vs the F16 file). Plan: measure its PPL effect first.
- This GGUF has no `tokenizer.ggml.pre`; pz and bitnet.cpp both fall back to `default`, which splits
  `real-world` differently from llama3. Plan: infer `llama3` for a gpt2-type vocab with 128256 tokens
  and BOS 128000, next to the existing BitNet template guard.

## 2. Correctness gate in the benchmark framework

### 2.1 Principle (from MLPerf, see § 3)
A speed number is valid only if the same binary, flags and model also pass an accuracy check. The gate
runs before timing and its verdict is attached to every result.

### 2.2 Design
- **Check 1, quality equivalence**: perplexity of each engine on a fixed text (the llama-perplexity
  method that pz, llama.cpp, bitnet.cpp and ik_llama.cpp all expose) must be within tolerance of the
  reference engine for that model. Default tolerance 1 % (configurable; today's measured honest gaps:
  0.006 %, 0.11 %, 0.46 %; broken: 9.7 % and 55 %).
- **Check 2, behavioural sanity** (cheap smoke, runs on every cell): a short greedy factual prompt must
  contain the expected answer ("Paris"), and the generated text must not be a degenerate loop
  (compression ratio over the tail). Catches wrong-function outputs even if no perplexity tool exists
  for an engine (ik_llama.cpp has none built).
- **Check 3, divergence handling** (vLLM `check_logprobs_close` idea): greedy outputs may diverge at
  near-ties; when they do, report the divergence position and require the chosen token to be inside the
  other engine's top-k when logprobs are available; otherwise rely on check 1.
- **Reference per model**: stored in `models.json` (`gate.reference`, `gate.text`, `gate.tolerance`,
  `gate.expect`), validated once against an independent implementation (numpy layer oracle for BitNet).
- **Verdicts**: PASS, WARN (within 2x tolerance), INVALID (fails). INVALID cells are still run and
  logged, but are marked in `summary.csv` (`valid` column), struck from `summary.md`, drawn hollow or
  omitted in the chart, and stamped in the screenshot caption.

### 2.3 Where it applies (scenario by interface)
| scenario / script | gate use |
|---|---|
| `h2h_bench.py` (all engines, models, threads) | per (engine, model) before timing; verdict on every row, chart, caption |
| `build_compare.py` (native vs portable, any two binaries) | PPL of each binary must match: catches numeric regressions between builds |
| `calibration_test.py` (forced tiers, classifiers) | every forced tier / classifier must keep PPL within tolerance; a fast-but-wrong tier fails |
| `setup_engines.py` (fetch and build) | post-build smoke (check 2) so a broken engine build is rejected at install time |
| `hardware_profile.py` | n/a (no model output) |
Flags common to all scripts: `--gate off|warn|strict` (default `strict`; `--preset published` forces
`strict`), `--gate-text FILE`, `--gate-tolerance PCT` (default read from pz's `quality-tolerance`, 1 %),
`--gate-reference ENGINE`, `--gate-only`. Result files: `gate.json` per run folder, `valid` column in
`summary.csv`, gate section in `summary.md`, `meta.json`.
pz's own interfaces (your decision) are in § 6.

### 2.4 Tests
Unit tests with synthetic perplexity outputs and synthetic degenerate text (INVALID/WARN/PASS
boundaries, missing-tool fallback), a dry-run test that the flags exist on every script, a test that
INVALID rows never reach the chart.

## 3. How leading tools handle correctness of what they measure (code and rules reviewed)
| tool | what it does | source read |
|---|---|---|
| **MLPerf Inference** | A result counts only if accuracy meets a threshold of the reference (99 % of FP16/FP32, some 99.9 %); accuracy is a separate run but "the same code must be run for both the accuracy and performance modes"; one validation run per performance result; tokens-per-sample must be 90-110 % of the reference (guards against degenerate output); output truncation to gain speed is forbidden. A result that fails accuracy is not a valid result. | `mlcommons/inference_policies` `inference_rules.adoc` |
| **vLLM** | Engine output is compared with HF Transformers; greedy text may differ, but at a divergence each side's chosen token must be in the other's top-k logprobs (`check_logprobs_close`, `near_tie_tol`). | `vllm/tests/models/utils.py` |
| **llama.cpp** | Per-backend op tests against the CPU reference (`test-backend-ops`, NMSE <= 1e-7); SIMD quantisers against the reference C (`test-quantize-fns`, RMSE <= 1e-4) plus round-trip and dot-product error bounds per type; `llama-perplexity` with a KL-divergence mode (mean KLD, RMS dp, same-top-p) for quality; `ci/run.sh` runs perplexity per quantisation but only logs it. | `tests/test-backend-ops.cpp`, `tests/test-quantize-fns.cpp`, `tools/perplexity`, `ci/run.sh` |
| **bitnet.cpp** | `utils/e2e_benchmark.py` runs `llama-bench` (speed only); `utils/test_perplexity.py` only reports numbers, "failed" means the process failed. No output check on the benchmark path, which is how a wrong FFN activation shipped. | `utils/e2e_benchmark.py`, `utils/test_perplexity.py` |
| **ik_llama.cpp** | inherits the llama.cpp op tests; its BitNet graph uses squared-ReLU (`src/graphs/build_bitnet.cpp:234`). | source |
Upstream status of the bitnet.cpp finding (read on 2026-11-18; merge states of the PRs not confirmed): already reported as
microsoft/BitNet issue #602 ("runs FFN with SiLU instead of relu2", open since 2026-08-08, same file and line) and #588
(perplexity 99.8 vs 17.1); fix PRs #603 (converters write `hidden_activation=relu2`, closed) and #605 (config-based FFN
activation). `main` is still the commit measured here (`0b341e5`). The model's own config (`hidden_act: relu2`) and card
("squared ReLU (ReLU²) activation in FFN layers") confirm squared-ReLU is correct. A third-party report that blamed the
tokenizer warning is not the cause (measured in the RCA). No new report is needed; a comment with the measurements here
could still help.

### 3.1 What that implies for the bitnet.cpp rows (your decision)
- **A. MLPerf rule**: an engine that fails the accuracy gate has no valid speed result. Report
  "bitnet.cpp at commit 0b341e5: fails accuracy (computes SiLU instead of squared-ReLU)", no tok/s in
  the comparison.
- **B. MLPerf "open division" analogue**: a modified engine may be reported if clearly labelled.
  Report "bitnet.cpp + 1-line FFN fix (not upstream)" as a separate row with its own gate result.
- **C. Both**: A as the primary row plus B in a separate, labelled "modified" row.
Recommendation: **C**, and report the bug upstream. A alone loses a fair speed comparison; B alone hides
that the released engine is wrong. The patched build needs a reproducible patch recorded in
`engines.json` (`patches`), applied by `setup_engines.py`, never by hand.

## 4. Order of work (each step ends at a green checkpoint)
1. Gate framework in the scripts (this PR #74): § 2; re-run the published sweep with it. bitnet.cpp and
   ik_llama.cpp BitNet cells get their verdicts; sweep results are not merged before this.
2. pz fix PR (separate, from master): § 1 (F1 quantiser + tests, F2 `act-quant` setting, F6 probe) and
   § 6 (settings, CLI, API, web UI, installer, docs). Re-run the BitNet cells after it merges.
3. Tracked items from § 1.4, one at a time.

## 5. Decisions (2026-11-18)
1. Gate scope: benchmark scripts **and** pz's own interfaces (CLI, API, web UI).
2. Tier numerics: nothing hard-coded; the system decides from the hardware on first run where
   applicable (calibration, accuracy-constrained), and users can change it in config, CLI and web UI.
   My reading of this answer is F2; tell me if it is wrong.
3. Gate tolerance: default 1 %, configurable in UI and CLI.
4. bitnet.cpp rows: option C, both rows, clearly labelled (released = INVALID; patched = modified),
   and the bug reported upstream.
"Financial addition" was read as "functional addition".

## 6. Settings and interfaces added by the fix and the gate (project DoD #6 and #7)
| setting | values | default | CLI | `pz config` key | web UI | API |
|---|---|---|---|---|---|---|
| `act-quant` | auto, int8, float | auto | `--act-quant` | `act-quant` (group hardware) | Settings: hardware | `/v1/config` field; read-only `runtime.act_quant` (effective mode) |
| `quality-tolerance` | percent, 0.1 to 10 | 1.0 | `--quality-tolerance` | `quality-tolerance` (new group quality) | Settings: quality | `/v1/config` field |
| self-check | action | none | `pz --self-check <model>` (exit code 0/1, prints the F6 probe) | n/a | button plus last result badge in the model panel | `POST /v1/selfcheck`, last result in `/v1/config` `runtime.quality` |
- Startup: when calibration runs, the probe result is stored with the calibration and printed in the
  startup summary ("act-quant int8 (probe 0.4 % <= 1.0 %)"); a model whose chosen path exceeds the
  tolerance prints a warning and falls back to the float path (`strict`) or just warns (`warn`).
- Where: `src/config/pz_config.c` (`g_schema`), `src/cli/args.c`, `include/cli/args.h`,
  `src/api/http_server.c` and `api_info.c`, `webui/src/lib/ModelInfoPanel.svelte` and settings view,
  `src/core/calibration.c` (cache key and eligibility), installer defaults in `install.sh` if the
  schema lists settings, README flag table, `docs/WEBUI_GUIDE.md`, CHANGELOG `Added`.
- Tests: schema round-trip, CLI parse, API field and endpoint, calibration eligibility (a synthetic
  fast-but-wrong path is rejected), web UI build check.
