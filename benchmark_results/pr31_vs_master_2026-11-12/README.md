# PR #31 (final head) vs master on DeepSeek-V2-Lite — 2026-11-12

**Question:** how fast is the final head of [PR #31](https://github.com/shifulegend/project-zero/pull/31)
(`--moe-threading`) today, compared with the commit it branched from and with current master?

**Short answer:** on this 4-vCPU host the PR's final head decodes about **1.6 tok/s** (default mode) to
**2.8 tok/s** (`legacy`) at 4 threads; master decodes **12.9–14.9 tok/s**. The default mode is also
about **40% slower than the commit the PR branched from**. Merging the PR on top of master could
not be tested: the merge conflicts in `src/` and `include/`.

All numbers are tok/s from the `[gen]` line, best of N (N is stated per table).

## What was compared

| Id | Build | Commit |
|---|---|---|
| C1 | master | `fefa37e` fix(api): release the generation slot before the final response is sent (#65) |
| C2 | PR #31 final head, default mode | `f26f36e` fix(core): resolve TN_THREAD_LOCAL macro expansion and restore weights.h header inclusion |
| C3 | PR #31 final head, `--moe-threading legacy` | `f26f36e` |
| C4 | PR #31 base (commit the PR branched from, `git merge-base`) | `bf66526` docs(benchmark): add PR #31 moe-threading verification results |
| C5 | PR #31 final head, `--moe-threading rowsplit` (extra, replaces the merged build) | `f26f36e` |
| – | master + PR #31 merged | **not built: merge not clean** |

- The PR's **default mode is `rowsplit-fused`** (`src/transformer/moe_ffn.c`: `s_moe_threading_mode = TN_MOE_THREADING_ROWSPLIT_FUSED`); modes are `legacy`, `rowsplit`, `rowsplit-fused`.
  C5 runs plain `rowsplit` explicitly. The flag is accepted only as `--moe-threading <mode>` (two words); the `=` form is not parsed.
- **Merge:** `git merge --no-commit pr31` on top of `fefa37e` conflicts in 16 files, including `src/` and `include/`
  (`transcripts/builds/merge_conflicts.txt`). It was aborted, nothing was resolved by hand, and the merged build was not made.
- The PR branch is built on a July master (`bf66526`, `v0.1.0-76`); it does not contain master's later changes
  (for example the 21 Oct dynamic row-chunk scheduling for MoE experts). This compares the PR as submitted with today's master.
  Older trees name the binary `adaptive_ai_engine`, master names it `pz`.

## Setup

- Host: Intel Xeon @ 2.10 GHz (model 207, AVX-512 VNNI), 4 vCPU, 16 GB RAM (about 14.7 GB available), Linux 6.18, gcc. Screenshot `screenshots/common/00_machine_facts.png`.
- Builds: `make release CC=gcc -j2` in separate `git worktree` checkouts (screenshots `02`–`04` in `screenshots/common/`).
- Model: `DeepSeek-V2-Lite-Chat.Q4_K_S.gguf` from `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`, 9,533,608,704 bytes,
  sha256 `40ce415f480cd8edf62bdb27962ee2387a7c8a43c9756ee5c6895e861c079808` (equals the Hugging Face `x-linked-etag`).
- Command: `<binary> --model <gguf> --prompt "<prompt>" --max-tokens 64 --temperature 0 --threads N --simd vnni [--moe-threading <mode>]`.
  Prompt: the one in `benchmark_results/deepseek_h2h_2026-10-29/meta.json` ("Explain what machine learning is and how neural networks work. …").
- Each round runs C1, C2, C3, C4, C5 once, strictly one after another; before a round the 1-minute load had to be below 0.3
  (the round waits up to 3 minutes for that). One discarded warm-up run per build before measuring.
- Screenshots are terminal sessions rendered by `tools/screenshots/cli/capture.mjs`: per round an INPUT, a PROCESSING and an OUTPUT image.
  INPUT and OUTPUT are `capture.mjs` captures. `capture.mjs` can only screenshot after the command exits, so the PROCESSING image
  renders the first part of the same recorded pty stream (cut when C3 starts) through the same `capture.html`/xterm pipeline
  (`tools/snap_prefix.mjs`). The raw pty recordings are `transcripts/*/N*_session.raw` (not `script -t` timing files);
  per-run stdout/stderr are the `.out`/`.err` files.

## Two batches, and why

**Run A (first batch) is confounded and is kept only as a record.** The calibration cache `~/.project-zero/calibration.bin` is shared by all
builds, and master (C1) and the PR default (C2) re-measured it on every run, each time choosing a different backend ("SIMD: scalar / avx2 / vnni256 /
avx512f (calibrated optimal)"); C3–C5 then reused whichever result C2 had saved. So the kernel backend differed between runs and between configs.
Run A: N=4 five rounds, N=2 three rounds, N=1 three rounds, no `--simd` pin.

**Run B (second batch, the valid one) pins `--simd vnni` for every config** and gives each config its own `HOME`, so its calibration cache is measured once in an
untimed warm-up (`transcripts/warmup_runB/`) and not re-measured. Run B is **partial: N=4, 3 rounds only.** It was stopped on the owner's instruction;
the unfinished N=2 round 1 (C1–C3 only) is kept in `transcripts/runB_aborted_N2_r1/` and is not used in any table.

N=2 and N=1 exist only in Run A. The two runs agree closely where both exist (N=4), so the confound did not change the picture, but
N=2 and N=1 are **not** pinned measurements.

## Results — Run B (valid, pinned): N=4 threads, second batch, partial: 3 rounds, best-of-3

| Config | r1 | r2 | r3 | best-of-3 | median |
|---|---|---|---|---|---|
| C1 master `fefa37e` | 14.90 | 13.22 | 12.92 | **14.90** | 13.22 |
| C2 PR31 default (rowsplit-fused) | 1.65 | 1.58 | 1.66 | 1.66 | 1.65 |
| C3 PR31 `legacy` | 2.71 | 2.68 | 2.76 | 2.76 | 2.71 |
| C4 PR31BASE `bf66526` | 2.81 | 2.72 | 2.77 | 2.81 | 2.77 |
| C5 PR31 `rowsplit` (extra) | 2.36 | 2.37 | 2.38 | 2.38 | 2.37 |

## Results — Run A (confounded, no `--simd` pin)

**N=4 threads, best-of-5**

| Config | r1 | r2 | r3 | r4 | r5 | best-of-5 | median |
|---|---|---|---|---|---|---|---|
| C1 master | 15.17 | 11.89 | 12.46 | 12.27 | 14.28 | **15.17** | 12.46 |
| C2 PR31 default | 1.64 | 1.58 | 1.57 | 1.54 | 1.57 | 1.64 | 1.57 |
| C3 PR31 `legacy` | 2.76 | 2.62 | 2.73 | 2.53 | 2.74 | 2.76 | 2.73 |
| C4 PR31BASE | 2.58 | 2.65 | 2.43 | 2.66 | 2.62 | 2.66 | 2.62 |
| C5 PR31 `rowsplit` | 2.28 | 2.34 | 2.22 | 2.22 | 2.26 | 2.34 | 2.26 |

**N=2 threads, best-of-3**

| Config | r1 | r2 | r3 | best-of-3 | median |
|---|---|---|---|---|---|
| C1 master | 8.44 | 8.46 | 8.75 | **8.75** | 8.46 |
| C2 PR31 default | 1.54 | 1.52 | 1.45 | 1.54 | 1.52 |
| C3 PR31 `legacy` | 2.14 | 2.15 | 2.09 | 2.15 | 2.14 |
| C4 PR31BASE | 2.23 | 2.07 | 2.07 | 2.23 | 2.07 |
| C5 PR31 `rowsplit` | 1.77 | 1.81 | 1.91 | 1.91 | 1.81 |

**N=1 thread, best-of-3**

| Config | r1 | r2 | r3 | best-of-3 | median |
|---|---|---|---|---|---|
| C1 master | 4.96 | 4.60 | 5.20 | **5.20** | 4.96 |
| C2 PR31 default | 1.16 | 1.26 | 1.26 | 1.26 | 1.26 |
| C3 PR31 `legacy` | 1.46 | 1.49 | 1.51 | 1.51 | 1.49 |
| C4 PR31BASE | 1.47 | 1.50 | 1.30 | 1.50 | 1.47 |
| C5 PR31 `rowsplit` | 1.41 | 1.39 | 1.37 | 1.41 | 1.39 |

## Comparison at N=4 (best-of-N, percent change)

| Comparison | Run B (pinned, best-of-3) | Run A (best-of-5) | Run-to-run spread (min–max) |
|---|---|---|---|
| PR31 default vs master | −88.9% (1.66 vs 14.90) | −89.2% | master 12.92–14.90 (B), 11.89–15.17 (A); PR31 default 1.58–1.66 |
| PR31 `legacy` vs master | −81.5% | −81.8% | legacy 2.68–2.76 (B) |
| PR31 default vs PR31BASE | −40.9% (1.66 vs 2.81) | −38.3% | base 2.72–2.81 (B), 2.43–2.66 (A) |
| PR31 `legacy` vs PR31BASE | −1.8% (2.76 vs 2.81) | +3.8% | within noise |
| PR31 `rowsplit` vs PR31BASE | −15.3% | −12.0% | rowsplit 2.36–2.38 (B) |
| merged vs master | UNKNOWN (merge not clean) | – | – |

N=2 / N=1 (Run A only): PR31 default vs master −82% / −76%; PR31 default vs PR31BASE −31% / −16%; `legacy` vs PR31BASE −4% / +1%.

## Correctness

Greedy decode (`--temperature 0`), text between the KV-cache line and the expert summary, sha256 (first 12 hex); see
`screenshots/common/11_runB_correctness.png` and `21_runA_correctness.png`.

- Within one config in Run B the text is identical in all 3 rounds. In Run A it is identical across rounds for C1, C2, C3, C5; **C4 (PR31BASE) produced 3 different texts in 5 rounds** (the unpinned run).
- **Across builds the text differs.** Run B, N=4, round 1: master `e71483bcf500`, PR31 default `f6316cf7b43f`, PR31 `legacy` `3a61d0acb1dc`, PR31BASE `05c26b445cd9`, PR31 `rowsplit` `3a61d0acb1dc` (equal to `legacy`).
  First differing character vs master: PR31 default and PR31BASE at char 9, `legacy`/`rowsplit` at char 23 of the first text line.
  Which output is "right" is UNKNOWN: there is no reference output here, and the PR builds also generate 65 tokens where master generates 64.
- All runs exited 0; no crashes. No NaN lines in the outputs (scan in the correctness screenshots).

## Verdict

(a) **Does the PR's final head beat the commit it branched from?** No. Default mode is about 38–41% slower than PR31BASE at 4 threads (about 31% at 2, 16% at 1);
`rowsplit` is about 12–15% slower; `legacy` is within noise of PR31BASE (−1.8% pinned, +3.8% unpinned).

(b) **Does it beat today's master?** No. Every PR31 mode is 81–89% below master at 4 threads (about 5–10x slower).

(c) **Would merging it on top of master help?** UNKNOWN. The merge is not clean (16 conflicting files), so a merged build could not be measured;
this data cannot say how the PR's kernels would behave on top of master's current MoE scheduling.

## Caveats

- Run B is partial (N=4, 3 rounds). N=2 and N=1 come from Run A only (3 rounds each, best-of-3) and are not pinned to one backend.
- Run B is not the originally planned 5/3/3 rounds; the plan was cut for time. Run A's first batch is invalid as a pinned comparison (above).
- One host, one prompt, 64 tokens; best-of-N with N = 3 or 5. Shared cloud VM: run-to-run variation of master is about 13–28% (min–max).
- The warm-up logs of Run A (one discarded round at N=4) were not kept; Run B's warm-up logs are included.
- `--moe-threading=rowsplit` (with `=`) was not tested because the parser rejects that form; C5 uses `--moe-threading rowsplit`.
- No screenshot was withheld; none shows an e-mail address (see the privacy scan in the commit).

## Contents

`screenshots/common/` machine facts, merge attempt, three builds, summary tables and correctness for both runs ·
`screenshots/runB_pinned/` and `screenshots/runA_unpinned/` per round `*_1_input`, `*_2_processing`, `*_3_output` ·
`transcripts/` raw pty recordings and per-run `.out`/`.err` · `tools/` the exact scripts (`round.sh`, `do_round.sh`, `summarize.py`, …) ·
`summary_runA.txt`, `summary_runB.txt` · `MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256` from this folder).
