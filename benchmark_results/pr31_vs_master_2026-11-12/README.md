# PR #31 (final head) vs master on DeepSeek-V2-Lite — 2026-11-12

**Question:** how fast is the final head of [PR #31](https://github.com/shifulegend/project-zero/pull/31)
(`--moe-threading`) today, compared with the commit it branched from and with current master?

**Short answer:** on this 4-vCPU host the PR's final head decodes about **1.6 tok/s** (default mode) to
**2.8 tok/s** (`legacy`) at 4 threads; master decodes **12.9–14.9 tok/s**. The default mode is also
about **40% slower than the commit the PR branched from**. Merging the PR on top of master could
not be tested: the merge conflicts in `src/` and `include/`.

All numbers are tok/s from the `[gen]` line, best of N (N is stated per table).

## Setup

| Id | Build | Commit |
|---|---|---|
| C1 | master | `fefa37e` |
| C2 | PR #31 final head, default mode (`rowsplit-fused`) | `f26f36e` |
| C3 | PR #31 final head, `--moe-threading legacy` | `f26f36e` |
| C4 | PR #31 base (`git merge-base`) | `bf66526` |
| C5 | PR #31 final head, `--moe-threading rowsplit` (extra) | `f26f36e` |
| – | master + PR #31 merged | not built: merge not clean (16 conflicting files, incl. `src/`, `include/`) |

- Host: Intel Xeon @ 2.10 GHz, 4 vCPU (AVX-512 VNNI), 16 GB RAM, Linux 6.18, gcc (`make release CC=gcc`).
- Model: `DeepSeek-V2-Lite-Chat.Q4_K_S.gguf` (mradermacher), 9,533,608,704 bytes, sha256 `40ce415f480cd8edf62bdb27962ee2387a7c8a43c9756ee5c6895e861c079808`.
- Command: `<binary> --model <gguf> --prompt "<prompt from deepseek_h2h_2026-10-29/meta.json>" --max-tokens 64 --temperature 0 --threads N --simd vnni`. The flag is accepted only as `--moe-threading <mode>` (two words).
- Rounds run C1..C5 once each, one after another, host load below 0.3 before each round, one discarded warm-up per build.
- Run A (N=1, 2, 4) was not pinned to one SIMD backend: master and the PR default re-calibrated on every run and picked different backends. Run B (second batch) pins `--simd vnni` with a calibration cache per config; it covers N=4 only (3 rounds) because it was stopped early.

## Results (tok/s, best-of-N and median)

| Config | N=1 best-of-3 (A) | N=2 best-of-3 (A) | N=4 best-of-5 (A) | N=4 best-of-3 (B, pinned) |
|---|---|---|---|---|
| C1 master | 5.20 / 4.96 | 8.75 / 8.46 | **15.17 / 12.46** | **14.90 / 13.22** |
| C2 PR31 default | 1.26 / 1.26 | 1.54 / 1.52 | 1.64 / 1.57 | 1.66 / 1.65 |
| C3 PR31 `legacy` | 1.51 / 1.49 | 2.15 / 2.14 | 2.76 / 2.73 | 2.76 / 2.71 |
| C4 PR31BASE | 1.50 / 1.47 | 2.23 / 2.07 | 2.66 / 2.62 | 2.81 / 2.77 |
| C5 PR31 `rowsplit` | 1.41 / 1.39 | 1.91 / 1.81 | 2.34 / 2.26 | 2.38 / 2.37 |

Each cell is best / median. Every run of every round is in [results/all_runs.md](results/all_runs.md).
At N=4: PR31 default vs master −89% (A and B); PR31 default vs PR31BASE −38% (A) / −41% (B); `legacy` vs PR31BASE +4% (A) / −2% (B).
Master varies 11.9–15.2 across rounds at N=4; PR31 default varies 1.54–1.66.

## Verdict

(a) **Beats the commit it branched from?** No. Default mode is about 38–41% slower at 4 threads; `rowsplit` 12–15% slower; `legacy` is within noise.
(b) **Beats today's master?** No. Every PR31 mode is 81–89% below master at 4 threads.
(c) **Would merging it on top of master help?** UNKNOWN: the merge is not clean, so a merged build was not measured.
The PR branches from a July master, so this compares the PR as submitted with today's master.

## Correctness

Greedy text (N=4) is stable across rounds within a build in Run B, and differs between builds: master `e71483bcf500`, PR31 default `f6316cf7b43f`, `legacy` and `rowsplit` `3a61d0acb1dc`, PR31BASE `05c26b445cd9`.
Which output is right is UNKNOWN (no reference); the PR builds generate 65 tokens, master 64. In Run A, PR31BASE produced 3 different texts in 5 rounds. No crashes and no NaN lines.

## Caveats

- N=1 and N=2 come from Run A only (best-of-3, not SIMD-pinned); N=4 has both runs. One host, one prompt, 64 tokens; the VM is shared.
- The warm-up logs of Run A were not kept; Run B's are in `transcripts/warmup_runB/`. The unfinished N=2 round 1 of Run B is in `transcripts/runB_aborted_N2_r1/` and is not used.
- PROCESSING screenshots render the first part of the same recorded pty stream through `capture.mjs`'s xterm page (`tools/snap_prefix.mjs`), because `capture.mjs` screenshots only after exit.

## Contents

`results/all_runs.md` per-run tables · `screenshots/` machine, builds, merge attempt, correctness, summary and per-round INPUT/PROCESSING/OUTPUT ·
`transcripts/` raw pty recordings and per-run `.out`/`.err` · `tools/` the scripts · `summary_runA.txt`, `summary_runB.txt` · `MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256`).
