# All runs — PR #31 vs master, DeepSeek-V2-Lite-Chat Q4_K_S, 2026-11-12

tok/s from the `[gen]` line, 64 tokens, greedy. Run A: no `--simd` pin (confounded, see README). Run B: `--simd vnni` pinned, N=4 only, second batch, partial (3 rounds). Raw logs: `../transcripts/`.

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
