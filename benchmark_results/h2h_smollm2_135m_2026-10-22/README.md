# SmolLM2-135M head-to-head: Project Zero vs llama.cpp — 2026-10-22

- Sweep (best of 3, the README's numbers): 2026-10-22 09:06:41–09:12:06 UTC, `tools/bench_sweep_seq.py`;
  raw logs `smollm2_135m_pz/`, `smollm2_135m_llamacpp/`, `summary.csv`, `meta.json`.
- Screenshots: real terminal sessions of the same commands captured with
  `tools/screenshots/cli/capture.mjs`, 2026-10-22 17:36–17:38 UTC (single runs, so their tok/s
  differ from the sweep: this host's speed drifts during the day).
- Host: Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 VNNI. Model: SmolLM2-135M-Instruct F16, 300
  tokens, temperature 0, `--raw-prompt --ignore-eos` / `--ignore-eos`.

| threads | PZ best of 3 (09:06–09:12) | llama.cpp best of 3 | PZ captured (17:36–17:38) | llama.cpp captured |
|---|---|---|---|---|
| 1 | 28.98 | 25.75 | 29.68 | 25.63 |
| 2 | 57.08 | 48.52 | 60.23 | 46.52 |
| 3 | 80.16 | 76.41 | 76.05 | 65.50 |
| 4 | 85.76 | 88.85 | 104.09 | 87.92 |
