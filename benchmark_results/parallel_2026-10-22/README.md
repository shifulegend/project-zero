# Parallel serving (`--parallel N`) — 2026-10-22

Phase 23 stage 2b. Aggregate throughput of `pz --server` when C clients send a chat request at the
same moment (temperature 0, `ignore_eos`, 128 completion tokens each, distinct prompts, prompt
cache off), with `--parallel 1` (one at a time, the rest wait in the queue) versus `--parallel N`.

- Host: Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 VNNI, 15 GB free RAM; `--threads 4`.
- Runs strictly sequential, one server at a time, 5 s apart, nothing else on the CPU.
- Every configuration is one real terminal session captured with
  `tools/screenshots/cli/capture.mjs` (server banner, startup, per-request `[gen]` lines, then
  the client-side result): `*/screenshots/p<parallel>_c<clients>.png`. Raw numbers:
  `*/p<parallel>_c<clients>.json`; tables: `*/summary.txt`. Reproduce with `./run.sh`.

| Model | clients | `--parallel 1` | `--parallel N` (N = clients) | gain |
|---|---|---|---|---|
| SmolLM2-135M-Instruct Q8_0 | 4 | 45.3 tok/s | 100.9 tok/s | 2.23× |
| SmolLM2-135M-Instruct Q8_0 | 8 | 48.6 tok/s | 116.8 tok/s | 2.40× |
| SmolLM2-1.7B-Instruct Q4_K_M | 4 | 12.7 tok/s | 29.8 tok/s | 2.34× |
| SmolLM2-1.7B-Instruct Q4_K_M | 8 | 13.1 tok/s | 35.4 tok/s | 2.71× |

Latency: with `--parallel 1` the last of 8 requests on the 1.7B model finishes after 78 s
(mean 44 s); with `--parallel 8` all finish at about 29 s. A single client (`p1_c1`) runs at
13.3 tok/s (1.7B) / 51.9 tok/s (135M) aggregate, including its prompt.
