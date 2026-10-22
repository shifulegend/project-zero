#!/usr/bin/env bash
# Reproduces benchmark_results/parallel_2026-10-22: every configuration is one
# real terminal session (server banner + log + client results) captured with
# tools/screenshots/cli/capture.mjs. Runs strictly one after another.
set -eu
D=benchmark_results/parallel_2026-10-22
for model in smollm2-q8_0 SmolLM2-1.7B-Instruct-Q4_K_M; do
  out=$D/$model; mkdir -p "$out/screenshots"
  for cfg in "1 1" "1 4" "4 4" "1 8" "8 8"; do
    set -- $cfg; p=$1; c=$2
    PZ_CAPTURE_ROWS=140 PZ_CAPTURE_TIMEOUT_MS=900000 node tools/screenshots/cli/capture.mjs \
      "$out/screenshots/p${p}_c${c}.png" \
      "Project Zero --parallel $p, $c concurrent clients, $model, 4 threads, 128 tokens each (Xeon 4 vCPU)" \
      -- python3 tools/bench_parallel.py --model "models/$model.gguf" --parallel "$p" --clients "$c" \
         --max-tokens 128 --threads 4 --show-server --out "$out"
    sleep 5
  done
  python3 tools/bench_parallel.py --model "models/$model.gguf" --parallel 1 4 8 --clients 1 4 8 \
    --max-tokens 128 --threads 4 --out "$out" --summary-only
done
