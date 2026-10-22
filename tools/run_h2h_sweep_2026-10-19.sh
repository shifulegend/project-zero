#!/usr/bin/env bash
# Head-to-head sweep of 2026-10-19 (Addendum AL). Resumable: re-run after a restart.
# Reference engines are built outside the repo (see docs/PERFORMANCE_CEILING_REPORT.md AL).
set -euo pipefail
cd "$(dirname "$0")/.."
O=benchmark_results/sweep_h2h_2026-10-19
L=${LLAMA_COMPLETION:-/home/user/ext/llama.cpp/build/bin/llama-completion}
B=${BITNET_COMPLETION:-/home/user/ext/BitNet/build/bin/llama-completion}
if [ "${1:-}" = "postfix" ]; then
  # Project Zero again after the startup fixes (bw cache, BPE merge lookup,
  # lazy calloc, I2_S repack) — same host, same method; decode should match,
  # wall time and RSS should not.
  exec python3 tools/bench_sweep_seq.py --out "$O" --threads 1-4 --runs 3 --tokens 500 --cooldown 10 \
    --fixed-length --resume \
    --suite smollm2_135m_pz_postfix:models/SmolLM2-135M-Instruct-f16.gguf \
    --suite smollm2_1p7b_q4km_pz_postfix:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
    --suite bitnet_2b_pz_postfix:models/bitnet-b1.58-2B-4T-i2_s.gguf \
    >> "$O/sweep.log" 2>&1
fi
python3 tools/bench_sweep_seq.py --out "$O" --threads 1-4 --runs 3 --tokens 500 --cooldown 10 \
  --fixed-length --resume \
  --suite smollm2_135m_pz:models/SmolLM2-135M-Instruct-f16.gguf \
  --ref-suite "smollm2_135m_llamacpp:$L:models/SmolLM2-135M-Instruct-f16.gguf" \
  --suite smollm2_1p7b_q4km_pz:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --ref-suite "smollm2_1p7b_q4km_llamacpp:$L:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf" \
  --suite bitnet_2b_pz:models/bitnet-b1.58-2B-4T-i2_s.gguf \
  --ref-suite "bitnet_2b_bitnetcpp:$B:models/bitnet-b1.58-2B-4T-i2_s.gguf" \
  >> "$O/sweep.log" 2>&1
