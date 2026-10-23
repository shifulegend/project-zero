#!/usr/bin/env bash
# Head-to-head sweep of 2026-10-23 on the 4-vCPU Xeon (model 207) — same method as
# run_h2h_sweep_2026-10-19.sh, plus BitNet with Project Zero's INT4 output layer
# (the configuration behind the i5-11300H headline). Resumable: re-run after a restart.
set -euo pipefail
cd "$(dirname "$0")/.."
O=benchmark_results/sweep_h2h_2026-10-23
L=${LLAMA_COMPLETION:-/home/user/ext/llama.cpp/build/bin/llama-completion}
B=${BITNET_COMPLETION:-/home/user/ext/BitNet/build/bin/llama-completion}
mkdir -p "$O"
python3 tools/bench_sweep_seq.py --out "$O" --threads 1-4 --runs 3 --tokens 500 --cooldown 10 \
  --fixed-length --resume \
  --suite smollm2_135m_pz:models/SmolLM2-135M-Instruct-f16.gguf \
  --ref-suite "smollm2_135m_llamacpp:$L:models/SmolLM2-135M-Instruct-f16.gguf" \
  --suite smollm2_1p7b_q4km_pz:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --ref-suite "smollm2_1p7b_q4km_llamacpp:$L:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf" \
  --suite bitnet_2b_pz:models/bitnet-b1.58-2B-4T-i2_s.gguf \
  --suite bitnet_2b_pz_int4:models/bitnet-b1.58-2B-4T-i2_s.gguf:--classifier,int4 \
  --ref-suite "bitnet_2b_bitnetcpp:$B:models/bitnet-b1.58-2B-4T-i2_s.gguf" \
  >> "$O/sweep.log" 2>&1
