#!/usr/bin/env bash
# 2026-10-29, same host as run_h2h_sweep_2026-10-29.sh (Xeon model 85): adds ik_llama.cpp.
# ik_llama.cpp runs with -rtr (run-time repack, its recommended CPU setting) on quantized
# models; BitNet is converted from the same official I2_S GGUF to its IQ2_BN format
# (llama-quantize --allow-requantize ... iq2_bn, as its maintainer documents).
# DeepSeek-V2-Lite: Project Zero vs llama.cpp vs ik_llama.cpp, 4 threads only.
set -euo pipefail
cd "$(dirname "$0")/.."
O=benchmark_results/sweep_h2h_2026-10-29
D=benchmark_results/deepseek_h2h_2026-10-29
L=${LLAMA_COMPLETION:-/home/user/ext/llama.cpp/build/bin/llama-completion}
K=${IK_CLI:-/home/user/ext/ik_llama.cpp/build/bin/llama-cli}
BN=models/bitnet-b1.58-2B-4T-iq2_bn.gguf
[ -f "$BN" ] || "$(dirname "$K")/llama-quantize" --allow-requantize models/bitnet-b1.58-2B-4T-i2_s.gguf "$BN" iq2_bn
python3 tools/bench_sweep_seq.py --out "$O" --threads 1-4 --runs 3 --tokens 500 --cooldown 10 \
  --fixed-length --resume \
  --cli-suite "smollm2_135m_ik:$K:models/SmolLM2-135M-Instruct-f16.gguf" \
  --cli-suite "smollm2_1p7b_q4km_ik:$K:models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf:-rtr" \
  --cli-suite "bitnet_2b_ik_iq2bn:$K:$BN:-rtr" \
  >> "$O/sweep.log" 2>&1
mkdir -p "$D"
python3 tools/bench_sweep_seq.py --out "$D" --threads 4 --runs 3 --tokens 200 --cooldown 20 \
  --fixed-length --resume \
  --suite deepseek_v2_lite_pz:models/DeepSeek-V2-Lite-Chat.Q4_K_S.gguf \
  --ref-suite "deepseek_v2_lite_llamacpp:$L:models/DeepSeek-V2-Lite-Chat.Q4_K_S.gguf" \
  --cli-suite "deepseek_v2_lite_ik:$K:models/DeepSeek-V2-Lite-Chat.Q4_K_S.gguf:-rtr" \
  >> "$D/sweep.log" 2>&1
