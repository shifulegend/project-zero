#!/usr/bin/env bash
# run_comparison.sh
#
# Runs llama.cpp then our engine SEQUENTIALLY — one at a time.
# Each dumps intermediate tensors to a CSV, exits, then the next starts.
# Use: ./run_comparison.sh ["prompt"] [max_tokens]

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="/home/<USER>/Documents/project-zero"
LLAMA="/home/ubuntu/llama.cpp"
MODEL="${PROJECT}/models/deepseek-v2-lite-chat-Q4_K_S.gguf"
TOKENIZER="${PROJECT}/models/deepseek-tokenizer-gguf.bin"

PROMPT="${1:-User: What is the capital of France?\n\nAssistant:}"
MAX_TOK="${2:-1}"

LLAMA_RAW="${DIR}/llama_raw.txt"
LLAMA_CSV="${DIR}/llama_tensors.csv"
OURS_CSV="${DIR}/ours_tensors.csv"

ram_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
drop_cache() {
    sync
    echo 1 | echo "<YOUR_SUDO_PASSWORD>" | sudo -S tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
}

echo "════════════════════════════════════════════════════════"
echo "  DeepSeek Sequential Tensor Comparison"
echo "  Never both engines in RAM at the same time"
echo "════════════════════════════════════════════════════════"
echo "  Prompt:     ${PROMPT}"
echo "  Max tokens: ${MAX_TOK}"
echo "  RAM now:    $(ram_mb) MB available"
echo ""

# ────────────────────────────────────────────────────────────────────────
# STEP 1 — llama.cpp: run eval-callback, capture all tensor output
# ────────────────────────────────────────────────────────────────────────
echo "━━━ [1/3] llama.cpp forward pass ━━━━━━━━━━━━━━━━━━━━━"
echo "    RAM before: $(ram_mb) MB"

LD_LIBRARY_PATH="${LLAMA}/build/bin" \
  "${LLAMA}/build/bin/llama-eval-callback" \
    --model      "${MODEL}" \
    --ctx-size   512 \
    --threads    4 \
    --n-predict  "${MAX_TOK}" \
    --prompt     "${PROMPT}" \
    --log-disable \
  2>"${LLAMA_RAW}"

echo "    RAM after:  $(ram_mb) MB"
echo "    Raw log:    $(wc -l < "${LLAMA_RAW}") lines"

# Parse the eval-callback output into CSV
python3 "${DIR}/parse_llama_log.py" "${LLAMA_RAW}" "${LLAMA_CSV}"
echo "    CSV rows:   $(( $(wc -l < "${LLAMA_CSV}") - 1 ))"

drop_cache
echo "    RAM after cache drop: $(ram_mb) MB"

# ────────────────────────────────────────────────────────────────────────
# STEP 2 — our engine: run with --dump-tensors
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [2/3] Our engine forward pass ━━━━━━━━━━━━━━━━━━━━"
echo "    RAM before: $(ram_mb) MB"

"${PROJECT}/adaptive_ai_engine" \
    --model         "${MODEL}" \
    --tokenizer     "${TOKENIZER}" \
    --threads       4 \
    --max-tokens    "${MAX_TOK}" \
    --prompt        "${PROMPT}" \
    --dump-tensors  "${OURS_CSV}"

echo "    RAM after:  $(ram_mb) MB"
echo "    CSV rows:   $(( $(wc -l < "${OURS_CSV}") - 1 ))"

drop_cache

# ────────────────────────────────────────────────────────────────────────
# STEP 3 — compare
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [3/3] Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "${DIR}/compare_tensors.py" "${LLAMA_CSV}" "${OURS_CSV}"
