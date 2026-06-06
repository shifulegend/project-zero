#!/usr/bin/env bash
# bench_full_sweep.sh — T1-T8 × SIMD × Classifier performance ceiling sweep
# Strategy: warm run (max-tokens=25) with a single benchmark prompt for speed grid
#           then capture quality outputs for best settings with all 5 prompts

set -euo pipefail
ENGINE="./adaptive_ai_engine"
CSV="benchmark_full_results.csv"
REPORT_MD="BENCHMARK_FULL_REPORT.md"

SIMDS=("avx512f" "vnni" "avx2" "scalar")
CLASSIFIERS=("bf16" "int8" "int4")
THREADS=(1 2 3 4 5 6 7 8)
BENCH_PROMPT="The theory of relativity states that"
MAX_TOKENS=25

echo "model,threads,simd,classifier,tok_per_sec,tokens,status" > "$CSV"

do_run() {
    local model="$1" tok="$2" label="$3" threads="$4" simd="$5" cls="$6"
    local out tps tokens
    out=$(timeout 60 "$ENGINE" \
        --model "$model" --tokenizer "$tok" \
        --prompt "$BENCH_PROMPT" --max-tokens $MAX_TOKENS \
        --threads "$threads" --simd "$simd" --classifier "$cls" \
        --seed 42 2>&1) || { echo "$label,$threads,$simd,$cls,0,0,timeout" >> "$CSV"; return; }
    
    tps=$(echo "$out" | grep -oP '[0-9]+\.[0-9]+ tok/s' | grep -oP '[0-9]+\.[0-9]+' | head -1)
    tokens=$(echo "$out" | grep -oP '\([0-9]+ tokens\)' | grep -oP '[0-9]+' | head -1)
    
    if [ -n "$tps" ]; then
        echo "$label,$threads,$simd,$cls,$tps,${tokens:-25},ok" >> "$CSV"
        printf "  %-14s T%-2s %-8s %-6s → %6s tok/s\n" "$label" "$threads" "$simd" "$cls" "$tps"
    else
        echo "$label,$threads,$simd,$cls,0,0,error" >> "$CSV"
        printf "  %-14s T%-2s %-8s %-6s → ERROR\n" "$label" "$threads" "$simd" "$cls"
    fi
}

echo "==================================================="
echo " Performance Ceiling Sweep — $(date '+%Y-%m-%d %H:%M')"
echo " Prompt: $BENCH_PROMPT"
echo "==================================================="

echo ""
echo "=== BitNet b1.58 2B-4T ==="
for t in "${THREADS[@]}"; do
  for simd in "${SIMDS[@]}"; do
    for cls in "${CLASSIFIERS[@]}"; do
      do_run "models/bitnet-b1.58-2B-4T.bin" \
             "models/bitnet-b1.58-2B-4T_tokenizer_proper.bin" \
             "bitnet-2B" "$t" "$simd" "$cls"
    done
  done
done

echo ""
echo "=== SmolLM2 135M F16 GGUF ==="
for t in "${THREADS[@]}"; do
  for simd in "${SIMDS[@]}"; do
    for cls in "${CLASSIFIERS[@]}"; do
      do_run "models/SmolLM2-135M-Instruct-f16.gguf" \
             "models/smollm2-135m-tokenizer.bin" \
             "smollm2-135M" "$t" "$simd" "$cls"
    done
  done
done

echo ""
echo "Done. Results saved to: $CSV"
