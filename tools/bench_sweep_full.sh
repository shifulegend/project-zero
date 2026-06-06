#!/usr/bin/env bash
# bench_sweep_full.sh — Full T1-T8 × SIMD × Classifier benchmark sweep
# Output: CSV + human-readable report

set -euo pipefail
ENGINE="./adaptive_ai_engine"
OUT_CSV="benchmark_results.csv"
OUT_REPORT="BENCHMARK_REPORT.md"

# 5 prompts per model
BITNET_PROMPTS=(
    "The capital of France is"
    "Explain how neural networks learn:"
    "Write a Python function to sort a list:"
    "The theory of relativity states that"
    "What causes the seasons to change?"
)
SMOLLM_PROMPTS=(
    "The capital of France is"
    "Explain how neural networks learn:"
    "Write a Python function to sort a list:"
    "The theory of relativity states that"
    "What causes the seasons to change?"
)

SIMDS=("avx512f" "vnni" "avx2" "scalar")
CLASSIFIERS=("bf16" "int8" "int4")
THREADS_LIST=(1 2 3 4 5 6 7 8)
MAX_TOKENS=30

echo "model,threads,simd,classifier,prompt_num,tok_per_sec,status" > "$OUT_CSV"

run_bench() {
    local model="$1" tok="$2" label="$3" prompt_idx="$4" prompt="$5"
    local threads="$6" simd="$7" cls="$8"
    local result
    result=$(timeout 90 "$ENGINE" \
        --model "$model" --tokenizer "$tok" \
        --prompt "$prompt" --max-tokens $MAX_TOKENS \
        --threads "$threads" --simd "$simd" --classifier "$cls" \
        --seed 42 2>&1) || true
    
    local tps
    tps=$(echo "$result" | grep -oP '[0-9]+\.[0-9]+ tok/s' | grep -oP '[0-9]+\.[0-9]+' | head -1)
    local output
    output=$(echo "$result" | grep -A1 "^Output:" | tail -1 | head -c 200)
    
    if [ -n "$tps" ]; then
        echo "$label,$threads,$simd,$cls,$prompt_idx,$tps,ok" >> "$OUT_CSV"
        echo "  [OK]  ${label} T${threads} ${simd} ${cls} p${prompt_idx}: ${tps} tok/s"
    else
        echo "$label,$threads,$simd,$cls,$prompt_idx,0,error" >> "$OUT_CSV"
        echo "  [ERR] ${label} T${threads} ${simd} ${cls} p${prompt_idx}"
    fi
}

echo "=== Starting full benchmark sweep ==="
echo "Models: BitNet b1.58 2B + SmolLM2-135M"
echo "Threads: 1-8 | SIMD: ${SIMDS[*]} | Classifiers: ${CLASSIFIERS[*]}"
echo "Prompts: 5 per model | Max tokens: $MAX_TOKENS"
echo ""

# BitNet
for t in "${THREADS_LIST[@]}"; do
for simd in "${SIMDS[@]}"; do
for cls in "${CLASSIFIERS[@]}"; do
    for i in "${!BITNET_PROMPTS[@]}"; do
        run_bench "models/bitnet-b1.58-2B-4T.bin" \
                  "models/bitnet-b1.58-2B-4T_tokenizer_proper.bin" \
                  "bitnet-2B" "$i" "${BITNET_PROMPTS[$i]}" \
                  "$t" "$simd" "$cls"
    done
done; done; done

# SmolLM2-135M
for t in "${THREADS_LIST[@]}"; do
for simd in "${SIMDS[@]}"; do
for cls in "${CLASSIFIERS[@]}"; do
    for i in "${!SMOLLM_PROMPTS[@]}"; do
        run_bench "models/SmolLM2-135M-Instruct-f16.gguf" \
                  "models/smollm2-135m-tokenizer.bin" \
                  "smollm2-135M" "$i" "${SMOLLM_PROMPTS[$i]}" \
                  "$t" "$simd" "$cls"
    done
done; done; done

echo ""
echo "=== Benchmark sweep complete. Results: $OUT_CSV ==="
