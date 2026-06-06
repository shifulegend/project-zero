#!/usr/bin/env bash
# bench_sweep.sh — Phase 17 Comprehensive Benchmark
#
# Runs adaptive_ai_engine for every combination of:
#   Threads T=1..MAX_T   × SIMD backends × Classifier formats
#
# Methodology: 25 generated tokens, 3s cooldown between cells.
# Matches Addendum AB methodology exactly (actual engine, real prompt).

set -euo pipefail

MODEL="${1:-models/bitnet-b1.58-2B-4T.bin}"
TOKENIZER="${2:-models/bitnet-b1.58-2B-4T_tokenizer_proper.bin}"
MAX_T="${3:-8}"
MAX_TOKENS=25
COOLDOWN=3
PROMPT="The capital of France is"
ENGINE=./adaptive_ai_engine
OUT_CSV="/tmp/bench_sweep_results.csv"
LOG="/tmp/bench_sweep.log"

SIMD_BACKENDS=("vnni" "vnni256" "avx512f" "avx2" "scalar")
SIMD_NAMES=("AVX-512VNNI" "VNNI-256" "AVX-512F" "AVX2" "Scalar")
CLASSIFIERS=("bf16" "int8" "int4")

echo "threads,simd,classifier,tok_per_sec" > "$OUT_CSV"

echo "══════════════════════════════════════════════════════════════════════"
echo "  Phase 17 Benchmark — $(basename $MODEL)"
echo "  Sweep: T=1..${MAX_T} × ${#SIMD_BACKENDS[@]} SIMD × ${#CLASSIFIERS[@]} classifiers"
echo "  Tokens/run: $MAX_TOKENS  |  Cooldown: ${COOLDOWN}s"
echo "══════════════════════════════════════════════════════════════════════"
printf "  %-7s  %-13s  %-7s  %9s\n" "Threads" "SIMD" "Cls" "tok/s"
echo "  ─────────────────────────────────────────────────────────────────"

PEAK_TPS=0
PEAK_CFG=""

for T in $(seq 1 $MAX_T); do
  for i in "${!SIMD_BACKENDS[@]}"; do
    BACKEND="${SIMD_BACKENDS[$i]}"
    BNAME="${SIMD_NAMES[$i]}"

    for CLS in "${CLASSIFIERS[@]}"; do
      # INT4 only with VNNI; skip for others
      if [[ "$CLS" == "int4" && "$BACKEND" != "vnni" && "$BACKEND" != "vnni256" ]]; then
        printf "  %-7d  %-13s  %-7s  %9s\n" "$T" "$BNAME" "$CLS" "N/A(no VNNI)"
        echo "$T,$BNAME,$CLS,NA" >> "$OUT_CSV"
        continue
      fi

      printf "  %-7d  %-13s  %-7s  ... " "$T" "$BNAME" "$CLS"

      TPS=$(TN_FORCE_BACKEND="$BACKEND" "$ENGINE" \
              --model "$MODEL" \
              --tokenizer "$TOKENIZER" \
              --prompt "$PROMPT" \
              --max-tokens $MAX_TOKENS \
              --threads $T \
              --classifier "$CLS" \
              2>/dev/null | grep -oP '[0-9]+\.[0-9]+ tok/s' | grep -oP '[0-9]+\.[0-9]+' || echo "ERR")

      if [[ "$TPS" == "ERR" || -z "$TPS" ]]; then
        printf "%-9s\n" "ERROR"
        echo "$T,$BNAME,$CLS,ERR" >> "$OUT_CSV"
      else
        MARK=" "
        if (( $(echo "$TPS > $PEAK_TPS" | bc -l) )); then
          PEAK_TPS="$TPS"
          PEAK_CFG="T=$T $BNAME $CLS"
          MARK="*"
        fi
        printf "%8.2f%s\n" "$TPS" "$MARK"
        echo "$T,$BNAME,$CLS,$TPS" >> "$OUT_CSV"
      fi

      sleep $COOLDOWN
    done
  done
done

echo "  ─────────────────────────────────────────────────────────────────"
echo "  Peak: ${PEAK_TPS} tok/s  @ ${PEAK_CFG}  (* marks peak rows)"
echo ""
echo "CSV → $OUT_CSV"
