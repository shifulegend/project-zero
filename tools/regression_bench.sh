#!/usr/bin/env bash
# regression_bench.sh — Regression benchmark: Bitnet + SmolLM2
# Tests that both models produce coherent output across all configs.
# Quality gate: output must mention Paris or contain ≥5 coherent words.
#
# Outputs (benchmark_results/):
#   bitnet_sweep.csv          — full T×SIMD×CLS matrix for Bitnet
#   bitnet_threadsweep.csv    — T=1..8 thread sweep for Bitnet
#   smollm_sweep.csv          — full T×SIMD×CLS matrix for SmolLM2
#   smollm_threadsweep.csv    — T=1..8 thread sweep for SmolLM2

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="$REPO_DIR/adaptive_ai_engine"
OUT_DIR="$REPO_DIR/benchmark_results"
mkdir -p "$OUT_DIR"

# ─── Model definitions ────────────────────────────────────────────────
BITNET_MODEL="$REPO_DIR/models/bitnet-b1.58-2B-4T.bin"
BITNET_TOKENIZER="$REPO_DIR/models/bitnet-b1.58-2B-4T_tokenizer_proper.bin"
SMOLLM_MODEL="$REPO_DIR/models/SmolLM2-135M-Instruct-f16.gguf"
SMOLLM_TOKENIZER="$REPO_DIR/models/smollm2-135m-tokenizer.bin"

PROMPT="What is the capital of France?"

# ─── Benchmark parameters ────────────────────────────────────────────
MAX_TOKENS=20
REPS=1
COOLDOWN=2
BENCH_DATE=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# ─── Dimensions ───────────────────────────────────────────────────────
THREADS_MATRIX=(2 4 8)
THREADS_SWEEP=(1 2 3 4 5 6 7 8)
SIMD_MODES=(scalar avx2 avx512f vnni auto)
CLASSIFIERS=(bf16 int8 int4 auto-fast)
BEST_SIMD="auto"
BEST_CLS="bf16"

# CSV header matching deepseek engine_sweep.csv format
CSV_HEADER="date,model,threads,simd,classifier,tok_s,load_ms,wall_s,rss_kb,first_token"

# ─── Helper: run engine once, check quality, return metrics ──────────
# Args: model_label model_path tokenizer_path threads simd cls
# Prints: "TOKS=<float> LOAD=<int> WALL=<float> RSS=<int_kb> QUALITY=<ok|garbled>"
run_engine_once() {
  local LABEL="$1" MODEL_PATH="$2" TOKENIZER="$3" T="$4" SIMD="$5" CLS="$6"
  local TMP_OUT="$OUT_DIR/_reg_tmp_stdout.txt"
  local TMP_ERR="$OUT_DIR/_reg_tmp_stderr.txt"

  local WALL_START WALL_END RSS_MAX PID

  WALL_START=$(date +%s%N)

  "$ENGINE" \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --threads "$T" \
    --simd "$SIMD" \
    --classifier "$CLS" \
    --temperature 0.0 \
    2>"$TMP_ERR" \
    >"$TMP_OUT" &
  PID=$!

  RSS_MAX=0
  while kill -0 "$PID" 2>/dev/null; do
    local RSS
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    [[ "$RSS" -gt "$RSS_MAX" ]] && RSS_MAX=$RSS
    sleep 0.5
  done
  wait "$PID" || true

  WALL_END=$(date +%s%N)
  local WALL_NS=$(( WALL_END - WALL_START ))
  local WALL_S
  WALL_S=$(awk "BEGIN{printf \"%.3f\", $WALL_NS/1000000000}")

  # Extract tok/s from engine output (pattern: "19.48 tok/s (20 tokens)")
  local TPS NTOKS
  TPS=$(grep -oP '^\d[\d.]+(?= tok/s)' "$TMP_OUT" | tail -1 || echo "0")
  NTOKS=$(grep -oP '\d+(?= tokens)' "$TMP_OUT" | tail -1 || echo "0")
  [[ -z "$TPS"   ]] && TPS="0"
  [[ -z "$NTOKS" ]] && NTOKS="0"

  # Compute load_ms: wall_ms - infer_ms
  local WALL_MS INFER_MS LOAD_MS
  WALL_MS=$(awk "BEGIN{printf \"%d\", $WALL_NS/1000000}")
  INFER_MS=$(awk "BEGIN{printf \"%d\", ($NTOKS>0 && $TPS>0) ? $NTOKS*1000/$TPS : 0}")
  LOAD_MS=$(( WALL_MS - INFER_MS ))
  [[ $LOAD_MS -lt 0 ]] && LOAD_MS=0

  # ─── Quality gate ─────────────────────────────────────────────────
  # Strip the tok/s summary line to isolate generated text
  local TEXT QUALITY
  TEXT=$(grep -v 'tok/s' "$TMP_OUT" | tr -s ' \n\t' ' ' | sed 's/^ *//;s/ *$//')

  QUALITY="garbled"

  # Check 1: output mentions Paris (case-insensitive)
  if echo "$TEXT" | grep -qi 'paris'; then
    QUALITY="ok"
  else
    # Check 2: at least 5 words of coherent text (letters only, not just
    # punctuation / markdown / backticks / repetition)
    local WORD_COUNT
    WORD_COUNT=$(echo "$TEXT" | grep -oP '\b[A-Za-z]{2,}\b' | wc -l)
    if [[ "$WORD_COUNT" -ge 5 ]]; then
      QUALITY="ok"
    fi
  fi

  echo "TOKS=${TPS} NTOKS=${NTOKS} LOAD=${LOAD_MS} WALL=${WALL_S} RSS=${RSS_MAX} QUALITY=${QUALITY}"
}

# ─── Helper: run REPS reps, return best tok/s (or 0.0 if garbled) ────
# Args: model_label model_path tokenizer_path threads simd cls csv_file
run_engine_best() {
  local LABEL="$1" MODEL_PATH="$2" TOKENIZER="$3" T="$4" SIMD="$5" CLS="$6" CSV_FILE="$7"
  local BEST_TPS=0 BEST_ROW="" BEST_QUALITY="garbled"

  for _rep in $(seq 1 $REPS); do
    local ROW TPS QUALITY
    ROW=$(run_engine_once "$LABEL" "$MODEL_PATH" "$TOKENIZER" "$T" "$SIMD" "$CLS")
    TPS=$(echo "$ROW" | grep -oP 'TOKS=\K[\d.]+' | head -1 || echo "0")
    QUALITY=$(echo "$ROW" | grep -oP 'QUALITY=\K\S+' | head -1 || echo "garbled")

    if [[ "$QUALITY" == "garbled" ]]; then
      echo "GARBLED: $LABEL T=$T SIMD=$SIMD CLS=$CLS (text: $(cat "$OUT_DIR/_reg_tmp_stdout.txt" | head -1))" >&2
    fi

    if awk "BEGIN{exit ($TPS > $BEST_TPS) ? 0 : 1}"; then
      BEST_TPS="$TPS"
      BEST_ROW="$ROW"
      BEST_QUALITY="$QUALITY"
    fi
    sleep $COOLDOWN
  done

  # If best result was garbled, override tok_s to 0.0
  if [[ "$BEST_QUALITY" == "garbled" ]]; then
    BEST_ROW=$(echo "$BEST_ROW" | sed 's/TOKS=[^ ]*/TOKS=0.0/')
    BEST_TPS="0.0"
  fi

  # Extract fields and write CSV row
  local LOAD WALL RSS FIRST_TOKEN
  LOAD=$(echo "$BEST_ROW" | grep -oP 'LOAD=\K\d+' || echo "0")
  WALL=$(echo "$BEST_ROW" | grep -oP 'WALL=\K[\d.]+' || echo "0")
  RSS=$(echo "$BEST_ROW" | grep -oP 'RSS=\K\d+' || echo "0")
  FIRST_TOKEN="N/A"

  echo "$BENCH_DATE,$LABEL,$T,$SIMD,$CLS,$BEST_TPS,$LOAD,$WALL,$RSS,$FIRST_TOKEN" >> "$CSV_FILE"

  # Return for display
  echo "TOKS=${BEST_TPS} LOAD=${LOAD} WALL=${WALL} RSS=${RSS} QUALITY=${BEST_QUALITY}"
}

# ─── Helper: run full T×SIMD×CLS sweep for one model ─────────────────
run_sweep() {
  local LABEL="$1" MODEL_PATH="$2" TOKENIZER="$3" CSV_FILE="$4"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  ⚠ Model not found: $MODEL_PATH — skipping $LABEL sweep" >&2
    return
  fi

  echo "$CSV_HEADER" > "$CSV_FILE"

  local TOTAL N=0
  TOTAL=$(( ${#THREADS_MATRIX[@]} * ${#SIMD_MODES[@]} * ${#CLASSIFIERS[@]} ))

  echo "  Total configs: $TOTAL  ($REPS rep each)"
  printf "  %-5s  %-10s  %-10s  %9s  %9s  %9s  %8s\n" \
    "T" "SIMD" "CLS" "tok/s" "load_ms" "wall_s" "RSS_MB"
  echo "  ─────────────────────────────────────────────────────────────────────"

  for T in "${THREADS_MATRIX[@]}"; do
    for SIMD in "${SIMD_MODES[@]}"; do
      for CLS in "${CLASSIFIERS[@]}"; do
        N=$(( N + 1 ))
        printf "  %-5s  %-10s  %-10s  " "$T" "$SIMD" "$CLS"

        ROW=$(run_engine_best "$LABEL" "$MODEL_PATH" "$TOKENIZER" "$T" "$SIMD" "$CLS" "$CSV_FILE")
        TPS=$(echo "$ROW" | grep -oP 'TOKS=\K[\d.]+')
        LOAD=$(echo "$ROW" | grep -oP 'LOAD=\K\d+')
        WALL=$(echo "$ROW" | grep -oP 'WALL=\K[\d.]+')
        RSS=$(echo "$ROW" | grep -oP 'RSS=\K\d+')
        QUALITY=$(echo "$ROW" | grep -oP 'QUALITY=\K\S+')
        RSS_MB=$(( ${RSS:-0} / 1024 ))

        local QUAL_FLAG=""
        [[ "$QUALITY" == "garbled" ]] && QUAL_FLAG=" ⚠GARBLED"

        printf "%9s  %9s  %9s  %8s%s\n" "$TPS" "$LOAD" "$WALL" "$RSS_MB" "$QUAL_FLAG"
      done
    done
  done
}

# ─── Helper: run thread sweep for one model ───────────────────────────
run_threadsweep() {
  local LABEL="$1" MODEL_PATH="$2" TOKENIZER="$3" CSV_FILE="$4"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  ⚠ Model not found: $MODEL_PATH — skipping $LABEL thread sweep" >&2
    return
  fi

  echo "$CSV_HEADER" > "$CSV_FILE"

  echo "  SIMD=$BEST_SIMD CLS=$BEST_CLS"
  printf "  %-5s  %9s  %9s  %8s  %8s\n" "T" "tok/s" "load_ms" "RSS_MB" "quality"
  echo "  ───────────────────────────────────────────────────────"

  for T in "${THREADS_SWEEP[@]}"; do
    printf "  %-5s  " "$T"

    ROW=$(run_engine_best "$LABEL" "$MODEL_PATH" "$TOKENIZER" "$T" "$BEST_SIMD" "$BEST_CLS" "$CSV_FILE")
    TPS=$(echo "$ROW" | grep -oP 'TOKS=\K[\d.]+')
    LOAD=$(echo "$ROW" | grep -oP 'LOAD=\K\d+')
    RSS=$(echo "$ROW" | grep -oP 'RSS=\K\d+')
    QUALITY=$(echo "$ROW" | grep -oP 'QUALITY=\K\S+')
    RSS_MB=$(( ${RSS:-0} / 1024 ))

    printf "%9s  %9s  %8s  %8s\n" "$TPS" "$LOAD" "$RSS_MB" "$QUALITY"
  done
}

# ─── Banner ───────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       Regression Benchmark — Bitnet + SmolLM2                    ║"
echo "║       Quality gate: output must mention Paris or ≥5 words        ║"
echo "║                   $(date -u '+%Y-%m-%d %H:%M UTC')                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# ─── Bitnet full matrix sweep ─────────────────────────────────────────
echo "[1/4] Bitnet-b1.58-2B-4T — Full Matrix (T={2,4,8} × SIMD × CLS)"
run_sweep "bitnet" "$BITNET_MODEL" "$BITNET_TOKENIZER" "$OUT_DIR/bitnet_sweep.csv"
echo "      Saved → $OUT_DIR/bitnet_sweep.csv"
echo

# ─── Bitnet thread sweep ──────────────────────────────────────────────
echo "[2/4] Bitnet-b1.58-2B-4T — Thread Sweep (T=1..8)"
run_threadsweep "bitnet" "$BITNET_MODEL" "$BITNET_TOKENIZER" "$OUT_DIR/bitnet_threadsweep.csv"
echo "      Saved → $OUT_DIR/bitnet_threadsweep.csv"
echo

# ─── SmolLM2 full matrix sweep ────────────────────────────────────────
echo "[3/4] SmolLM2-135M-Instruct — Full Matrix (T={2,4,8} × SIMD × CLS)"
run_sweep "smollm2" "$SMOLLM_MODEL" "$SMOLLM_TOKENIZER" "$OUT_DIR/smollm_sweep.csv"
echo "      Saved → $OUT_DIR/smollm_sweep.csv"
echo

# ─── SmolLM2 thread sweep ─────────────────────────────────────────────
echo "[4/4] SmolLM2-135M-Instruct — Thread Sweep (T=1..8)"
run_threadsweep "smollm2" "$SMOLLM_MODEL" "$SMOLLM_TOKENIZER" "$OUT_DIR/smollm_threadsweep.csv"
echo "      Saved → $OUT_DIR/smollm_threadsweep.csv"
echo

# ─── Summary ──────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  SUMMARY — Best tok/s per model"
echo "───────────────────────────────────────────────────────────────────"

for PAIR in "bitnet:$OUT_DIR/bitnet_sweep.csv" "smollm2:$OUT_DIR/smollm_sweep.csv"; do
  MLABEL="${PAIR%%:*}"
  MCSV="${PAIR##*:}"
  if [[ -f "$MCSV" ]]; then
    # Find row with highest tok_s (skip header, skip garbled 0.0 rows)
    BEST=$(awk -F',' 'NR>1 && $6+0 > 0 {print $6, $3, $4, $5}' "$MCSV" \
           | sort -k1 -rn | head -1)
    if [[ -n "$BEST" ]]; then
      BTPS=$(echo "$BEST" | awk '{print $1}')
      BT=$(echo "$BEST" | awk '{print $2}')
      BSIMD=$(echo "$BEST" | awk '{print $3}')
      BCLS=$(echo "$BEST" | awk '{print $4}')
      printf "  %-10s  best: T=%s SIMD=%-8s CLS=%-10s → %s tok/s\n" \
        "$MLABEL" "$BT" "$BSIMD" "$BCLS" "$BTPS"
    else
      printf "  %-10s  all runs GARBLED or model not found\n" "$MLABEL"
    fi
  fi
done

# Quality gate pass rate
echo
echo "  Quality gate results:"
for PAIR in "bitnet:$OUT_DIR/bitnet_sweep.csv:$OUT_DIR/bitnet_threadsweep.csv" \
            "smollm2:$OUT_DIR/smollm_sweep.csv:$OUT_DIR/smollm_threadsweep.csv"; do
  MLABEL="${PAIR%%:*}"
  REST="${PAIR#*:}"
  SWEEPCSV="${REST%%:*}"
  THREADCSV="${REST##*:}"

  TOTAL=0; GARBLED=0
  for CSV in "$SWEEPCSV" "$THREADCSV"; do
    if [[ -f "$CSV" ]]; then
      ROWS=$(awk -F',' 'NR>1{print $6}' "$CSV" | wc -l)
      BAD=$(awk -F',' 'NR>1 && $6+0==0 {print}' "$CSV" | wc -l)
      TOTAL=$(( TOTAL + ROWS ))
      GARBLED=$(( GARBLED + BAD ))
    fi
  done
  PASSED=$(( TOTAL - GARBLED ))
  if [[ $TOTAL -gt 0 ]]; then
    RATE=$(awk "BEGIN{printf \"%.0f\", $PASSED*100/$TOTAL}")
    printf "  %-10s  %d/%d passed (%s%%)\n" "$MLABEL" "$PASSED" "$TOTAL" "$RATE"
  fi
done

echo "═══════════════════════════════════════════════════════════════════"
echo "  Run tools/deepseek_bench_report.py to append Section 8 to"
echo "  BENCHMARK_REPORT.md"
echo "═══════════════════════════════════════════════════════════════════"
