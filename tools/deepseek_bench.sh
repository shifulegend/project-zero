#!/usr/bin/env bash
# deepseek_bench.sh — Comprehensive DeepSeek inference benchmark
#
# Dimensions:
#   Our engine: T={1..8} × SIMD={scalar,avx2,avx512f,vnni} × CLS={bf16,int8,int4,auto-fast}
#   + fine-grained thread sweep T=1..8 at best SIMD/cls
#   llama.cpp:  T={1..8} via llama-bench
#
# Metrics (no-perf pass):
#   tok/s (from engine output), model load time, wall time, peak RSS, CPU freq
#
# Outputs:
#   benchmark_results/engine_sweep.csv      — full T×SIMD×CLS matrix
#   benchmark_results/engine_threadsweep.csv — T=1..8 at best SIMD/cls
#   benchmark_results/llama_bench.csv       — llama.cpp reference
#   benchmark_results/system_info.txt       — hardware context

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="$REPO_DIR/adaptive_ai_engine"
MODEL="$REPO_DIR/models/deepseek-v2-lite-chat-Q4_K_S.gguf"
LLAMA_BENCH="/home/ubuntu/llama.cpp/build/bin/llama-bench"
LLAMA_CLI="/home/ubuntu/llama.cpp/build/bin/llama-cli"
OUT_DIR="$REPO_DIR/benchmark_results"
mkdir -p "$OUT_DIR"

# Benchmark parameters
MAX_TOKENS=5          # keep each run fast; enough for stable tok/s reading
REPS=1                # repetitions per config (take best)
COOLDOWN=2            # seconds between runs
PROMPT="List the first 10 prime numbers:"
BENCH_DATE=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Dimensions (override with env, e.g. DEEPSEEK_THREADS_MATRIX="1 2 4 8")
THREADS_MATRIX=(${DEEPSEEK_THREADS_MATRIX:-1 2 3 4 5 6 7 8})
THREADS_SWEEP=(${DEEPSEEK_THREADS_SWEEP:-1 2 3 4 5 6 7 8})
SIMD_MODES=(${DEEPSEEK_SIMD_MODES:-scalar avx2 avx512f vnni})
CLASSIFIERS=(${DEEPSEEK_CLASSIFIERS:-bf16 int8 int4 auto-fast})
LLAMA_THREADS_CSV="${DEEPSEEK_LLAMA_THREADS_CSV:-1,2,3,4,5,6,7,8}"
BEST_SIMD="${DEEPSEEK_BEST_SIMD:-auto}"
BEST_CLS="${DEEPSEEK_BEST_CLS:-bf16}"

# ─── System info snapshot ──────────────────────────────────────────────
collect_sysinfo() {
  {
    echo "=== Benchmark date: $BENCH_DATE ==="
    echo "=== CPU ==="
    lscpu | grep -E "Model name|CPU.s.|Thread|Core|Socket|MHz|cache"
    echo "=== Memory ==="
    free -h
    echo "=== OS ==="
    uname -a
    echo "=== Disk ==="
    df -h "$MODEL"
    echo "=== Competing processes (top 10 by CPU) ==="
    ps aux --sort=-%cpu | head -12
    echo "=== CPU governor ==="
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u || echo "N/A"
    echo "=== CPU current freq (MHz) ==="
    awk '{printf "%d\n", $1/1000}' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || echo "N/A"
  } > "$OUT_DIR/system_info.txt"
}

# ─── Helper: run engine once, return tok/s and timing ─────────────────
# Args: threads simd cls
# Prints: "TOKS=<float> LOAD=<float> WALL=<float> RSS=<int_kb>"
run_engine_once() {
  local T="$1" SIMD="$2" CLS="$3"
  local TMP_OUT="$OUT_DIR/_run_tmp.txt"
  local TMP_ERR="$OUT_DIR/_run_stderr.txt"

  # Background RSS monitor
  local RSS_MAX=0
  local WALL_START WALL_END

  WALL_START=$(date +%s%N)

  # Run engine, capture output
  "$ENGINE" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --threads "$T" \
    --simd "$SIMD" \
    --classifier "$CLS" \
    --temperature 0.0 \
    2>"$TMP_ERR" \
    >"$TMP_OUT" &
  local PID=$!

  # Poll RSS while engine runs
  RSS_MAX=0
  while kill -0 "$PID" 2>/dev/null; do
    local RSS
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    [[ "$RSS" -gt "$RSS_MAX" ]] && RSS_MAX=$RSS
    sleep 0.5
  done
  wait "$PID" || true

  WALL_END=$(date +%s%N)
  local WALL_MS=$(( (WALL_END - WALL_START) / 1000000 ))

  # Extract tok/s from engine output.
  # DeepSeek GGUF runs may print summary lines to either stdout or stderr, so
  # always parse the combined streams to avoid false 0 tok/s rows.
  local TPS NTOKS LOAD_MS INFER_MS
  TPS=$(cat "$TMP_OUT" "$TMP_ERR" 2>/dev/null | grep -oP '^\d[\d.]+(?= tok/s)' | tail -1 || echo "0")
  NTOKS=$(cat "$TMP_OUT" "$TMP_ERR" 2>/dev/null | grep -oP '\(\K\d+(?= tokens\))' | tail -1 || echo "0")
  [[ -z "$TPS"   ]] && TPS="0"
  [[ -z "$NTOKS" ]] && NTOKS="0"

  # Load time = wall_time - inference_time (awk for float arithmetic)
  INFER_MS=$(awk "BEGIN{printf \"%d\", ($NTOKS>0 && $TPS>0) ? $NTOKS*1000/$TPS : 0}")
  LOAD_MS=$(( WALL_MS - INFER_MS ))
  [[ $LOAD_MS -lt 0 ]] && LOAD_MS=0

  echo "TOKS=${TPS} NTOKS=${NTOKS} LOAD=${LOAD_MS} WALL=${WALL_MS} RSS=${RSS_MAX}"
}

# ─── Helper: run N reps, return best tok/s row ────────────────────────
run_engine_best() {
  local T="$1" SIMD="$2" CLS="$3"
  local BEST_TPS=0 BEST_ROW=""
  for rep in $(seq 1 $REPS); do
    local ROW TPS
    ROW=$(run_engine_once "$T" "$SIMD" "$CLS")
    TPS=$(echo "$ROW" | tr ' ' '\n' | grep '^TOKS=' | cut -d= -f2 | head -1 || echo "0")
    # awk for float comparison — avoids (( )) integer-only limitation
    if awk "BEGIN{exit ($TPS > $BEST_TPS) ? 0 : 1}"; then
      BEST_TPS="$TPS"
      BEST_ROW="$ROW"
    fi
    sleep $COOLDOWN
  done
  echo "$BEST_ROW"
}

row_get() {
  local ROW="$1" KEY="$2"
  echo "$ROW" | tr ' ' '\n' | grep "^${KEY}=" | cut -d= -f2 | head -1
}

# ─── Phase 0: System info ────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       DeepSeek-V2-Lite Inference Benchmark — Project Zero        ║"
echo "║                   $(date -u '+%Y-%m-%d %H:%M UTC')                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "[0/4] Collecting system info..."
collect_sysinfo
echo "      CPU: $(grep 'Model name' "$OUT_DIR/system_info.txt" | head -1 | sed 's/.*: //')"
echo "      RAM: $(free -h | awk '/^Mem:/{print $2}') total, $(free -h | awk '/^Mem:/{print $7}') available"
echo "      Competing CPU: $(ps aux --sort=-%cpu | awk 'NR>1 && NR<5 {print $11}' | tr '\n' ' ')"
echo

# ─── Phase 1: Full matrix T×SIMD×CLS ──────────────────────────────────
ENGINE_CSV="$OUT_DIR/engine_sweep.csv"
echo "date,engine,threads,simd,classifier,tok_s,ntoks,load_ms,wall_ms,rss_kb" > "$ENGINE_CSV"

TOTAL=$(( ${#THREADS_MATRIX[@]} * ${#SIMD_MODES[@]} * ${#CLASSIFIERS[@]} ))
N=0

echo "[1/4] Engine sweep: T={${THREADS_MATRIX[*]}} × SIMD={${SIMD_MODES[*]}} × CLS={${CLASSIFIERS[*]}}"
echo "      Total configs: $TOTAL  (${REPS} rep each)"
echo
printf "  %-4s  %-9s  %-9s  %8s  %8s  %8s  %8s\n" "T" "SIMD" "CLS" "tok/s" "load_ms" "wall_ms" "RSS_MB"
echo "  ──────────────────────────────────────────────────────────────────"

for T in "${THREADS_MATRIX[@]}"; do
  for SIMD in "${SIMD_MODES[@]}"; do
    for CLS in "${CLASSIFIERS[@]}"; do
      N=$(( N + 1 ))
      printf "  %-4s  %-9s  %-9s  " "$T" "$SIMD" "$CLS"

      ROW=$(run_engine_best "$T" "$SIMD" "$CLS")
      TPS=$(row_get "$ROW" TOKS)
      NTOKS=$(row_get "$ROW" NTOKS)
      LOAD=$(row_get "$ROW" LOAD)
      WALL=$(row_get "$ROW" WALL)
      RSS=$(row_get "$ROW" RSS)
      RSS_MB=$(( RSS / 1024 ))

      printf "%8s  %8s  %8s  %8s\n" "$TPS" "$LOAD" "$WALL" "$RSS_MB"
      echo "$BENCH_DATE,project-zero,$T,$SIMD,$CLS,$TPS,$NTOKS,$LOAD,$WALL,$RSS" >> "$ENGINE_CSV"
    done
  done
done

echo
echo "      Saved → $ENGINE_CSV"
echo

# ─── Phase 2: Fine-grained thread sweep (best SIMD/CLS) ────────────────
THREAD_CSV="$OUT_DIR/engine_threadsweep.csv"
echo "date,engine,threads,simd,classifier,tok_s,ntoks,load_ms,wall_ms,rss_kb" > "$THREAD_CSV"

echo "[2/4] Thread sweep T=1..8 at SIMD=$BEST_SIMD CLS=$BEST_CLS"
printf "  %-4s  %8s  %8s  %8s\n" "T" "tok/s" "load_ms" "RSS_MB"
echo "  ────────────────────────────────────────"

for T in "${THREADS_SWEEP[@]}"; do
  printf "  %-4s  " "$T"
  ROW=$(run_engine_best "$T" "$BEST_SIMD" "$BEST_CLS")
  TPS=$(row_get "$ROW" TOKS)
  NTOKS=$(row_get "$ROW" NTOKS)
  LOAD=$(row_get "$ROW" LOAD)
  WALL=$(row_get "$ROW" WALL)
  RSS=$(row_get "$ROW" RSS)
  RSS_MB=$(( RSS / 1024 ))
  printf "%8s  %8s  %8s\n" "$TPS" "$LOAD" "$RSS_MB"
  echo "$BENCH_DATE,project-zero,$T,$BEST_SIMD,$BEST_CLS,$TPS,$NTOKS,$LOAD,$WALL,$RSS" >> "$THREAD_CSV"
done

echo "      Saved → $THREAD_CSV"
echo

# ─── Phase 3: llama.cpp benchmark ─────────────────────────────────────
LLAMA_CSV="$OUT_DIR/llama_bench.csv"
echo "[3/4] llama.cpp llama-bench T={${LLAMA_THREADS_CSV}} (pp=32, tg=32)"

if [[ -f "$LLAMA_BENCH" ]]; then
  # Run llama-bench and capture output
  "$LLAMA_BENCH" \
    -m "$MODEL" \
    -t "$LLAMA_THREADS_CSV" \
    -p 32 \
    -n 32 \
    -r 3 \
    --output csv \
    2>/dev/null | tee "$LLAMA_CSV"
  echo "  Saved → $LLAMA_CSV"
else
  echo "  WARNING: llama-bench not found at $LLAMA_BENCH"
fi
echo

# ─── Phase 4: Environment monitoring snapshot ──────────────────────────
echo "[4/4] Final system snapshot..."
{
  echo "=== Post-benchmark snapshot: $(date -u) ==="
  echo "=== CPU freq after benchmark (MHz) ==="
  awk '{printf "%d\n", $1/1000}' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || echo "N/A"
  echo "=== Memory after benchmark ==="
  free -h
  echo "=== Top CPU consumers ==="
  ps aux --sort=-%cpu | head -12
  echo "=== Disk I/O summary ==="
  iostat -x 1 1 2>/dev/null || echo "N/A"
} >> "$OUT_DIR/system_info.txt"

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  Phase 1-4 COMPLETE. Results saved to: $OUT_DIR"
echo "  Run tools/deepseek_bench_perf.sh for hardware counter profiling."
echo "  Run tools/deepseek_bench_report.py to generate BENCHMARK_REPORT.md"
echo "═══════════════════════════════════════════════════════════════════"
