#!/usr/bin/env bash
# thread_sweep.sh — T=1..8 benchmark sweep
# For each thread count: 2 clean runs + 1 perf stat run (with per-CPU, RAM, disk monitoring)
# Output: tests/thread_sweep_results.txt

set -e
cd "$(dirname "$0")/.."

ENGINE=./adaptive_ai_engine
MODEL=models/bitnet-b1.58-2B-4T.bin
TOKENIZER=models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
PROMPT="The speed of light is approximately"
MAX_TOKENS=100
TEMP=0.0
OUT=tests/thread_sweep_results.txt

# ── Parse optional --resume flag (start from a given thread count) ──
RESUME_FROM=1
for arg in "$@"; do
  case "$arg" in
    --resume=*) RESUME_FROM="${arg#--resume=}" ;;
  esac
done

# ── Header ──
{
echo "============================================================"
echo "  Project Zero — T=1..8 Thread Sweep"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Prompt : \"$PROMPT\""
echo "  Tokens : $MAX_TOKENS  Temperature: $TEMP  (greedy/deterministic)"
echo "  KV     : Sliding Window I4, 1024 ctx"
echo "  CPU    : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  Kernel : $(uname -r)"
echo "  RAM    : $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
echo "============================================================"
echo ""
} | tee -a "$OUT"

run_clean() {
  local T=$1 RUN=$2
  local result tok_s output_text
  result=$($ENGINE \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --threads "$T" 2>&1)
  tok_s=$(echo "$result" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1)
  output_text=$(echo "$result" | grep -v -E 'KV|Detected|tok/s|thread|model|tokenizer|Loading|Warning|Error' | head -3 | tr '\n' ' ' | cut -c1-120)
  printf "  Run %d (clean)  : %s  |  \"%s...\"\n" "$RUN" "${tok_s:-N/A}" "$output_text"
}

run_perf() {
  local T=$1
  local perf_out sys_out ram_out disk_out tok_s output_text result

  # Background system monitors — sampled during inference
  # per-CPU utilization (mpstat)
  mpstat -P ALL 1 999 > /tmp/mpstat_t${T}.txt &
  MPSTAT_PID=$!

  # RAM + disk via vmstat
  vmstat 1 999 > /tmp/vmstat_t${T}.txt &
  VMSTAT_PID=$!

  # disk I/O via iostat
  iostat -x 1 999 > /tmp/iostat_t${T}.txt &
  IOSTAT_PID=$!

  # perf stat run
  perf_out=$(perf stat -e instructions,cycles,cache-misses,cache-references,\
LLC-load-misses,LLC-loads \
    $ENGINE \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMP" \
      --threads "$T" 2>&1)

  # Kill monitors
  kill $MPSTAT_PID $VMSTAT_PID $IOSTAT_PID 2>/dev/null || true
  wait $MPSTAT_PID $VMSTAT_PID $IOSTAT_PID 2>/dev/null || true

  tok_s=$(echo "$perf_out" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1)
  output_text=$(echo "$perf_out" | grep -v -E 'KV|Detected|tok/s|thread|model|tokenizer|Loading|Warning|Error|Performance|instructions|cycles|cache|LLC|elapsed|CPUs|seconds|msec|#' | head -3 | tr '\n' ' ' | cut -c1-120)

  printf "  Run 3 (perf)   : %s  |  \"%s...\"\n" "${tok_s:-N/A}" "$output_text"
  echo ""
  echo "  ── perf stat ──────────────────────────────────────────────"
  echo "$perf_out" | grep -E 'instructions|cycles|cache|LLC|elapsed' | sed 's/^/  /'
  echo ""

  # Per-CPU utilization summary (avg of busy interval)
  echo "  ── Per-CPU utilization (sampled during inference) ──────────"
  awk '/%idle/ { next } /^[0-9]/ && $3 ~ /^[0-9]/ {
    cpu=$3; idle=$NF; util=100-idle
    sum[cpu]+=util; n[cpu]++
  } END {
    for (c in sum) printf "  CPU%s avg: %.1f%%\n", c, sum[c]/n[c]
  }' /tmp/mpstat_t${T}.txt 2>/dev/null | sort -V || echo "  (mpstat data unavailable)"
  echo ""

  # RAM snapshot (average during run)
  echo "  ── RAM during inference ────────────────────────────────────"
  awk 'NR>2 && NF==17 {
    used+=$4; free+=$5; n++
  } END {
    if(n>0) printf "  Used avg: %.0f MB   Free avg: %.0f MB\n", used/n, free/n
    else print "  (vmstat data unavailable)"
  }' /tmp/vmstat_t${T}.txt 2>/dev/null || echo "  (vmstat data unavailable)"
  echo ""

  # Disk I/O summary
  echo "  ── Disk I/O during inference ───────────────────────────────"
  awk '/^[a-z]/ { dev=$1; read+=$6; write+=$7; n++ }
  END {
    if(n>0) printf "  %s  read: %.1f kB/s  write: %.1f kB/s\n", dev, read/n, write/n
    else print "  (iostat data unavailable)"
  }' /tmp/iostat_t${T}.txt 2>/dev/null || echo "  (iostat data unavailable)"
  echo ""
}

# ── Main sweep ──
for T in 1 2 3 4 5 6 7 8; do
  if [ "$T" -lt "$RESUME_FROM" ]; then
    continue
  fi

  {
  echo "============================================================"
  echo "  T = $T threads"
  echo "============================================================"
  } | tee -a "$OUT"

  run_clean "$T" 1 2>&1 | tee -a "$OUT"
  run_clean "$T" 2 2>&1 | tee -a "$OUT"
  run_perf  "$T"    2>&1 | tee -a "$OUT"

  echo "" | tee -a "$OUT"
done

echo "Sweep complete. Results: $OUT"
