#!/usr/bin/env bash
# deepseek_bench_perf.sh — Hardware counter profiling for Project Zero vs llama.cpp
#
# Captures: IPC, cache-miss%, branch-miss%, context-switches, page-faults,
#           per-core utilization (mpstat), RAM (vmstat), I/O (iostat), RSS peak
#
# Usage: ./tools/deepseek_bench_perf.sh
#
# Requires: perf_event_paranoid=-1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="$REPO_DIR/adaptive_ai_engine"
MODEL="$REPO_DIR/models/deepseek-v2-lite-chat-Q4_K_S.gguf"
LLAMA_CLI="/home/ubuntu/llama.cpp/build/bin/llama-cli"
LLAMA_BENCH="/home/ubuntu/llama.cpp/build/bin/llama-bench"
OUT_DIR="$REPO_DIR/benchmark_results"
RAW_DIR="$OUT_DIR/perf_raw"
mkdir -p "$RAW_DIR"

PROMPT="List the first 10 prime numbers:"
MAX_TOKENS=5
BENCH_DATE=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,context-switches,page-faults"
PZ_THREADS=(${DEEPSEEK_PERF_PZ_THREADS:-1 2 3 4 5 6 7 8})
PZ_SIMDS=(${DEEPSEEK_PERF_PZ_SIMDS:-scalar avx2 avx512f vnni})
PZ_CLASSIFIERS=(${DEEPSEEK_PERF_PZ_CLASSIFIERS:-bf16 int8 int4})
LLAMA_THREADS=(${DEEPSEEK_PERF_LLAMA_THREADS:-1 2 3 4 5 6 7 8})

PERF_CSV="$OUT_DIR/perf_results.csv"
echo "date,engine,threads,simd,cls,tok_s,ipc,cycles,instructions,cache_refs,cache_misses,cache_miss_pct,branch_miss_pct,ctx_switches,page_faults,rss_peak_mb,cpu_freq_mhz" > "$PERF_CSV"

check_perf() {
  local PARANOID
  PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
  if [[ "$PARANOID" -gt 1 ]]; then
    echo "ERROR: perf_event_paranoid=$PARANOID, need -1"
    echo "Run: sudo sysctl -w kernel.perf_event_paranoid=-1"
    exit 1
  fi
}
check_perf

# Parse perf stat stderr output file
parse_perf_file() {
  local F="$1"
  [[ ! -f "$F" ]] && echo "CYCLES=0 INSTRS=0 CREF=0 CMISS=0 CMISS_PCT=0 BMISS_PCT=0 CTX=0 PAGE=0 IPC=0" && return

  local CYCLES INSTRS CREF CMISS BRANCHES BMISS CTX PAGE IPC CMISS_PCT BMISS_PCT
  CYCLES=$(grep -oP '[\d,]+(?=\s+cycles)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  INSTRS=$(grep -oP '[\d,]+(?=\s+instructions)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  CREF=$(grep -oP '[\d,]+(?=\s+cache-references)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  CMISS=$(grep -oP '[\d,]+(?=\s+cache-misses)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  BRANCHES=$(grep -oP '[\d,]+(?=\s+branches)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  BMISS=$(grep -oP '[\d,]+(?=\s+branch-misses)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  CTX=$(grep -oP '[\d,]+(?=\s+context-switches)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  PAGE=$(grep -oP '[\d,]+(?=\s+page-faults)' "$F" | head -1 | tr -d ',' 2>/dev/null || echo 0)
  IPC=$(grep -oP '#\s+\K[\d.]+(?=\s+insn per cycle)' "$F" | head -1 2>/dev/null || echo "0")

  CMISS_PCT=0
  BMISS_PCT=0
  [[ "$CREF" -gt 0 ]] && CMISS_PCT=$(echo "scale=2; $CMISS * 100 / $CREF" | bc 2>/dev/null || echo 0)
  [[ "$BRANCHES" -gt 0 ]] && BMISS_PCT=$(echo "scale=2; $BMISS * 100 / $BRANCHES" | bc 2>/dev/null || echo 0)

  echo "CYCLES=$CYCLES INSTRS=$INSTRS CREF=$CREF CMISS=$CMISS CMISS_PCT=$CMISS_PCT BMISS_PCT=$BMISS_PCT CTX=$CTX PAGE=$PAGE IPC=$IPC"
}

# Run one perf-monitored execution
# Args: engine_label threads simd cls -- cmd [args...]
run_perf_monitored() {
  local ELABEL="$1" T="$2" SIMD="$3" CLS="$4"
  shift 4
  local CMD=("$@")
  local SAFE_SIMD SAFE_CLS
  SAFE_SIMD=$(printf '%s' "$SIMD" | tr '/ ' '__')
  SAFE_CLS=$(printf '%s' "$CLS" | tr '/ ' '__')

  local TAG="${ELABEL//\//_}_T${T}_S${SAFE_SIMD}_C${SAFE_CLS}"
  local PERF_FILE="$RAW_DIR/${TAG}_perf.txt"
  local STDOUT_FILE="$RAW_DIR/${TAG}_out.txt"
  local STDERR_FILE="$RAW_DIR/${TAG}_err.txt"
  local MPSTAT_FILE="$RAW_DIR/${TAG}_mpstat.txt"
  local VMSTAT_FILE="$RAW_DIR/${TAG}_vmstat.txt"
  local IOSTAT_FILE="$RAW_DIR/${TAG}_iostat.txt"

  # CPU freq at start
  local FREQ_AVG
  FREQ_AVG=$(awk 'BEGIN{s=0;n=0}{s+=$1;n++}END{printf "%d",s/n/1000}' \
    /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)

  printf "  %-22s T=%-2s  running...\n" "$ELABEL" "$T"

  # Start monitors in background
  mpstat -P ALL 2 60 > "$MPSTAT_FILE" 2>&1 &
  local MPID=$!
  vmstat 2 60 > "$VMSTAT_FILE" 2>&1 &
  local VPID=$!
  iostat -x 2 60 > "$IOSTAT_FILE" 2>&1 &
  local IPID=$!

  # Capture perf output separately so child stderr remains available for parsing.
  perf stat -o "$PERF_FILE" -e "$PERF_EVENTS" -- "${CMD[@]}" > "$STDOUT_FILE" 2>"$STDERR_FILE" || true

  # Stop monitors
  { kill "$MPID" 2>/dev/null; kill "$VPID" 2>/dev/null; kill "$IPID" 2>/dev/null; } || true
  wait "$MPID" "$VPID" "$IPID" 2>/dev/null || true

  # Parse results
  local PDATA
  PDATA=$(parse_perf_file "$PERF_FILE")

  local TPS=0
  if [[ "$ELABEL" == "llama.cpp" ]]; then
    TPS=$(grep -oP '"avg_ts":"?\K[\d.]+' "$STDOUT_FILE" 2>/dev/null | tail -1 || true)
    [[ -z "$TPS" ]] && TPS=$(awk -F',' 'NR>1 {print $(NF-1)}' "$STDOUT_FILE" 2>/dev/null | tr -d '"' | tail -1 || true)
  else
    TPS=$(cat "$STDOUT_FILE" "$STDERR_FILE" 2>/dev/null | grep -oP '[\d.]+(?= tok/s)' | tail -1 2>/dev/null || true)
  fi
  [[ -z "$TPS" ]] && TPS=0

  local CYCLES INSTRS CREF CMISS CMISS_PCT BMISS_PCT CTX PAGE IPC
  CYCLES=$(echo "$PDATA" | grep -oP 'CYCLES=\K\d+')
  INSTRS=$(echo "$PDATA" | grep -oP 'INSTRS=\K\d+')
  CREF=$(echo "$PDATA" | grep -oP 'CREF=\K\d+')
  CMISS=$(echo "$PDATA" | grep -oP 'CMISS=\K\d+')
  CMISS_PCT=$(echo "$PDATA" | grep -oP 'CMISS_PCT=\K[\d.]+')
  BMISS_PCT=$(echo "$PDATA" | grep -oP 'BMISS_PCT=\K[\d.]+')
  CTX=$(echo "$PDATA" | grep -oP 'CTX=\K\d+')
  PAGE=$(echo "$PDATA" | grep -oP 'PAGE=\K\d+')
  IPC=$(echo "$PDATA" | grep -oP 'IPC=\K[\d.]+')

  # Peak RSS from vmstat (free memory dip)
  local RSS_MB=0
  RSS_MB=$(grep -oP 'VmRSS:\s+\K\d+' /proc/self/status 2>/dev/null | head -1 || echo 0)
  # Better: parse stdout for memory info if engine prints it
  local MEM_LINE
  MEM_LINE=$(grep -i "free ram\|rss\|memory" "$STDOUT_FILE" "$STDERR_FILE" 2>/dev/null | head -1 || echo "")

  printf "  %-22s T=%-2s  tok/s=%-6s  IPC=%-5s  cache_miss%%=%-5s  branch_miss%%=%-5s  ctx_sw=%-5s  freq=%s MHz\n" \
    "$ELABEL" "$T" "$TPS" "$IPC" "$CMISS_PCT" "$BMISS_PCT" "$CTX" "$FREQ_AVG"

  echo "$BENCH_DATE,$ELABEL,$T,$SIMD,$CLS,$TPS,$IPC,$CYCLES,$INSTRS,$CREF,$CMISS,$CMISS_PCT,$BMISS_PCT,$CTX,$PAGE,$RSS_MB,$FREQ_AVG" >> "$PERF_CSV"
  sleep 3
}

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║      Hardware Counter Profiling — with perf stat + mpstat        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid) ✓"
echo "Monitoring: cycles, instructions, IPC, cache-miss, branch-miss, ctx-sw, page-faults"
echo "Background: mpstat (per-core), vmstat (memory), iostat (disk I/O)"
echo

echo "══ [A] Project Zero engine (T={${PZ_THREADS[*]}} × SIMD={${PZ_SIMDS[*]}} × CLS={${PZ_CLASSIFIERS[*]}}) ══"
for T in "${PZ_THREADS[@]}"; do
  for SIMD in "${PZ_SIMDS[@]}"; do
    for CLS in "${PZ_CLASSIFIERS[@]}"; do
      run_perf_monitored "project-zero" "$T" "$SIMD" "$CLS" \
        "$ENGINE" --model "$MODEL" --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" --threads "$T" \
        --simd "$SIMD" --classifier "$CLS" --temperature 0.0
    done
  done
done

echo
echo "══ [B] llama.cpp (T={${LLAMA_THREADS[*]}}) ══"
for T in "${LLAMA_THREADS[@]}"; do
  run_perf_monitored "llama.cpp" "$T" "auto" "N/A" \
    "$LLAMA_BENCH" -m "$MODEL" -t "$T" -p 0 -n "$MAX_TOKENS" -r 1 -o csv
done

echo
echo "══ [C] Per-core CPU utilization samples ══"
for ENG in "project-zero" "llama.cpp"; do
  if [[ "$ENG" == "project-zero" ]]; then
    TAG="${ENG//\//_}_T4_Sauto_Cbf16"
    [[ -f "$RAW_DIR/${TAG}_mpstat.txt" ]] || TAG="${ENG//\//_}_T4_Savx512f_Cbf16"
  else
    TAG="${ENG//\//_}_T4_Sauto_CN_A"
  fi
  MPSTAT_F="$RAW_DIR/${TAG}_mpstat.txt"
  if [[ -f "$MPSTAT_F" ]]; then
    echo "  $ENG T=4 CPU utilization:"
    # Show average utilization per CPU
    awk '/^Average/ && /[0-9]/{printf "    CPU%s: usr=%.1f%% sys=%.1f%% iowait=%.1f%% idle=%.1f%%\n", $2, $3, $5, $6, $NF}' "$MPSTAT_F" 2>/dev/null || true
    echo
  fi
done

echo "══ [D] Memory pressure (vmstat sample) ══"
for ENG in "project-zero" "llama.cpp"; do
  if [[ "$ENG" == "project-zero" ]]; then
    TAG="${ENG//\//_}_T4_Sauto_Cbf16"
    [[ -f "$RAW_DIR/${TAG}_vmstat.txt" ]] || TAG="${ENG//\//_}_T4_Savx512f_Cbf16"
  else
    TAG="${ENG//\//_}_T4_Sauto_CN_A"
  fi
  VMSTAT_F="$RAW_DIR/${TAG}_vmstat.txt"
  if [[ -f "$VMSTAT_F" ]]; then
    echo "  $ENG T=4 vmstat (free, buff, cache, swpd, si, so):"
    grep -v "^$\|procs\|swpd" "$VMSTAT_F" | awk 'NR>=3 && NR<=8 {printf "    %s\n", $0}' 2>/dev/null || true
    echo
  fi
done

echo "══ [E] Disk I/O during inference (iostat) ══"
for ENG in "project-zero" "llama.cpp"; do
  if [[ "$ENG" == "project-zero" ]]; then
    TAG="${ENG//\//_}_T4_Sauto_Cbf16"
    [[ -f "$RAW_DIR/${TAG}_iostat.txt" ]] || TAG="${ENG//\//_}_T4_Savx512f_Cbf16"
  else
    TAG="${ENG//\//_}_T4_Sauto_CN_A"
  fi
  IOSTAT_F="$RAW_DIR/${TAG}_iostat.txt"
  if [[ -f "$IOSTAT_F" ]]; then
    echo "  $ENG T=4 I/O:"
    grep -E "Device|[0-9]" "$IOSTAT_F" | head -6 | awk '{printf "    %s\n", $0}' 2>/dev/null || true
    echo
  fi
done

echo "═══════════════════════════════════════════════════════════════════"
echo "  Perf profiling complete."
echo "  CSV: $PERF_CSV"
echo "  Raw: $RAW_DIR/"
echo "═══════════════════════════════════════════════════════════════════"
