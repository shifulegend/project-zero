#!/usr/bin/env bash
# a6_replication.sh — Exact replication of Addendum A6 benchmark
#
# A6 conditions (from commit 1c892f5 + docs/PERFORMANCE_CEILING_REPORT.md §A6):
#   - Model: bitnet-b1.58-2B-4T
#   - Threads: 4 (was auto-detect at A6 time, cpu_probe.c returned 4 physical cores)
#   - max-tokens: 50  (from appendix benchmark command)
#   - temperature: 0.7 (CLI default)
#   - top-p: 0.9 (CLI default)
#   - KV strategy: KV_SLIDING_I4 @ 1024 ctx
#   - earlyoom: disabled
#   - CPU governor: performance
#   - Page cache: warm
#
# This script runs each of the 6 A6 prompts:
#   Pass 1: clean run (no monitoring overhead)
#   Pass 2: full perf stat + mpstat (per-CPU) + vmstat (RAM) + iostat (storage)

set -e
cd "$(dirname "$0")/.."

ENGINE=./adaptive_ai_engine
MODEL=models/bitnet-b1.58-2B-4T.bin
TOKENIZER=models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
THREADS=4
MAX_TOKENS=50
# NOTE: temperature and top_p are NOT passed — using CLI defaults (0.7 and 0.9)
# This matches the original A6 measurement where no sampling flags were explicit.
OUT=tests/a6_replication_results.txt

rm -f "$OUT"

# ─── Pre-flight verification ────────────────────────────────────────────────
{
echo "================================================================"
echo "  Project Zero — A6 Exact Replication"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "─── Hardware ───────────────────────────────────────────────────"
echo "  CPU   : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  Cores : $(grep -c '^processor' /proc/cpuinfo) logical, $(grep 'cpu cores' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs) physical"
echo "  RAM   : $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB total", $2/1024/1024}')"
echo "  Kernel: $(uname -r)"
echo ""
echo "─── Environment ────────────────────────────────────────────────"
echo "  CPU governor       : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'unknown')"
echo "  earlyoom           : $(systemctl is-active earlyoom 2>/dev/null || echo 'inactive/not-installed')"
echo "  CPU freq (cpu0)    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.2f GHz", $1/1000000}' || echo 'unknown')"
echo "  CPU freq (cpu7)    : $(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.2f GHz", $1/1000000}' || echo 'unknown')"
echo "  Turbo boost        : $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null | awk '{print ($1==0)?"enabled":"DISABLED"}' || echo 'unknown')"
echo ""
echo "─── RAM at benchmark start ─────────────────────────────────────"
grep "MemTotal\|MemFree\|MemAvailable\|Buffers\|^Cached\|SwapTotal\|SwapFree" /proc/meminfo | awk '{printf "  %-20s %8.1f MB\n", $1, $2/1024}'
echo ""
echo "─── Benchmark parameters ────────────────────────────────────────"
echo "  Engine   : $ENGINE  ($(ls -la $ENGINE | awk '{print $5, $6, $7, $8}'))"
echo "  Model    : $MODEL"
echo "  Threads  : $THREADS  (matches A6 — 4 physical cores, pre-E.2 auto-detect)"
echo "  max_tokens: $MAX_TOKENS  (from appendix benchmark command)"
echo "  temperature: 0.7  (CLI default — not explicitly passed)"
echo "  top-p    : 0.9  (CLI default — not explicitly passed)"
echo "  KV       : $(./adaptive_ai_engine --model $MODEL --tokenizer $TOKENIZER --prompt x --max-tokens 1 --threads $THREADS 2>&1 | grep 'KV Strategy' || echo 'see run output')"
echo ""
echo "─── A6 reference (2026-03-15, commit 1c892f5) ──────────────────"
echo "  Prompt 1: The capital of France is          → 14.76 tok/s"
echo "  Prompt 2: The boiling point of water is     → 14.75 tok/s"
echo "  Prompt 3: The largest planet in our solar   → 15.72 tok/s"
echo "  Prompt 4: Albert Einstein was born in       → 15.86 tok/s"
echo "  Prompt 5: The chemical symbol for gold is   → 15.60 tok/s"
echo "  Prompt 6: The speed of light is approximately→ 16.09 tok/s"
echo "  Average                                     → ~15.5 tok/s"
echo "================================================================"
echo ""
} | tee "$OUT"

# ─── Prompts ────────────────────────────────────────────────────────────────
PROMPTS=(
  "The capital of France is"
  "The boiling point of water is"
  "The largest planet in our solar system is"
  "Albert Einstein was born in"
  "The chemical symbol for gold is"
  "The speed of light is approximately"
)
A6_REF=(14.76 14.75 15.72 15.86 15.60 16.09)

# ─── Run helper ─────────────────────────────────────────────────────────────
run_clean() {
  local PROMPT="$1"
  $ENGINE \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    --threads $THREADS 2>&1
}

run_with_perf() {
  local PROMPT="$1" IDX="$2"

  # Background system monitors
  mpstat -P ALL 1 9999 > /tmp/a6_mpstat_${IDX}.txt 2>/dev/null &
  MPSTAT_PID=$!
  vmstat -t 1 9999     > /tmp/a6_vmstat_${IDX}.txt 2>/dev/null &
  VMSTAT_PID=$!
  iostat -xmt 1 9999   > /tmp/a6_iostat_${IDX}.txt 2>/dev/null &
  IOSTAT_PID=$!

  # Capture CPU freq every 0.5s during run
  (while true; do
    paste /sys/devices/system/cpu/cpu{0,1,2,3,4,5,6,7}/cpufreq/scaling_cur_freq 2>/dev/null | \
      awk '{for(i=1;i<=NF;i++) printf "%.2f ", $i/1000000; print ""}'
    sleep 0.5
  done) > /tmp/a6_freq_${IDX}.txt 2>/dev/null &
  FREQ_PID=$!

  # perf stat with extended counters
  perf stat -e instructions,cycles,cache-misses,cache-references,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,dTLB-load-misses,dTLB-loads,iTLB-load-misses,branch-misses,branches,context-switches,minor-faults,major-faults \
    $ENGINE \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --prompt "$PROMPT" \
      --max-tokens $MAX_TOKENS \
      --threads $THREADS 2>&1

  # Stop monitors
  kill $MPSTAT_PID $VMSTAT_PID $IOSTAT_PID $FREQ_PID 2>/dev/null || true
  wait $MPSTAT_PID $VMSTAT_PID $IOSTAT_PID $FREQ_PID 2>/dev/null || true
}

# ─── Main loop ──────────────────────────────────────────────────────────────
TOTAL_CLEAN1=0
TOTAL_CLEAN2=0
TOTAL_PERF=0
COUNT=0

for i in "${!PROMPTS[@]}"; do
  P="${PROMPTS[$i]}"
  REF="${A6_REF[$i]}"
  IDX=$((i+1))

  {
  echo "================================================================"
  printf "  Prompt %d / 6: \"%s\"\n" "$IDX" "$P"
  printf "  A6 reference : %.2f tok/s\n" "$REF"
  echo "================================================================"
  echo ""
  } | tee -a "$OUT"

  # ── Pass 1: Clean run ──────────────────────────────────────────────
  echo "── Pass 1: Clean (no monitoring overhead) ──────────────────────" | tee -a "$OUT"
  R1=$(run_clean "$P" 2>&1)
  TOK1=$(echo "$R1" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1 | awk '{print $1}')
  GEN1=$(echo "$R1" | grep -Eo '[0-9]+ tok/s' | head -1 | awk '{print $1}' || echo "?")
  OUTPUT1=$(echo "$R1" | grep -v -E '^(Project|SIMD|Model|n_|vocab|seq_len|act_type|rope|head_dim|kv_dim|KV|Detected|Detect|Loading|Warning|Loaded|dim:|hidden_dim:|n_|rope_|act_type:)' | head -5 | tr '\n' ' ')
  {
  echo "  tok/s     : $TOK1"
  echo "  Output    : $OUTPUT1"
  echo "  Raw       : $(echo "$R1" | grep -E 'KV Strategy|Detected:' | head -2)"
  echo ""
  } | tee -a "$OUT"

  # ── Pass 2: Clean run (second, same conditions) ────────────────────
  echo "── Pass 2: Clean run 2 (variance check) ────────────────────────" | tee -a "$OUT"
  R2=$(run_clean "$P" 2>&1)
  TOK2=$(echo "$R2" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1 | awk '{print $1}')
  OUTPUT2=$(echo "$R2" | grep -v -E '^(Project|SIMD|Model|n_|vocab|seq_len|act_type|rope|head_dim|kv_dim|KV|Detected|Detect|Loading|Warning|Loaded|dim:|hidden_dim:|n_|rope_|act_type:)' | head -5 | tr '\n' ' ')
  {
  echo "  tok/s     : $TOK2"
  echo "  Output    : $OUTPUT2"
  echo ""
  } | tee -a "$OUT"

  # ── Pass 3: Full perf run ──────────────────────────────────────────
  {
  echo "── Pass 3: Full perf + system monitoring ────────────────────────"
  echo "  (perf stat: IPC, LLC, TLB, branches, faults + mpstat per-CPU + vmstat RAM + iostat)"
  echo ""
  } | tee -a "$OUT"

  PERF_OUT=$(run_with_perf "$P" "$IDX" 2>&1)
  TOK3=$(echo "$PERF_OUT" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1 | awk '{print $1}')
  OUTPUT3=$(echo "$PERF_OUT" | grep -v -E '^(Project|SIMD|Model|n_|vocab|seq_len|act_type|rope|head_dim|kv_dim|KV|Detected|Detect|Loading|Warning|Loaded|dim:|hidden_dim:|n_|rope_|act_type:|Performance|Perf event)' | grep -v -E '^\s*(instructions|cycles|cache|LLC|dTLB|iTLB|branch|context|minor|major|elapsed|CPUs|seconds|msec|#|\s*$)' | head -5 | tr '\n' ' ')

  {
  echo "  ── Model output ────────────────────────────────────────────────"
  echo "  tok/s     : $TOK3"
  echo "  Output    : $OUTPUT3"
  echo ""
  echo "  ── perf stat (raw counters) ─────────────────────────────────────"
  echo "$PERF_OUT" | grep -E 'instructions|cycles|cache-misses|cache-references|LLC-load-misses|LLC-loads|LLC-store|dTLB|iTLB|branch-misses|branches|context-switches|minor-faults|major-faults|elapsed|insn per cycle|GHz' | sed 's/^/  /'
  echo ""
  echo "  ── Per-CPU utilization (mpstat average during inference) ─────────"
  awk '
    /^[0-9]/ && /all/ { next }
    /^[0-9]/ && $3 ~ /^[0-9]/ {
      cpu=$3; idle=$NF; util=100-idle
      sum[cpu]+=util; n[cpu]++
    }
    END {
      printf "  %-8s %-8s %-8s %-8s\n","CPU","avg%%","max%%","samples"
      PROCINFO["sorted_in"]="@ind_num_asc"
      for (c in sum) {
        avg=sum[c]/n[c]
        printf "  CPU%-5s %6.1f%%  %6.1f%%  %d\n", c, avg, max[c], n[c]
      }
    }
    /^[0-9]/ && $3 ~ /^[0-9]/ { cpu=$3; util=100-$NF; if(util>max[cpu]) max[cpu]=util }
  ' /tmp/a6_mpstat_${IDX}.txt 2>/dev/null || echo "  (mpstat unavailable)"
  echo ""
  echo "  ── RAM during inference (vmstat) ────────────────────────────────"
  # vmstat columns: r b swpd free buff cache si so bi bo in cs us sy id wa st
  awk 'NR>2 && NF>=16 && $4 ~ /^[0-9]+$/ {
    free+=$4; buff+=$5; cache+=$6; n++
  } END {
    if(n>0) {
      printf "  Samples: %d\n", n
      printf "  Free RAM avg   : %.0f MB\n", free/n/1024
      printf "  Buffers avg    : %.0f MB\n", buff/n/1024
      printf "  Page cache avg : %.0f MB\n", cache/n/1024
      printf "  RAM used avg   : %.0f MB  (total-free-buff-cache)\n", (15775 - free/n/1024 - buff/n/1024 - cache/n/1024)
    } else print "  (vmstat data unavailable — sample interval may have been too short)"
  }' /tmp/a6_vmstat_${IDX}.txt 2>/dev/null
  echo ""
  echo "  ── CPU frequency during inference ───────────────────────────────"
  if [ -s /tmp/a6_freq_${IDX}.txt ]; then
    echo "  (GHz per logical CPU: CPU0 CPU1 CPU2 CPU3 CPU4 CPU5 CPU6 CPU7)"
    awk 'NF==8 {
      for(i=1;i<=8;i++){sum[i]+=$i; n[i]++}
    } END {
      printf "  avg: "
      for(i=1;i<=8;i++) printf "%.2f  ", sum[i]/n[i]
      printf "\n"
    }' /tmp/a6_freq_${IDX}.txt
    echo "  First sample:"
    head -1 /tmp/a6_freq_${IDX}.txt | awk '{printf "       "; for(i=1;i<=NF;i++) printf "%.2f  ",$i; printf "\n"}'
    echo "  Last sample:"
    tail -1 /tmp/a6_freq_${IDX}.txt | awk '{printf "       "; for(i=1;i<=NF;i++) printf "%.2f  ",$i; printf "\n"}'
  else
    echo "  (freq sampling unavailable)"
  fi
  echo ""
  echo "  ── Disk I/O (iostat) ────────────────────────────────────────────"
  grep -A3 "^nvme\|^sda\|^vda" /tmp/a6_iostat_${IDX}.txt 2>/dev/null | \
    awk 'NF>5 && $1~/^[a-z]/ {r+=$4; w+=$5; util+=$NF; n++}
    END {if(n>0) printf "  Device: %s  read: %.1f kB/s  write: %.1f kB/s  util: %.1f%%\n", dev, r/n, w/n, util/n}' || \
    awk '/nvme|sda|vda/{dev=$1; r=$4; w=$5; print "  Device: "dev"  read: "r" kB/s  write: "w" kB/s"}' \
      /tmp/a6_iostat_${IDX}.txt 2>/dev/null | head -3 || echo "  (iostat data unavailable)"
  echo ""
  echo "  ── Summary for this prompt ─────────────────────────────────────"
  AVG_TOK=$(awk "BEGIN{printf \"%.2f\", ($TOK1+$TOK2+$TOK3)/3}")
  DELTA=$(awk -v avg="$AVG_TOK" -v ref="$REF" 'BEGIN{printf "%+.2f", avg - ref}')
  printf "  Pass 1 (clean) : %s tok/s\n" "$TOK1"
  printf "  Pass 2 (clean) : %s tok/s\n" "$TOK2"
  printf "  Pass 3 (perf)  : %s tok/s\n" "$TOK3"
  printf "  Average        : %s tok/s   A6 ref: %.2f   Δ: %s\n" "$AVG_TOK" "$REF" "$DELTA"
  echo ""
  } | tee -a "$OUT"

  # Accumulate averages for final summary
  TOTAL_CLEAN1=$(awk "BEGIN{printf \"%.2f\", $TOTAL_CLEAN1 + $TOK1}")
  TOTAL_CLEAN2=$(awk "BEGIN{printf \"%.2f\", $TOTAL_CLEAN2 + $TOK2}")
  TOTAL_PERF=$(awk "BEGIN{printf \"%.2f\", $TOTAL_PERF + $TOK3}")
  COUNT=$((COUNT+1))
done

# ─── Final summary ───────────────────────────────────────────────────────────
AVG1=$(awk "BEGIN{printf \"%.2f\", $TOTAL_CLEAN1 / $COUNT}")
AVG2=$(awk "BEGIN{printf \"%.2f\", $TOTAL_CLEAN2 / $COUNT}")
AVG3=$(awk "BEGIN{printf \"%.2f\", $TOTAL_PERF / $COUNT}")
OVERALL=$(awk "BEGIN{printf \"%.2f\", ($TOTAL_CLEAN1+$TOTAL_CLEAN2+$TOTAL_PERF) / ($COUNT*3)}")

{
echo "================================================================"
echo "  FINAL SUMMARY"
echo "================================================================"
echo ""
printf "  %-35s  %8s  %8s  %8s  %8s\n" "Prompt" "Pass1" "Pass2" "Perf" "A6 ref"
printf "  %-35s  %8s  %8s  %8s  %8s\n" "------" "-----" "-----" "----" "------"
} | tee -a "$OUT"

for i in "${!PROMPTS[@]}"; do
  echo "  Prompt $((i+1)) [see above]  → recorded in individual sections" | tee -a "$OUT"
done

{
echo ""
printf "  Pass 1 avg (clean)    : %s tok/s\n" "$AVG1"
printf "  Pass 2 avg (clean)    : %s tok/s\n" "$AVG2"
printf "  Pass 3 avg (perf)     : %s tok/s\n" "$AVG3"
printf "  Overall avg (all 3)   : %s tok/s\n" "$OVERALL"
printf "  A6 reference avg      : ~15.5 tok/s\n"
printf "  Delta vs A6           : %+.2f tok/s\n" "$(awk "BEGIN{printf \"%.2f\", $OVERALL - 15.5}")"
echo ""
echo "─── System state at end of benchmark ──────────────────────────"
echo "  CPU freq (cpu0): $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.2f GHz", $1/1000000}' || echo 'unknown')"
grep "MemFree\|MemAvailable" /proc/meminfo | awk '{printf "  %-20s %8.1f MB\n", $1, $2/1024}'
echo ""
echo "Results saved to: $OUT"
echo "================================================================"
} | tee -a "$OUT"
