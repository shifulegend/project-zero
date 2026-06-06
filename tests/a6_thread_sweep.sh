#!/usr/bin/env bash
# a6_thread_sweep.sh — A6 benchmark conditions across T=1..8
#
# Runs the exact 6 A6 prompts at each thread count 1–8.
# Per thread count: 2 clean passes (all 6 prompts) + 1 perf pass (all 6 prompts).
#
# A6 conditions (commit 1c892f5):
#   max-tokens: 50  |  temperature: 0.7 (CLI default)  |  top-p: 0.9 (CLI default)
#   earlyoom: disabled  |  governor: performance  |  page cache: warm  |  no taskset
#
# Total: 8 threads × 3 passes × 6 prompts = 144 inference runs (~15–25 min)
# Resume: bash a6_thread_sweep.sh --resume=N  (start from thread count N)

set -e
cd "$(dirname "$0")/.."

ENGINE=./adaptive_ai_engine
MODEL=models/bitnet-b1.58-2B-4T.bin
TOKENIZER=models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
MAX_TOKENS=50
# temperature and top-p NOT passed — CLI defaults (0.7 / 0.9) match A6 exactly
OUT=tests/a6_thread_sweep_results.txt

RESUME_FROM=1
for arg in "$@"; do
  case "$arg" in --resume=*) RESUME_FROM="${arg#--resume=}" ;; esac
done

PROMPTS=(
  "The capital of France is"
  "The boiling point of water is"
  "The largest planet in our solar system is"
  "Albert Einstein was born in"
  "The chemical symbol for gold is"
  "The speed of light is approximately"
)
A6_REF=(14.76 14.75 15.72 15.86 15.60 16.09)

[ "$RESUME_FROM" -eq 1 ] && rm -f "$OUT"

{
echo "================================================================"
echo "  Project Zero — A6 Conditions  T=1..8 Thread Sweep"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo "  CPU    : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  Kernel : $(uname -r)"
echo "  RAM    : $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB total", $2/1024/1024}')"
echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "  earlyoom: $(systemctl is-active earlyoom 2>/dev/null || echo inactive)"
echo "  Turbo  : $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null | awk '{print ($1==0)?"enabled":"DISABLED"}')"
grep "MemTotal\|MemAvailable\|MemFree" /proc/meminfo | awk '{printf "  %-14s: %.0f MB\n",$1,$2/1024}'
echo ""
echo "  Benchmark params:"
echo "    --max-tokens $MAX_TOKENS | --temperature 0.7 (default) | --top-p 0.9 (default)"
echo "    No taskset | Page cache warm | Same 6 A6 prompts"
echo ""
echo "  A6 reference (commit 1c892f5, 2026-03-15, T=4):"
for i in "${!PROMPTS[@]}"; do
  printf "    P%d %-44s → %.2f tok/s\n" "$((i+1))" "\"${PROMPTS[$i]}\"" "${A6_REF[$i]}"
done
echo "    Average                                         → ~15.5 tok/s"
echo "================================================================"
} | tee -a "$OUT"

# Warmup
echo "" | tee -a "$OUT"
echo "=== Warming page cache ===" | tee -a "$OUT"
./adaptive_ai_engine --model "$MODEL" --tokenizer "$TOKENIZER" \
  --prompt "The capital of France is" --max-tokens "$MAX_TOKENS" --threads 4 2>&1 | \
  grep -E "KV Strategy|tok/s" | tee -a "$OUT"
echo "" | tee -a "$OUT"

run_clean() {
  local T="$1" PROMPT="$2"
  $ENGINE --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --threads "$T" 2>&1
}

run_perf() {
  local T="$1" PROMPT="$2" TIDX="$3" PIDX="$4"
  local TAG="t${T}_p${PIDX}"

  mpstat -P ALL 1 9999 > /tmp/a6sw_mpstat_${TAG}.txt 2>/dev/null &
  local MPID=$!
  vmstat 1 9999 > /tmp/a6sw_vmstat_${TAG}.txt 2>/dev/null &
  local VPID=$!
  (while true; do
    paste /sys/devices/system/cpu/cpu{0,1,2,3,4,5,6,7}/cpufreq/scaling_cur_freq 2>/dev/null | \
      awk '{for(i=1;i<=NF;i++) printf "%.2f ",$i/1000000; print ""}'; sleep 0.5
  done) > /tmp/a6sw_freq_${TAG}.txt 2>/dev/null &
  local FPID=$!

  perf stat -e instructions,cycles,cache-misses,cache-references,LLC-load-misses,LLC-loads,dTLB-load-misses,branch-misses,branches,context-switches,minor-faults,major-faults \
    $ENGINE --model "$MODEL" --tokenizer "$TOKENIZER" \
      --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --threads "$T" 2>&1

  kill $MPID $VPID $FPID 2>/dev/null || true
  wait $MPID $VPID $FPID 2>/dev/null || true
}

extract_tok() { echo "$1" | grep -Eo '[0-9]+\.[0-9]+ tok/s' | head -1 | awk '{print $1}'; }
extract_out() {
  echo "$1" | grep -v -E "^(Project|SIMD|Model|n_|vocab|seq_len|act_type|rope|head_dim|kv_dim|dim:|hidden_dim:|KV|Detected|Loading|Loaded|Warning|^\s*$)" | \
    grep -v "^  [a-z_]*:" | head -3 | tr '\n' ' ' | cut -c1-100
}
extract_ipc() { echo "$1" | grep "insn per cycle" | grep -Eo '[0-9]+\.[0-9]+' | head -1; }
extract_llc() { echo "$1" | grep "LLC-load-misses" | grep -Eo '[0-9]+\.[0-9]+%' | head -1; }
extract_wall() { echo "$1" | grep "seconds time elapsed" | awk '{print $1}'; }

for T in 1 2 3 4 5 6 7 8; do
  [ "$T" -lt "$RESUME_FROM" ] && continue

  {
  echo "================================================================"
  echo "  T = $T threads"
  echo "================================================================"
  } | tee -a "$OUT"

  # Per-thread summary accumulator
  SUM_C1=0; SUM_C2=0; SUM_PF=0
  declare -A C1_TOKS C2_TOKS PF_TOKS

  # ── Pass 1: all 6 prompts clean ──────────────────────────────────
  echo "── Pass 1 (clean, no monitoring overhead) ──────────────────────" | tee -a "$OUT"
  for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    R=$(run_clean "$T" "$P" 2>&1)
    TOK=$(extract_tok "$R")
    OUT_TEXT=$(extract_out "$R")
    C1_TOKS[$i]="$TOK"
    SUM_C1=$(awk -v s="$SUM_C1" -v v="${TOK:-0}" 'BEGIN{printf "%.2f",s+v}')
    printf "  P%d %-44s | %s tok/s\n" "$((i+1))" "\"$P\"" "${TOK:-N/A}" | tee -a "$OUT"
    printf "     Answer: \"%s\"\n" "$OUT_TEXT" | tee -a "$OUT"
  done

  AVG_C1=$(awk -v s="$SUM_C1" 'BEGIN{printf "%.2f",s/6}')
  printf "  ─── Pass 1 avg: %s tok/s ─────────────────────────────────────\n" "$AVG_C1" | tee -a "$OUT"
  echo "" | tee -a "$OUT"

  # ── Pass 2: all 6 prompts clean ──────────────────────────────────
  echo "── Pass 2 (clean run 2 — variance check) ────────────────────────" | tee -a "$OUT"
  for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    R=$(run_clean "$T" "$P" 2>&1)
    TOK=$(extract_tok "$R")
    C2_TOKS[$i]="$TOK"
    SUM_C2=$(awk -v s="$SUM_C2" -v v="${TOK:-0}" 'BEGIN{printf "%.2f",s+v}')
    printf "  P%d %-44s | %s tok/s\n" "$((i+1))" "\"$P\"" "${TOK:-N/A}" | tee -a "$OUT"
  done

  AVG_C2=$(awk -v s="$SUM_C2" 'BEGIN{printf "%.2f",s/6}')
  printf "  ─── Pass 2 avg: %s tok/s ─────────────────────────────────────\n" "$AVG_C2" | tee -a "$OUT"
  echo "" | tee -a "$OUT"

  # ── Pass 3: all 6 prompts with perf ──────────────────────────────
  echo "── Pass 3 (full perf + CPU/RAM/freq monitoring) ─────────────────" | tee -a "$OUT"
  for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    PERF_R=$(run_perf "$T" "$P" "$T" "$((i+1))" 2>&1)
    TOK=$(extract_tok "$PERF_R")
    PF_TOKS[$i]="$TOK"
    SUM_PF=$(awk -v s="$SUM_PF" -v v="${TOK:-0}" 'BEGIN{printf "%.2f",s+v}')
    IPC=$(extract_ipc "$PERF_R")
    LLC=$(extract_llc "$PERF_R")
    WALL=$(extract_wall "$PERF_R")

    printf "  P%d %-44s | %s tok/s  IPC:%s  LLC-miss:%s  wall:%ss\n" \
      "$((i+1))" "\"$P\"" "${TOK:-N/A}" "${IPC:-?}" "${LLC:-?}" "${WALL:-?}" | tee -a "$OUT"

    # Full raw perf counters
    {
    echo "     Raw perf stat:"
    echo "$PERF_R" | grep -E 'instructions|cycles|cache-misses|cache-references|LLC-load-misses|LLC-loads|dTLB-load-misses|branch-misses|branches|context-switches|minor-faults|major-faults|seconds time elapsed|insn per cycle' | \
      grep -v "%" | sed 's/^/     /' | head -15
    echo "$PERF_R" | grep "insn per cycle" | sed 's/^/     /'
    echo ""

    # Per-CPU utilization
    echo "     Per-CPU util (avg%/max%):"
    awk '
      /^[0-9]/ && $3 ~ /^[0-9]/ && $3 != "all" {
        cpu=$3; util=100-$NF; sum[cpu]+=util; n[cpu]++
        if(util>mx[cpu]) mx[cpu]=util
      }
      END {
        printf "     "
        for(c=0;c<8;c++) {
          if(n[c]>0) printf "CPU%d:%4.0f%%/%4.0f%%  ",c,sum[c]/n[c],mx[c]
        }
        printf "\n"
      }
    ' /tmp/a6sw_mpstat_t${T}_p$((i+1)).txt 2>/dev/null || echo "     (mpstat unavailable)"

    # CPU freq
    if [ -s /tmp/a6sw_freq_t${T}_p$((i+1)).txt ]; then
      awk 'NF==8{for(i=1;i<=8;i++){s[i]+=$i;n[i]++}}
        END{printf "     Freq avg (GHz): "; for(i=1;i<=8;i++) printf "CPU%d:%.2f  ",i-1,s[i]/n[i]; printf "\n"}' \
        /tmp/a6sw_freq_t${T}_p$((i+1)).txt 2>/dev/null
    fi

    # RAM
    awk 'NR>2 && NF>=16 && $4~/^[0-9]+$/{f+=$4;b+=$5;c+=$6;n++}
      END{if(n>0) printf "     RAM: free=%.0fMB  cache=%.0fMB  used=%.0fMB\n",f/n/1024,c/n/1024,15775-f/n/1024-b/n/1024-c/n/1024}' \
      /tmp/a6sw_vmstat_t${T}_p$((i+1)).txt 2>/dev/null

    echo ""
    } | tee -a "$OUT"
  done

  AVG_PF=$(awk -v s="$SUM_PF" 'BEGIN{printf "%.2f",s/6}')
  OVERALL=$(awk -v c1="$SUM_C1" -v c2="$SUM_C2" -v pf="$SUM_PF" 'BEGIN{printf "%.2f",(c1+c2+pf)/18}')
  DELTA_A6=$(awk -v o="$OVERALL" 'BEGIN{printf "%+.2f",o-15.5}')

  {
  echo "── T=$T Summary ─────────────────────────────────────────────────"
  printf "  Pass 1 avg (clean)  : %s tok/s\n" "$AVG_C1"
  printf "  Pass 2 avg (clean)  : %s tok/s\n" "$AVG_C2"
  printf "  Pass 3 avg (perf)   : %s tok/s\n" "$AVG_PF"
  printf "  Overall avg         : %s tok/s   vs A6@T=4 (~15.5):  %s tok/s\n" "$OVERALL" "$DELTA_A6"
  echo ""
  printf "  Per-prompt comparison (avg of 3 passes vs A6 ref):\n"
  printf "  %-3s %-44s %8s %8s %8s %8s %8s\n" "P#" "Prompt" "P1" "P2" "P3" "Avg" "A6 ref"
  for i in "${!PROMPTS[@]}"; do
    V1="${C1_TOKS[$i]:-0}"; V2="${C2_TOKS[$i]:-0}"; V3="${PF_TOKS[$i]:-0}"
    AVG=$(awk -v a="$V1" -v b="$V2" -v c="$V3" 'BEGIN{printf "%.2f",(a+b+c)/3}')
    printf "  P%d %-44s %8s %8s %8s %8s %8.2f\n" \
      "$((i+1))" "\"${PROMPTS[$i]}\"" "$V1" "$V2" "$V3" "$AVG" "${A6_REF[$i]}" | tee -a "$OUT"
  done
  echo ""
  } | tee -a "$OUT"

  unset C1_TOKS C2_TOKS PF_TOKS
  declare -A C1_TOKS C2_TOKS PF_TOKS
done

# ─── Final cross-thread summary ──────────────────────────────────────────────
{
echo "================================================================"
echo "  FINAL CROSS-THREAD SUMMARY"
echo "================================================================"
echo ""
echo "  (Per-thread overall averages extracted from per-thread summaries above)"
echo "  See individual T=N sections for per-prompt breakdown and raw perf data."
echo ""
echo "  Key reference points:"
echo "    A6 baseline (T=4, 2026-03-15)     :  ~15.5 tok/s avg"
echo "    §E.2 T=6 sweet spot (2026-03-17)   :   16.09 tok/s"
echo "    §E.2 T=8 cliff      (2026-03-17)   :    2.53 tok/s"
echo ""
echo "  Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results file: $OUT"
echo "================================================================"
} | tee -a "$OUT"
