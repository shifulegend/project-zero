#!/usr/bin/env bash
# round.sh N ROUND [WARMUP] — runs C1..C5 once each, strictly sequentially, for N threads.
# Greedy decode, 64 tokens, repo prompt (benchmark_results/deepseek_h2h_2026-10-29/meta.json).
N=$1; R=$2; WARM=${3:-}
EV=${EV:-$HOME/evidence}; WT=$HOME/wt
HOME_REAL=$HOME; MODEL=$HOME/models/DeepSeek-V2-Lite-Chat.Q4_K_S.gguf
PROMPT="Explain what machine learning is and how neural networks work. Discuss supervised learning, unsupervised learning, and reinforcement learning with real-world examples of each."
declare -A BIN=( [C1]="$WT/wt-master/pz" [C2]="$WT/wt-pr31/adaptive_ai_engine" [C3]="$WT/wt-pr31/adaptive_ai_engine" [C4]="$WT/wt-pr31base/adaptive_ai_engine" [C5]="$WT/wt-pr31/adaptive_ai_engine" )
declare -A EXTRA=( [C1]="" [C2]="" [C3]="--moe-threading legacy" [C4]="" [C5]="--moe-threading rowsplit" )
declare -A NAME=( [C1]="master fefa37e" [C2]="pr31 default f26f36e" [C3]="pr31 legacy f26f36e" [C4]="pr31base bf66526" [C5]="pr31 rowsplit(extra) f26f36e" )
echo "\$ date -u"; date -u
echo "MASTER=fefa37e PR31=f26f36e PR31BASE=bf66526 | model $(basename $MODEL) $(stat -c %s $MODEL) bytes"
echo "N=$N threads  round=$R ${WARM:+(WARM-UP, discarded)}  --max-tokens 64 --temperature 0"
# idle check: wait up to 3 min for 1-min load < 0.3
for i in $(seq 1 36); do L=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($L<0.3)}" && break; sleep 5; done
echo "\$ uptime"; uptime
echo "stray pz procs: $(pgrep -c -f 'adaptive_ai_engine|/pz ' || true)"
echo
for C in C1 C2 C3 C4 C5; do
  tag=${WARM:+warm_}N${N}_r${R}_$C
  echo "[$C] ${NAME[$C]}: $(basename ${BIN[$C]}) --model <gguf> --prompt <ml-prompt> --max-tokens 64 --temperature 0 --threads $N --simd vnni ${EXTRA[$C]}"
  HOME=$HOME/h_$C ${BIN[$C]} --model $MODEL --prompt "$PROMPT" --max-tokens 64 --temperature 0 --threads $N --simd vnni ${EXTRA[$C]} > $EV/transcripts/$tag.out 2> $EV/transcripts/$tag.err
  echo "    exit=$? $(grep -a '\[gen\]' $EV/transcripts/$tag.err | tail -1)"
  sync; sleep 2
done
echo; echo "\$ date -u"; date -u
