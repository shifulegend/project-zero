#!/usr/bin/env bash
# Shows the exact script and identities for a round (INPUT screenshot).
N=$1; R=$2; WT=$HOME/wt
echo "\$ date -u"; date -u
echo "\$ git rev-parse --short HEAD   # per worktree"
for w in wt-master wt-pr31 wt-pr31base; do printf '%-12s %s\n' $w "$(git -C $WT/$w rev-parse --short HEAD) $(git -C $WT/$w log -1 --format=%s | cut -c1-60)"; done
echo "\$ ls -l model"; ls -l $HOME/models/DeepSeek-V2-Lite-Chat.Q4_K_S.gguf | awk '{print $5, $9}'
echo "round: N=$N threads, round $R; configs C1..C5 run once each, in order"
echo "\$ cat round.sh"
cat $HOME/evidence/tools/round.sh | cut -c1-175
