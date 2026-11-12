#!/usr/bin/env bash
# Run B, trimmed for time: N=4 rounds 1-3 (r1 started by driver3), N=2 rounds 1-2, N=1 rounds 1-2.
while pgrep -f "do_round.sh 4 2" >/dev/null; do sleep 3; done
for R in 3; do bash $HOME/evidence/tools/do_round.sh 4 $R; done
for N in 2 1; do for R in 1 2; do bash $HOME/evidence/tools/do_round.sh $N $R; done; done
echo ALLDONE
