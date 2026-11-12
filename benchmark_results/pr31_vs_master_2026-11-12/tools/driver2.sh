#!/usr/bin/env bash
# N=4: 5 rounds (r5 already started by driver.sh); N=2 and N=1: 3 rounds each (owner trimmed scope for time).
while pgrep -f do_round.sh >/dev/null; do sleep 3; done
for N in 2 1; do for R in 1 2 3; do bash $HOME/evidence/tools/do_round.sh $N $R; done; done
echo ALLDONE
