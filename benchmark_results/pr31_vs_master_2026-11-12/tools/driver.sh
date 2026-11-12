#!/usr/bin/env bash
for N in 4 2 1; do for R in 1 2 3 4 5; do bash $HOME/evidence/tools/do_round.sh $N $R; done; done
echo ALLDONE
