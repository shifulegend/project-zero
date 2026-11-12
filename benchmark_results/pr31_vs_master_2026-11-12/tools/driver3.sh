#!/usr/bin/env bash
# Run B (pinned --simd vnni, per-config HOME). N=4: 5 rounds; N=2 and N=1: 3 rounds each.
for R in 1 2 3 4 5; do bash $HOME/evidence/tools/do_round.sh 4 $R; done
for N in 2 1; do for R in 1 2 3; do bash $HOME/evidence/tools/do_round.sh $N $R; done; done
echo ALLDONE
