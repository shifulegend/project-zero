#!/usr/bin/env bash
# Run B, final trim for the owner's time limit: N=2 rounds 1-2 (r1 started earlier), N=1 round 1.
while pgrep -f "do_round.sh 2 1" >/dev/null; do sleep 3; done
bash $HOME/evidence/tools/do_round.sh 2 2
bash $HOME/evidence/tools/do_round.sh 1 1
echo ALLDONE
