#!/usr/bin/env bash
# do_round.sh N R : INPUT shot, then the round inside capture.mjs (OUTPUT shot + raw pty), then PROCESSING shot (prefix render).
N=$1; R=$2; EV=$HOME/evidence; REPO=/home/user/project-zero; S=$EV/screenshots; T=$EV/transcripts
CAP="node $REPO/tools/screenshots/cli/capture.mjs"
export PZ_CAPTURE_ROWS=75
$CAP $S/N${N}_r${R}_1_input.png "INPUT: round $R, N=$N threads" -- bash $EV/tools/input_view.sh $N $R >/dev/null
PZ_CAPTURE_TIMEOUT_MS=1800000 PZ_CAPTURE_RAW_OUT=$T/N${N}_r${R}_session.raw \
  $CAP $S/N${N}_r${R}_3_output.png "OUTPUT: round $R, N=$N threads" -- bash -c "bash $EV/tools/round.sh $N $R; echo exit=\$?" >/dev/null
CAPTURE_HTML=$REPO/tools/screenshots/cli/capture.html \
  node $EV/tools/snap_prefix.mjs $T/N${N}_r${R}_session.raw $S/N${N}_r${R}_2_processing.png '\[C3\] pr31 legacy' "PROCESSING: round $R, N=$N threads (frame while C3 is running)" >/dev/null
echo "$(date -u +%T) done N=$N r=$R"
