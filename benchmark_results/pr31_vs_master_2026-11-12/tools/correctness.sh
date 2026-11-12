echo "\$ date -u"; date -u; echo "\$ python3 summarize.py   (correctness section)"; EV=$EV python3 $HOME/evidence/tools/summarize.py | sed -n '/^sha256/,$p'
echo; echo "NaN/crash scan over all measured runs:"; grep -aci "nan\b" $EV/transcripts/N*_r*_C*.out | awk -F: '{s+=$2} END{print "lines containing nan:", s}'
for f in $EV/transcripts/N*_r*_C*.err; do :; done; echo "nonzero exits:"; grep -ac "exit=[1-9]" $EV/transcripts/*_session.raw | awk -F: '{s+=$2} END{print s}'
