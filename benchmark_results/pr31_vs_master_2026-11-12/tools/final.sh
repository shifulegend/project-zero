echo "\$ date -u"; date -u; echo "\$ python3 summarize.py   (parses the raw .err transcripts)"; EV=$EV python3 $HOME/evidence/tools/summarize.py | sed -n '1,/^sha256/p' | sed '$d'
