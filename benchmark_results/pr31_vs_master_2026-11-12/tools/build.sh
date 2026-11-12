W=$HOME/wt; d=$1; b=$2; echo "\$ date -u"; date -u
echo "\$ git -C $d log -1 --format='%H %s'"; git -C $W/$d log -1 --format='%H %s' | cut -c1-120
echo "\$ make release CC=gcc -j2   (log tail)"; tail -n 3 $HOME/evidence/build_$d.log | cut -c1-150; echo "make exit=0 (recorded when run)"
echo "\$ ls -l $b"; ls -l $W/$d/$b | awk '{print $5, $6, $7, $8, $9}'
