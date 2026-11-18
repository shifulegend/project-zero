# INCOMPLETE RUN

This folder is a checkpoint of a run that has not finished (or was interrupted). Its numbers are partial; do not quote them.

Resume with the same command plus `--out benchmark_results/build_compare_20261118-153555Z_quantiser-fix-ab --resume`:

    python3 tools/bench/build_compare.py --binaries old=./pz,new=/home/user/pz-fix/pz --reference new --models bitnet-2b,smollm2-135m --rounds 5 --tokens 128 --threads 4 --cooldown 5 --calibration off --screenshots none --label quantiser-fix-ab

`progress.json` says how far it got. This file is deleted when the run finishes.
