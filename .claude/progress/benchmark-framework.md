# Progress: independent benchmark / calibration test scripts (PR #74)

Branch `claude/great-cannon-ch0rtd`. Rule from the user: **no merge to master until the scripts
reproduce at least everything in the published sweeps' screenshot sets** (sweep_h2h_2026-10-19 / -10-29,
sweep_2026-10-18): per-thread screenshots at full length, throughput-vs-threads chart, pz default + int4,
every engine, 3 rounds. Everything runs **one process at a time**. Remove this file when the PR merges.

## Stories and acceptance
1. As a maintainer I run one command and get the hardware profile, branded live-terminal screenshots
   stamped with UTC date/time, and the raw output saved. Check: `h2h_bench.py --preset published`.
2. As a maintainer I choose full or partial runs (engines, models, threads, rounds, classifiers, fetch or
   reuse engines/models). Check: `tools/bench/README.md`, `--dry-run`.
3. As a reviewer I can trust partial results are marked and a restart loses at most one run.
   Check: `PARTIAL.md` + `progress.json` in the run folder, `--resume` skips recorded runs and screenshots.

## State (update at each checkpoint)
- Code: committed and pushed (preset, chart, mamba model, resume fixes). CI green on `56f9546`.
- Official sweep: folder `benchmark_results/h2h_20261118-053147Z_published-sweep/`, 180 runs + 60 screenshots.
  Container restarts kill background jobs (seen at 06:28 UTC) — just resume.
- Resume command (run in the background, tracked):
  `python3 tools/bench/h2h_bench.py --preset published --label published-sweep --out benchmark_results/h2h_20261118-053147Z_published-sweep --resume`
- The running sweep process was started before the marker code existed, so `PARTIAL.md`/`progress.json` in its
  folder were written by hand: delete both by hand when it finishes (resumed processes use the new code).
- After it finishes: compare with the old
  sweeps cell by cell; run `python3 tools/bench/test_bench.py` (deferred while the sweep ran so tests do not
  disturb measurements); commit the folder, push, recheck CI, then the user decides on the merge.

## Open items
- Unit tests for PARTIAL.md / progress.json / screenshot resume (add after the sweep, not during it).
- Re-run `calibration_test.py` / `build_compare.py` only if their screenshot sets are judged short of the minimum.
