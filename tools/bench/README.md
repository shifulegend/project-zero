# tools/bench -- benchmark and calibration test scripts

Independent, flag-driven scripts. Each one runs the hardware profile, runs its test with the full pz
branding, captures live-terminal screenshots (a real pty rendered with xterm.js, never an image drawn
from a log), stamps them with the UTC date and time, and saves everything in one folder:

    benchmark_results/<kind>_<YYYYMMDD-HHMMSSZ>[_<label>]/
      command.txt        the exact command line
      system_info.txt    CPU, ISA flags, memory, kernel, lscpu, pz commit and binary checksums, engine commits
      hardware_profile.txt   the pz hardware-profile box as printed
      summary.md / summary.csv / meta.json   results (per-run rows, best/mean/stdev, ratios)
      raw/               verbatim output of every run (DATE / HOST_CPU / CMD / PROMPT header) and raw terminal bytes
      screenshots/       <UTC stamp>_<name>.png, the caption repeats date/time, CPU, commit and command
      run.log            what the script did, with UTC time stamps

Results are plain files in the working tree. Nothing here commits, pushes or edits tracked files.

| Script | Job |
|---|---|
| `hardware_profile.py` | host details and the branded pz hardware profile (banner, wordmark, profile box) |
| `calibration_test.py` | calibration sweeps and stability, SIMD tier A/B, classifier A/B, unit tests |
| `h2h_bench.py` | head-to-head decode benchmark: pz, llama.cpp, bitnet.cpp, ik_llama.cpp |
| `build_compare.py` | interleaved A/B of pz binaries (native vs portable, before vs after a change) |
| `setup_engines.py` | fetch and build the reference engines (also patched variants), fetch and convert the models; `--smoke` checks each engine answers correctly |
| `gate.py` | the correctness gate used by every script above (a module, not run by hand) |

Every script takes `--dry-run` (print the plan and exit), `--out`, `--label`, `--pz`, `--screenshots
none|summary|all`, `--shot-rows`, and `--calibration isolated|off|existing`. `isolated` (default) gives pz
its own empty HOME and cache directory inside the run folder, so the first run calibrates like a
first-time user's and your `~/.project-zero` is never read or written; `off` passes `--no-calibrate`;
`existing` uses the real one. Run `<script> --help` for every flag.

## What is needed, and how to get it

Nothing is assumed installed. Check with `python3 tools/bench/setup_engines.py --list`.

| Element | Needed by | How it is obtained |
|---|---|---|
| `pz` | all | `make release` (native) or `make dist` (portable); `build_compare.py --build native,portable` does both |
| models (`models/*.gguf`) | all | `setup_engines.py --models ...` (downloads, checks the size; `--verify` checks sha256) |
| llama.cpp, bitnet.cpp, ik_llama.cpp | `h2h_bench.py` only | `setup_engines.py --engines ...` (clone, pinned commit, cmake build) |
| node, Chromium (Playwright), `script(1)` | screenshots | already present here; `npm install` for xterm.js runs on first use |
| cmake, clang/g++, git, curl | engine builds | system packages |

`setup_engines.py` and `h2h_bench.py` decide, per run, whether to use what is already on disk or fetch it:

- `--engine-fetch never|missing|always` and `--model-fetch never|missing|always` (`never` is the default
  for `h2h_bench.py`: the plan lists what is missing and prints the command that gets it);
- `--ref pinned|latest|<sha>`: `pinned` builds the commit of the published 2026-10-29 sweep
  (`engines.json`: llama.cpp `ff5a233`, bitnet.cpp `0b341e5`, ik_llama.cpp `06d729c`);
- `--engine-dir` (default `$PZ_BENCH_ENGINE_DIR` or `~/ext`) and `--models-dir` (default `models`).

Model files, sizes and sources are in `models.json`; `deepseek-v2-lite` is 9.5 GB and is only included
when named or with `--models all`.

## The correctness gate

A speed number is only meaningful when the same binary and flags also compute the model correctly, so every
timed script runs a gate first (`gate.py`; defaults in `gate.json`, frozen text in `gate_text.txt`):

1. **smoke**: a short greedy prompt ("The capital of France is") must contain the expected answer;
2. **quality**: perplexity on the frozen text (llama-perplexity's method; `pz --perplexity` for pz) must be
   within a tolerance of the **reference engine** of that model (`models.json` -> `gate.reference`; for
   `build_compare` and `calibration_test` the reference is the first binary / the default tier).

| verdict | meaning |
|---|---|
| `PASS` / `WARN` | within the tolerance / within 2x the tolerance |
| `INVALID` | outside that, wrong smoke answer, or the engine failed |
| `UNVERIFIED` | no reference result to compare with |
| `REF` | the reference itself (it still has to pass the smoke check) |

Flags on every script: `--gate off|warn|strict` (default `strict`: a cell that fails the gate is neither timed nor
screenshotted (a speed for a wrong computation means nothing), is listed in `summary.md` as not timed, and the exit
status is 1; `gate.json` `time_invalid` changes that; `warn` only marks them),
`--gate-tolerance PCT` (default 1.0), `--gate-text`, `--gate-chunks`, `--gate-ctx`, `--gate-reference VARIANT`,
and `--gate-only` (run just the gate; with `--out FOLDER --resume` it adds verdicts to an existing run
folder: `gate.json`, the `gate` column of `summary.csv`, `summary.md`, the chart). Every row, chart note and
screenshot caption carries the verdict.

Timed scripts also refuse a pz binary built from uncommitted state (its version ends in `-dirty`): nobody can
reproduce its numbers. Rebuild from a clean tree (`make clean && make release`), or pass `--allow-dirty-binary`
(the run is then marked in `meta.json`). Set `require_clean_binary` in `gate.json` to change the default. Engines that need a modified source tree are listed in
`engines.json` with the recorded patch (`bitnetcpp-fix`) and are always labelled "modified".
Why this exists: `docs/rca/2026-11-18-ternary-output-correctness.md`.

## Common runs

```bash
# 1. what do I have / what is missing
python3 tools/bench/setup_engines.py --list
python3 tools/bench/setup_engines.py --engines llamacpp,ik --models smollm2-1.7b-q4km --dry-run

# 2. the branded hardware profile with first-run calibration
python3 tools/bench/hardware_profile.py

# 3. calibration: stability over 5 forced calibrations; tier and classifier A/B
python3 tools/bench/calibration_test.py --tests sweeps --calibrations 5 --label stability
python3 tools/bench/calibration_test.py --tests tiers,classifiers --models bitnet-2b --rounds 5 --tiers default,vnni,avx2,scalar

# 4. pz alone, selected models and classifiers
python3 tools/bench/h2h_bench.py --engines pz --models bitnet-2b --pz-classifiers default,int4 --threads 1-4 --rounds 3

# 5. the full head-to-head (fetches what is missing; long)
python3 tools/bench/h2h_bench.py --engines all --models all --threads 1-4 --rounds 3 \
        --engine-fetch missing --model-fetch missing --label full

# 6. native vs portable build, with the per-step split and the ISA check
python3 tools/bench/build_compare.py --build native,portable --rounds 5 --label native-vs-portable

# 7. the published sweep (everything the published sweeps had, gate included)
python3 tools/bench/h2h_bench.py --preset published --engine-fetch missing --model-fetch missing

# 8. only the gate, on an existing run folder; or a quick install check of the engines
python3 tools/bench/h2h_bench.py --preset published --gate-only --out benchmark_results/<folder> --resume
python3 tools/bench/setup_engines.py --engines llamacpp,bitnetcpp,ik --engine-fetch never --smoke
```

## Method (the same as the published sweeps unless a flag says otherwise)

One process at a time; temperature 0; fixed-length runs (raw prompt, EOS ignored, exactly `--tokens`
generated, the same work for every engine) unless `--chat`; best / mean / stdev per (model, variant,
threads). Variants run **interleaved** by default (every variant once per round, the starting variant
rotating) so thermal drift and background load hit all of them alike; `--order blocks` reproduces the
published one-engine-at-a-time layout. A screenshot run is a separate, shorter run in a real terminal (it
shows the live tok/s display and the banner); its tok/s is recorded in `meta.json` and never mixed into the
statistics.

The earlier dated scripts (`tools/run_h2h_sweep_*.sh`, `tools/bench_sweep_seq.py`) stay as the record of
how the published numbers were produced; `benchlib.py` reuses `bench_sweep_seq.py`'s output parsers; the engine and model registries, command builders and variants live in `benchlib_variants.py`, which `benchlib` re-exports.
