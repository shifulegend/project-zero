# Generation slot released before the final write — 2026-11-11 17:51–18:24 UTC

Fix: `handle_chat_completions` (`src/api/http_server.c`) now calls `release_generation_slot()`
before the last SSE event / JSON body is written, instead of after it. Before, a client that sent
its next request as soon as it had read the answer could get `429 "Another generation is already
in flight"` while nothing was running.

- BASE = `master` `757b6b6`, FIX = branch `fix/release-slot-before-send` commit `19566e6`. R1–R4 FIX
  prompts show `19b8156` and G1/G2 `757b6b6-dirty`: earlier states of the same source (only a test
  comment and the commit message changed). R5 builds are scratch copies with `usleep(5000)` added
  just before the release (BASE) or after the send (FIX), hence `-dirty`.
- Host: 4 vCPU x86-64 (AVX-512 VNNI) cloud container, gcc 13.3, clang 18.1. Model:
  SmolLM2-135M-Instruct Q8_0 (SHA-256 `5a1395…bba83`, as in `src/cli/model_catalog.c`).
- Race client: `race.py <pz-dir> <port> <N> <json|stream_done>` (server `--threads 1`, N strictly
  sequential requests, `max_tokens` 2, temperature 0; `stream_done` stops reading at `data: [DONE]`).
  "hogs" = two `timeout 200 python3 -c 'while True: pass'` processes for the run's duration.
- Each run is one `script -q -t` session (`transcripts/<ID>.typescript` + `.timing`). Screenshots
  are `tools/screenshots/cli/capture.mjs` renders of that recorded pty stream cut at three points:
  `_INPUT` (prompt with `date -u`, commit and command), `_PROCESSING` (middle of the run by the
  timing file), `_OUTPUT` (all of it, ending with `exit=`). Runs that print only at the end show the
  command still running in `_PROCESSING`.

## Race experiment (429 count among N sequential requests)

| ID | what | BASE | FIX | files | exit code |
|---|---|---|---|---|---|
| R1 | json, N=300, idle | 0/300 | 0/300 | `R1_json_idle_{BASE,FIX}*` | 0 / 0 |
| R2 | json, N=300, 2 CPU hogs | 0/300 | 0/300 | `R2_json_hogs_{BASE,FIX}*` | 0 / 0 |
| R3 | stream_done, N=200, idle | 0/200 | 0/200 | `R3_stream_idle_{BASE,FIX}*` | 0 / 0 |
| R4 | stream_done, N=200, 2 CPU hogs | 0/200 | 0/200 | `R4_stream_hogs_{BASE,FIX}*` | 0 / 0 |
| R5a | json, N=100, `usleep(5000)` in the post-send gap | 88/100 | 0/100 | `R5a_usleep_json_{BASE,FIX}*` | 0 / 0 |
| R5b | stream_done, N=100, `usleep(5000)` | 92/100 | 0/100 | `R5b_usleep_stream_{BASE,FIX}*` | 0 / 0 |

On this host the unwidened window rarely fires with `--threads 1` (R1–R4: none). With the default
thread count it fires on about 1% of streaming requests: an unrecorded probe of the BASE server gave
3, 3 and 1 429s in three rounds of 200 `stream_done` requests. Widening the gap (R5) shows it is the
post-send window: 88–92% on BASE, 0 on FIX.

## Regression guard (`tests/e2e_api_concurrency.sh`, new checks: 20 sequential streaming + 30 non-streaming)

| ID | what | result | files | exit code |
|---|---|---|---|---|
| G1 | BASE build, new script | 7 passed, 1 failed (streaming 19/20, one 429) | `G1_guard_BASE*` | 1 |
| G2 | FIX build | 8 passed | `G2_guard_FIX*` | 0 |
| G3 | BASE build, again | 8 passed (window not hit) | `G3_guard_BASE*` | 0 |
| G4 | FIX build, again | 8 passed | `G4_guard_FIX*` | 0 |
| G5 | BASE + `usleep(5000)` | 6 passed, 2 failed (streaming 2/20, json 4/30 got 200) | `G5_guard_BASE_usleep*` | 1 |

## Verification matrix (FIX `19566e6`)

| ID | what | result | files | exit code |
|---|---|---|---|---|
| M01 | `make release CC=clang` | built | `M01_clang_release*` | 0 |
| M02 | `make test CC=clang` | link failed: clang 18 ASan runtime not installed on the host (`libclang_rt.asan-x86_64.a` missing) | `M02_clang_test*` | 2 |
| M02b | `make test CC=clang` after `apt-get install libclang-rt-18-dev` | All tests passed | `M02b_clang_test*` | 0 |
| M03 / M03b | `make debug CC=clang` (before / after the runtime install) | built | `M03_clang_debug*`, `M03b_clang_debug*` | 0 / 0 |
| M04 | `make test-tsan CC=gcc` | every TSan test binary passed, "TSan: no races reported" | `M04_tsan_gcc*` | 0 |
| M05 | `make coverage CC=gcc` | lines 51.2%, functions 57.9% | `M05_coverage_gcc*` | 0 |
| M06 | CMake build as in ci.yml | `build_cmake/pz` built | `M06_cmake*` | 0 |
| M07–M09 | `make release` / `make test` / `make debug`, CC=gcc | All tests passed | `M07_*`, `M08_*`, `M09_*` | 0 |
| M10 | `PZ_TEST_MODEL=… build/tests/test_batch_scheduler`, `test_prompt_injection` | both pass (11/11 injection) | `M10_batch_sched_injection*` | 0 |
| M11 | `make release CC=gcc` again (e2e need the release `./pz`) | `Project Zero v0.2.0-11-g19566e6` | `M11_gcc_release_again*` | 0 |
| M12 | `tests/golden_regression.sh` | 26 passed, 0 failed | `M12_golden_regression*` | 0 |
| M13 | `tests/e2e_json_mode.sh` | 17 passed, 0 failed | `M13_e2e_json_mode*` | 0 |
| M14 | `tests/e2e_prompt_cache.sh` | all PASS | `M14_e2e_prompt_cache*` | 0 |
| M15 | `tests/e2e_api_concurrency.sh` | 8 passed, 0 failed | `M15_e2e_api_concurrency*` | 0 |
| M16 | `tests/e2e_parallel.sh` | 6 passed, 0 failed | `M16_e2e_parallel*` | 0 |
| M17 | `tools/changelog.py check --require-unreleased`, `sync_ai_adapters.py --check` | ok | `M17_changelog_adapters*` | 0 |
| M18 | `sh tools/test_install.sh ./pz` | 24 passed, 0 failed | `M18_test_install*` | 0 |
| E1 | answer equivalence BASE vs FIX (`equivalence.py`: 2 prompts × stream on/off, temperature 0, seed 1) | 4/4 identical text, usage and status | `E1_equivalence*` | 0 |
| S0 | capture tool smoke test (`echo hello; date -u`) | ok | `S0_smoke*` | 0 |

`MANIFEST.sha256` lists every file in this folder (paths relative to it).
