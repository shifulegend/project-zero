# Test Execution Results — 2026-09-18

> Executes `docs/reports/TEST_PLAN_2026-09-18.md`'s Phase 1 (P0) items, per explicit user
> confirmation ("Yes do it. Do full testing"). Newest work at top of each section's history lives in
> `docs/ai/mistakes.md`/`docs/ai/change-trace.md`; this doc is the consolidated summary and final
> disposition.

## Headline result

**6 real, confirmed bugs found and fixed** (2 in production code paths that ship, 1 build-system, 1
concurrency, 1 sandbox-security, plus the open bug from the previous session fully root-caused and
closed) — via differential testing against a real reference implementation, not hand-written examples.
Two of the test tools themselves initially had bugs in their own assumptions, corrected before being
trusted (documented, not swept under the rug — see `mistakes.md`).

Final state: `release`/`test`/`debug` all green on **both gcc and clang**, plus a clean `make test-tsan`
run (0 races), from a from-scratch clean build. Working tree clean, all commits pushed to
`claude/architecture-progress-review-ahv93e`.

## P0 items executed

| # | Item | Result | Commit |
|---|---|---|---|
| 1 | Build llama.cpp/ggml reference + install tools | Done (llama.cpp pinned `4fea119`, valgrind/clang-tidy/cppcheck/lcov installed) | — |
| 2 | TS-1.2 block-size/geometry assertions | **153/153 pass** — `tests/test_gguf_geometry.c` | (part of below) |
| 3 | TS-1.1 differential dequant vs real ggml | **Found + fixed: IQ4_NL nibble-packing bug** | `e5b6c7f` |
| 4 | TS-4 open-bug RCA (2026-09-17 degenerate output) | **Closed** — same root cause as #3 | `d37fd42` |
| 5 | TS-5.1 golden-output regression | **26/26 pass** — deterministic across all SIMD×thread combos | `435ed64` |
| 6 | TS-5.3 TSan (previously missing entirely) | **Found + fixed: real `ThreadPool.shutdown` data race** | `b2c467f` |
| 7 | TS-3.1 sandbox adversarial corpus | **Found + fixed: exit-code race + symlink-escape gap** | `ff83c25` |
| 8 | TS-2.1 grammar differential fuzzing vs `json.loads` | **Found + fixed: leading-zero numbers accepted** | `0c301ca` |
| 9 | TS-2.2 token-masking, real 49152-token vocab | **637/637 pass** — no production bugs, 2 test-tool fixes | `5356bf2` |
| 10 | TS-2.3/2.4 E2E JSON-mode + non-JSON regression | **17/17 pass** — no production bugs, 2 test-tool fixes | `f72da37` |

Also found and fixed along the way, not originally its own line item: a Makefile bug where
`make debug` run after `make release` silently linked stale, non-sanitized objects (0 ASan/UBSan
symbols despite exit 0) — see `e5b6c7f`.

## Bugs found — detail (see `docs/ai/mistakes.md` 2026-09-18 entries for full RCA on each)

1. **IQ4_NL dequant nibble-packing** (`src/core/gguf_quant.c`) — used interleaved-pair packing instead
   of ggml's split-half layout. Root cause of the entire 2026-09-17 "degenerate repeating-token output"
   bug for `Q3_K_S`/`IQ4_XS` GGUF files (both dominated by `IQ4_NL`-quantized weights). Verified fix via
   bit-exact differential test against real ggml (20/20 types) and real-model re-repro (both files now
   produce coherent, correct output, e.g. "The capital of France is Paris.").
2. **Makefile build-variant staleness** — `build/%.o` paths shared across `release`/`debug`/`dist`
   variants; Make only tracks mtimes, not flags, so switching variants without `make clean` silently
   linked mismatched objects. A `make debug` run right after `make release` produced a binary with
   **zero** ASan/UBSan symbols despite exiting 0. Fixed with a `build/.variant` stamp that forces a
   clean rebuild on variant switch.
3. **`ThreadPool.shutdown` data race** (`include/threading/thread_pool.h`) — a plain `bool` read
   without the mutex in the lock-free spin fast path, written under the mutex elsewhere. Real UB (not
   just a TSan nag): could in principle hang `threadpool_destroy()`. Fixed with `atomic_bool` +
   explicit acquire/release, matching the struct's other atomics. TSan was entirely missing from this
   project's sanitizer coverage before this session; now `make test-tsan` exists.
4. **`execute_command()` exit-code race** (`src/agent/cmd_exec.c`) — the read loop's pipe-EOF branch
   broke without ever calling `waitpid()`, leaving `status` at 0 (which decodes as "exited normally,
   code 0") for a command that actually failed. Fixed by reaping the child on that path too.
5. **Symlink-escape sandbox gap** (`src/agent/cmd_exec.c`) — closed a previously-documented "known gap":
   a relative name that is itself a symlink pointing outside the CWD bypassed the lexical `..`
   /absolute-path check. Added `realpath()` + `getcwd()` prefix confinement.
6. **JSON grammar leading-zero acceptance** (`src/sampling/grammar_json.c`) — the PDA treated `'0'` like
   any other leading digit, accepting `"01"`/`"09"` as complete valid JSON (RFC 8259 forbids this).
   Added a dedicated `JSON_ST_NUM_INT_ZERO` state.

## What did NOT find bugs (equally load-bearing evidence)

- TS-1.2: 153 geometry assertions across all 25 quant types — all correct.
- TS-2.2: 637 assertions against the real 49152-token SmolLM2 vocab (no-deadlock invariant, EOS gating,
  non-destructive scratch-copy, multi-byte/byte-level-BPE token handling) — `fsm.c`/`grammar_json.c`
  confirmed correct.
- TS-2.3/2.4: 17 assertions across the CLI/API JSON-mode matrix (3 temperatures, forced truncation,
  adversarial and prompt-injection prompts, streaming + non-streaming, non-JSON regression) — no
  markdown fences, no trailing prose, non-JSON path unaffected.
- TS-5.1: 26 assertions confirming bit-for-bit determinism across all 4 SIMD backends × up to 8 threads
  at temperature 0 (the correctness property greedy decoding requires).
- TS-3.1: 69 adversarial-corpus assertions confirmed the sandbox's absolute-path/home-shortcut/traversal
  blocking and traversal-lookalike allow-listing were already correct; only the two items above were
  real gaps.

## Tools added (not part of `make test` — depend on a real downloaded model or an external llama.cpp
checkout; run manually per their own header comments)

| Tool | Purpose |
|---|---|
| `tools/difftest_dequant.c` | TS-1.1 — bit-exact dequant vs real ggml |
| `tools/grammar_json_verdict.c` + `tools/fuzz_grammar_json.py` | TS-2.1 — differential fuzzing vs `json.loads` |
| `tools/fsm_real_vocab_check.c` | TS-2.2 — token-masking correctness against a real 49152-token vocab |

## Tests added (part of `make test`, run every time)

`tests/test_gguf_geometry.c`, `tests/test_cmd_exec_adversarial.c`, plus new assertions in
`tests/test_gguf_quant_new_formats.c` (IQ4_NL regression) and `tests/test_grammar_json.c` (leading-zero
regression).

## Scripts added (manual, real-model-dependent)

`tests/golden_regression.sh` (TS-5.1), `tests/e2e_json_mode.sh` (TS-2.3/2.4).

## Build system

`make test-tsan` (TS-5.3) and the `build/.variant` staleness guard are both new; `Makefile` and
`CMakeLists.txt` remain in sync (no new `src/` files were added — only `tests/*.c`/`tools/*.c`, which
both build systems already glob/list appropriately).

## Deferred (P1/P2 per the test plan, not attempted this pass — explicitly not silently dropped)

TS-1.3 property tests, TS-1.4 GGUF fuzzing, TS-1.5 negative cases, TS-2.5 masking perf cost, TS-2.6 API
concurrency/no-state-leak, TS-3.2 remaining gap documentation (PATH hijacking, TOCTOU,
`PROJECT_ZERO_AGENT_AUTO_APPROVE`), TS-3.3 rlimit verification, TS-3.4 PTY approval flow, TS-5.2
perplexity/KL-divergence, TS-5.4 static analysis (clang-tidy/cppcheck run, not yet triaged), TS-5.5
coverage, TS-5.6 perf regression gate, TS-5.7 soak, TS-5.8 build matrix, mutation testing, Windows/ARM.
These are the test plan's own Phase 2/3 items; none were part of the Phase 1 (P0) scope this pass covered.

## Verification of this state

```
make clean && make release CC=gcc  && make test CC=gcc  && make debug CC=gcc  && make test-tsan CC=gcc
make clean && make release CC=clang && make test CC=clang && make debug CC=clang
```
All green, from a clean tree, both compilers, immediately before this report was written.
