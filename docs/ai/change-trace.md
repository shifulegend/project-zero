# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-09-18.

### 2026-09-18 — TS-2.2: added a real-vocab (49152-token) token-masking correctness check; no production bugs, confirms fsm.c/grammar_json.c correct
- What: `tools/fsm_real_vocab_check.c` loads the real embedded tokenizer from a GGUF model file and
  drives `fsm_compute_token_mask`/`fsm_advance` through 18 realistic JSON documents (all number forms,
  strings with escapes/unicode, nested objects/arrays), checking every case from the test plan: the
  no-deadlock invariant, EOS gating (exact equivalence with a trial-finalize, not an assumed shape),
  the scratch-copy non-destructive invariant, and multi-byte/byte-level-BPE token consistency.
- Result: 637/637 assertions pass against a real 49152-token vocab (gcc and clang, no warnings) — this
  run found two false assumptions in the *test itself* (fixed), not in the production code. See
  `mistakes.md` 2026-09-18 for the two subtleties (number-grammar closeability without finalize; a
  mid-number position can be a legitimate EOS-eligible stopping point).

### 2026-09-18 — TS-2.1: added grammar differential fuzzer, found and fixed a real JSON leading-zero bug
- What: `tools/grammar_json_verdict.c` (batch C harness driving the real `json_grammar_step`/
  `json_grammar_finalize`) + `tools/fuzz_grammar_json.py` (generator + differential comparison against
  `json.loads`, plus a prefix-truncation asymmetry check). Immediately found that the PDA accepted
  leading-zero numbers (`"01"`, `"09"`) as valid JSON, which RFC 8259 forbids and `json.loads` correctly
  rejects. Fixed with a new `JSON_ST_NUM_INT_ZERO` state (`include/sampling/grammar.h`,
  `src/sampling/grammar_json.c`) that, unlike `JSON_ST_NUM_INT`, does not accept a further digit after a
  leading `0`. Added `test_leading_zero_numbers_rejected` regression test.
- Verified: 0 disagreements across ~61,000 fuzzed candidates (5 seeds), 0 prefix-asymmetry violations;
  end-to-end confirmed on a real model with `--json`; full release/test/debug green on gcc and clang.
  See `mistakes.md` 2026-09-18 for full RCA.

### 2026-09-18 — TS-3.1 sandbox adversarial corpus: fixed a real exit-code bug + closed the symlink-escape gap
- What: `tests/test_cmd_exec_adversarial.c` (TS-3.1) table-drives the full adversarial argument corpus
  from the test plan against `src/agent/cmd_exec.c`. Found and fixed two real issues along the way:
  (1) `execute_command()` could report exit code 0 for a command that actually failed, because the
  read loop's pipe-EOF branch broke out without ever calling `waitpid()` — fixed by reaping the child
  there too. (2) The documented "symlink escape" known gap (a relative name that is itself a symlink
  pointing outside the CWD bypassed the lexical `..`/absolute-path check) is now closed: added
  `path_escapes_cwd_via_symlink()`, a `realpath()`+`getcwd()` prefix check.
- Affected: `src/agent/cmd_exec.c`, `include/threading/thread_pool.h` unaffected (separate fix earlier
  today). Needed `_DEFAULT_SOURCE` for `realpath()` under strict `-std=c99` on clang (gcc didn't need it).
- Verified: `test_cmd_exec_adversarial` 69/69, `test_cmd_exec` 22/22, full release/test/debug green on
  gcc and clang. See `mistakes.md` 2026-09-18 for full detail.

### 2026-09-18 — TS-5.3: added `make test-tsan`, found and fixed a real thread-pool data race
- What: added `CFLAGS_TSAN`/`LDFLAGS_TSAN` and a `make test-tsan` target (`Makefile`) scoped to the
  thread pool, parallel matmul, and API server concurrency surfaces (`audit_threadpool_stress`,
  `test_threading`, `test_api_server`, `test_q4k_x8_matmul`, `test_q2_0_matmul`) — TSan was previously
  entirely missing from this project's sanitizer coverage.
- Found: `ThreadPool.shutdown` (`include/threading/thread_pool.h`) was a plain `bool` read without the
  mutex in `worker_entry()`'s lock-free spin fast path while written under the mutex in
  `threadpool_destroy()` — a real data race (undefined behavior, not just a TSan nag) that could in
  principle let an optimizing compiler hoist the read out of the spin loop and hang shutdown. Fixed by
  making it `atomic_bool` with explicit acquire/release ops at every site, matching the struct's
  existing atomics. See `mistakes.md` 2026-09-18 for full details.
- Verified: `make test-tsan` now reports 0 races on gcc and clang; full `release`/`test`/`debug` still
  green on both compilers after the fix.

### 2026-09-18 — TS-1.1 differential dequant testing vs real ggml: found and fixed a genuine IQ4_NL bug + a Makefile build-staleness bug
- What: built a pinned llama.cpp/ggml reference (`4fea119de30f6a923992780f6fd5ccb0bee5d47d`) and wrote
  `tools/difftest_dequant.c` (TS-1.1), which quantizes a deterministic tensor with ggml's real
  `ggml_quantize_chunk()` and compares ggml's real `dequantize_row_<type>()` against this repo's
  `gguf_dequant_<type>()` for bit-exact equality across all 20 non-trivial types.
- Found: `IQ4_NL` used interleaved-pair nibble packing instead of ggml's split-half layout (this repo's
  own `IQ4_XS` decoder already used the correct layout for the same codebook). Fixed in
  `src/core/gguf_quant.c`; added regression coverage in `tests/test_gguf_quant_new_formats.c`
  (`test_iq4_nl_single_block`). All 20 types now bit-exact; see `mistakes.md` 2026-09-18 for full RCA
  and its relation to the 2026-09-17 open degenerate-output bug.
- Found (while validating the fix via `make release && make test && make debug`): `make debug` run
  after `make release` silently links stale, non-sanitized objects (0 ASan/UBSan symbols) because
  `build/%.o` paths are shared across CFLAGS variants and Make only checks mtimes. Fixed in `Makefile`
  with a `build/.variant` stamp that forces a clean rebuild on variant switch. See `mistakes.md`
  2026-09-18 for details. Verified release/test/debug all green, gcc and clang.
- Added: `tests/test_gguf_geometry.c` (TS-1.2, byte-exact geometry assertions via the real
  `gguf_read_header()` parsing path, 153 assertions) and `tools/difftest_dequant.c` (not a `tests/*.c`
  file — depends on an external llama.cpp checkout at `/tmp/llama-ref`, run manually, not part of
  `make test`).
- Closed the 2026-09-17 open bug (dense llama-arch models with IQ4_NL/IQ4_XS/Q3_K weights producing
  degenerate repeating-token output): re-downloaded `bartowski/SmolLM2-135M-Instruct-GGUF`'s Q3_K_S and
  IQ4_XS variants (TS-4 repro) and re-ran the exact original prompts after the IQ4_NL fix — both now
  produce fully coherent, correct output at `--temperature 0` with no degeneration over 40 tokens. The
  nibble-packing bug was the entire root cause.

### 2026-09-17 — Agent sandbox hardening: argument-level exec policy (Tier 1 item 3)
- What: `src/agent/cmd_exec.c`'s allow-list only ever checked the command *name*
  (echo/ls/cat/pwd/uname/date/id), never its arguments -- `cat`/`ls` could read any file the
  process could read, and a deployment with `PROJECT_ZERO_AGENT_AUTO_APPROVE=1` set gets zero
  human review of that before it runs (see `user_approval.c`). Added
  `exec_policy_allows_args()`: confines `cat`/`ls` path arguments to the CWD subtree (no
  absolute paths, no `~`, no `..` traversal segment), restricts `date` to read-only forms (no
  args or `+FORMAT` only, blocking `-s`/`--set`). Added `setrlimit(RLIMIT_CPU/RLIMIT_AS)` in
  the forked child as defense-in-depth independent of the parent's own timeout loop.
- Correction to the architecture doc's own framing: `CPU_LLM_TERNARY_ENGINE.md` calls the
  current exec path "terrifyingly dangerous" and frames it as `popen()`-based; it's actually
  `fork()`+`execvp()` (no shell), so there was never a shell-metacharacter injection vector --
  the real, confirmed gap was argument-level path confinement, not shell injection. Corrected
  in `docs/architecture/IMPLEMENTATION_PLAN.md`'s Phase 14 section.
- Verification: 7 new assertions in `tests/test_cmd_exec.c` (path confinement, ".." detection
  including the "not actually traversal" edge cases like `"..."`/`"foo..bar.txt"`, the `date`
  restriction, confirming unrelated allow-listed commands aren't accidentally over-restricted,
  and an end-to-end check that `execute_command()` itself -- not just the policy function in
  isolation -- rejects `cat /etc/hostname`). Not verified via live LLM tool-call elicitation
  (coaxing a real small model into emitting a malicious `<exec>` tag deterministically isn't a
  reliable test) -- the security boundary itself is deterministic and fully covered by direct
  unit/integration tests of the sandbox.
- Why: user's `/goal` — Tier 1 item 3, completing all three Tier 1 items.
- Areas: `include/agent/cmd_exec.h`, `src/agent/cmd_exec.c`, `tests/test_cmd_exec.c`,
  `docs/architecture/IMPLEMENTATION_PLAN.md`.

### 2026-09-17 — Phase 20: Grammar-constrained JSON-mode decoding
- What: implemented grammar-constrained decoding for JSON output, wired into the CLI (`--json`
  flag, REPL `/json` toggle) and the API (`"response_format": {"type": "json_object"}`, matching
  OpenAI's actual request schema). Deliberately built a JSON-specific pushdown automaton
  instead of the plan's generic flat-FSM/BNF-compiler design — a flat FSM (no stack) cannot
  represent JSON's real, unbounded-nesting grammar without hard-coding a max depth, which the
  project's own "no hardcoding" rule exists to avoid. Full rationale and file map in
  `docs/architecture/IMPLEMENTATION_PLAN.md`'s Phase 20 section.
- Verified against a real downloaded model (`bartowski/SmolLM2-135M-Instruct-GGUF`, Q8_0), both
  via the CLI and the live API server: `response_format=json_object` produced
  `{"fruit_name": "apple", "fruit_color": "red", "fruit_calories": 500}` (clean, parses with
  `json.loads`), vs. the same prompt without it producing a stray ` ```json ` fence plus
  trailing prose that would break a naive parser.
- One related, deliberately-deferred observation from that testing: the CLI's startup
  banner/hardware-profile/model-config diagnostics print to stdout (not stderr), so
  `--json`'s own output is clean but piping `adaptive_ai_engine --json ... | jq` still needs to
  skip the banner first. Fixing that means auditing/redirecting many pre-existing `printf()`
  call sites in `src/cli/main.c` unrelated to this feature — flagged here rather than done as a
  drive-by change in this pass; the API path (already wired, returns clean structured JSON with
  no stdout concern at all) is the better-suited integration point for programmatic use anyway.
- Not wired into `src/agent/agent_loop.c` (which has its own, separate sampling cascade) --
  which part of an agentic turn should be JSON-constrained is a real design decision, not an
  oversight; left for a follow-up rather than guessed at.
- Why: user's `/goal` — Tier 1 item 2.
- Areas: `include/sampling/{grammar,fsm,constrained_sample}.h` (new),
  `src/sampling/{grammar_json,fsm,constrained_sample}.c` (new), `tests/{test_grammar_json,
  test_fsm}.c` (new, 47+25 assertions), `include/transformer/generate.h` + `src/transformer/
  generate.c` (new `json_mode` parameter), `include/cli/args.h` + `src/cli/{args,main,repl}.c`,
  `include/api/chat_request.h` + `src/api/{json_parse,http_server}.c`, `CMakeLists.txt`
  (SAMPLING_SOURCES), `docs/architecture/IMPLEMENTATION_PLAN.md`.

### 2026-09-17 — Phase 37 complete (all 16 GGUF quant sub-formats) + real-model test surfaced a new open bug
- What: implemented Q4_1, Q8_1, Q8_K, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ1_S/IQ1_M, IQ4_XS dequant
  functions (`src/core/gguf_quant.c`), completed the `GGUFType` enum, extracted BF16 dequant, and fixed
  a real bug found along the way (`gguf_block_size(Q8_1)` in `gguf_reader.c` was 40 bytes, should be 36 —
  the plan doc's own "d fp32 + s fp32" field description was wrong, both are fp16). All codebook tables
  for the IQ-family (iq1s_grid, iq2xxs/xs/s_grid, iq3xxs/s_grid, ksigns_iq2xs) were extracted
  programmatically from a freshly-fetched copy of llama.cpp's `ggml-common.h`, not hand-transcribed, to
  avoid errors across ~2500 hex table entries. New `tests/test_gguf_quant_new_formats.c` (64 assertions,
  ground-truth hand-computed expected values, not self-consistency checks). Also corrected two
  IMPLEMENTATION_PLAN.md mismarks found along the way: 37.2 (Q3_K) and 37.9 (IQ4_NL) were already fully
  implemented but marked "pending" — same doc-lag pattern as the earlier progress-audit session.
- Real-model testing (per explicit user instruction to test every substep against downloaded models, not
  just synthetic fixtures): downloaded 4 variants of `bartowski/SmolLM2-135M-Instruct-GGUF` (Q8_0, Q4_0,
  Q3_K_S, IQ4_XS). Q8_0/Q4_0 work correctly; Q3_K_S/IQ4_XS produce degenerate repeating-token output.
  Spent significant additional effort proving the new dequant code is NOT the cause (bit-exact verified
  against real file bytes at multiple tensor positions, matching an independent reimplementation of
  llama.cpp's actual algorithms) before concluding the root cause is elsewhere and logging it as an open,
  unresolved bug rather than either silently shipping a false "tested and working" claim or burning
  unbounded further time chasing it single-session. Full writeup: `docs/ai/mistakes.md` 2026-09-17 entry.
- Why: user's `/goal` — "complete tier 1 [doc-hygiene items already done + Phase 37 + grammar decoding +
  agent sandbox hardening]. Continuously keep testing every substeps using models downloaded sequentially."
- Areas: `src/core/{gguf_quant,gguf_loader,gguf_reader}.c`, `include/core/{gguf_quant,gguf_reader}.h`,
  `tests/test_gguf_quant_new_formats.c` (new), `docs/architecture/IMPLEMENTATION_PLAN.md`,
  `docs/ai/mistakes.md`. Models downloaded to `models/` for testing, deleted after (not committed,
  `models/` is gitignored).

### 2026-09-17 — IMPLEMENTATION_PLAN.md doc-hygiene sweep (progress audit follow-up)
- What: a full architecture/implementation-plan progress audit (live `make release`/`make test`
  run + cross-referencing `docs/architecture/IMPLEMENTATION_PLAN.md` against actual `src/`/
  `tests/`/`tools/` contents and the companion `docs/architecture/CPU_LLM_TERNARY_ENGINE.md`)
  found the plan doc had stalled updating its header-level ✅ marks around Phase 16-S/K-5, even
  though Phases 10, 11, 12, 13, 15, 16-D, 16-E, 17 (correctness only — perf still open), 24
  (YaRN — the single biggest miss, fully shipped with zero prior acknowledgment), 34.2b, 34.5,
  and 35 were all implemented, tested, and in some cases already documented as done in
  `CPU_LLM_TERNARY_ENGINE.md`'s own "Implementation Record" sections. Added ✅ markers with
  evidence (file paths, test files, decision-log dates) to all 12 headers. Also found and fixed
  a genuine cross-doc inconsistency: `IMPLEMENTATION_PLAN.md`'s numeric "Phase 22" (unbuilt
  Mamba/RWKV state-space support) collided with `decision-log.md`/`mistakes.md`/
  `change-trace.md`'s own pervasive use of "Phase 22" (sub-phases 22.0-22.5) for the
  already-shipped Web UI & API/DX hardening work — renamed the plan doc's phase to "22-SSM"
  with an explicit disambiguation note pointing at the other three docs, rather than touching
  their historical entries. (One earlier claim from the audit subagent — that Phase 15's header
  had a mojibake character instead of a ✅ — was checked and found to be a false positive from
  reading raw UTF-8 bytes via `cat -A`/`grep`; the header already had a correct ✅, left as-is.)
- Why: `docs/ai/tool-sync-policy.md`/`.claude/rules/docs.md` call for proactively keeping
  `docs/ai/**` and related canonical docs synced with reality, without being asked per-instance;
  user explicitly asked for this sweep after reviewing the audit.
- Areas: `docs/architecture/IMPLEMENTATION_PLAN.md` only (doc-only change, no engine code
  touched). No new bugs found beyond what the audit's "known open risks" list already covered
  (MoE perf gap, Qwen3-8B dense throughput gap, issue #32 segfault, classifier re-benchmark,
  pre-2026-07-17 ceiling claims needing re-verification — all pre-existing and already tracked
  elsewhere, not introduced by this sweep).

### 2026-08-04 — DeepSeek MoE benchmark run: project-zero vs colibri vs llama.cpp
- What: user asked to run DeepSeek MoE on project-zero, colibri (`shifulegend/colibri`), and
  llama.cpp and report results. Found colibri has no DeepSeek-arch loader (only GLM-5.2/Inkling/
  Kimi K3/OLMoE, one engine per family — DeepSeek referenced only as prior-art in `colibri.c`)
  and project-zero has no OLMoE-arch loader (`gguf_loader.c`'s MoE dispatch only special-cases
  `deepseek2`/`qwen35`/`qwen3moe`; an `olmoe`-arch GGUF hard-fails in the generic dense loader).
  Ran two pairwise comparisons instead of one 3-way table, both anchored on llama.cpp: (1)
  DeepSeek-V2-Lite-Chat Q4_K_S on project-zero (4.07 tok/s avg) vs llama.cpp (12.87 tok/s), T=4;
  (2) OLMoE-1B-7B-0924-Instruct on colibri (0.63 tok/s avg, 0.31-0.90 range) vs llama.cpp
  (27.50 tok/s), T=4. Full methodology, raw logs, and caveats (different CPU than the standing
  `BENCHMARK_SUMMARY.md` record; colibri's streaming design is built for disk-resident
  100B+-scale models, not this 6.9B one) in `benchmark_results/deepseek_moe_2026-08-04/README.md`.
- Why: user request, made directly in this session.
- Areas: `benchmark_results/deepseek_moe_2026-08-04/**` (new). No engine code changed — this was
  a measurement run, not a fix; the memory-bandwidth-bound MoE gap remains the open item tracked
  in `README.md`'s Help Wanted table and `docs/DEEPSEEK_Q8_HANDOVER.md`.
- Branch: `claude/deepseek-moe-benchmarks-j15vh5`.

### 2026-07-24 — Documented: no Claude/AI signature/attribution unless explicitly asked
- What: added a rule to `docs/ai/commit-log-guidance.md` (canonical) and mirrored it into the
  entry adapters (`CLAUDE.md`, `.claude/rules/core.md`, `.github/copilot-instructions.md`,
  `gemini/GEMINI.md`, `AGENTS.md`) — no Claude/AI signature or attribution (commit trailers,
  code comments, generated artifacts) unless the user explicitly asks for it in that instance.
- Why: user request, made directly in this session.
- Areas: docs/ai/commit-log-guidance.md, CLAUDE.md, .claude/rules/core.md,
  .github/copilot-instructions.md, gemini/GEMINI.md, AGENTS.md.

### 2026-07-24 — Full threads×SIMD×classifier sweep report (repo-only) + INT4-slower RCA
- What: full 52-config sweep report (`docs/design/reports/sweep-2026-07-24.html` +
  `.template.html`), raw CSVs (`docs/design/reports/sweep_2026-07-24/`), 5 terminal screenshots
  (`docs/design/screenshots/sweep_2026-07-24/`) — all committed in-repo, no external hosting.
- Why: user asked for a full threads×SIMD×classifier sweep of both engines with terminal
  screenshots and a final infographic, following the earlier F16/BF16 RCA below.
- Also: caught and fixed two methodology bugs found while building this report — an early-EOS
  short-generation measurement bug (fixed by moving to a 251-real-token run per config) and a
  concurrent-capture CPU-contention bug — both in `mistakes.md`. Root-caused why INT4 classifier
  measures slower than INT8/BF16 (nibble-unpack cost without AVX-512 VBMI + classifier already
  being cache-resident, so INT4's bandwidth saving buys nothing) — full analysis in
  `decision-log.md`; added as a Help Wanted item in `README.md` rather than fixed this pass.
- Areas: `docs/design/reports/sweep-2026-07-24.*`, `docs/design/reports/sweep_2026-07-24/*.csv`,
  `docs/design/screenshots/sweep_2026-07-24/*.png`, `docs/ai/mistakes.md`,
  `docs/ai/decision-log.md`, `README.md` (Help Wanted table + benchmark section notes).

### 2026-07-24 — Fixed FMA-latency-bound F16/BF16 GEMV kernels after user-requested RCA
- What: `parallel_matmul_f16` (`src/math/matmul_f16.c`) and `parallel_matmul_bf16`
  (`src/math/parallel_matmul.c`) AVX2/AVX-512 paths rewritten from a single FMA accumulator per
  row to 4 independent accumulators, matching llama.cpp/ggml's `ggml_vec_dot_f16` structure.
- Why: user asked why project-zero measured behind llama.cpp on a SmolLM2-135M F16 comparison and
  asked for an RCA before any fix. Repeated measurement showed the single-sample gap was noise at
  T=2 but a real ~13% gap at T=1 (zero thread-dispatch overhead), traced to the single-accumulator
  FMA dependency chain being latency-bound rather than throughput-bound.
- Areas: `src/math/matmul_f16.c`, `src/math/parallel_matmul.c`; `docs/ai/decision-log.md`,
  `docs/ai/mistakes.md`; `docs/design/reports/pz-vs-llamacpp-smollm2.html` (in-repo comparison
  report, moved off external artifact hosting per explicit instruction — all deliverables now
  live in the repo, nothing outside).
- Verified: 5-rep before/after T=1 and T=2 measurement (see decision-log), golden output
  unchanged, full gcc+clang release/test/debug clean, ASan/UBSan-clean real-model run.
- Branch: `claude/pr31-branding-cli-discussion-mwwe12`.

### 2026-07-24 — CLI banner always prints; Qwen3-MoE GGUF loader support added; rename deferred to roadmap
- What: (1) `tn_banner_print()` now shows the "PROJECT ZERO" banner for every invocation
  (TTY or piped/redirected), animated only in a real terminal — new `tn_banner_format_plain()`,
  new `tests/test_banner.c`. (2) Added `qwen3moe` GGUF architecture support (fixes issue #32):
  `MoEConfig.has_qk_norm`, new `src/transformer/qwen3moe_attention.c` +
  `src/core/qwen3moe_run_state.c` (dedicated buffers/KV-cache, mirroring the `qwen35`/MLA
  independent-head-dim precedent), new `weights_from_gguf_qwen3moe()` loader, new
  `tests/test_gguf_loader_qwen3moe.c` (synthetic GGUF fixture + real forward-pass token).
  (3) Documented the `adaptive_ai_engine`→`projectzero` rename as a deferred item in
  `.github/ROADMAP.md` (not implemented — research-only per explicit instruction).
- Why: (1) branding must not depend on the CLI's invocation path (user request, traced to
  `tools/make_screenshot.py` rendering redirected-stdout captures that never had a banner to
  begin with). (2) jpsoto's bug report — `qwen3moe` fell through to the generic dense loader,
  which expects `ffn_gate.weight` instead of the MoE router+stacked-expert tensors this arch
  actually has. (3) scoped/estimated in an earlier discussion; explicitly deferred, not executed.
- Areas: `src/cli/banner.c`/`.h`, `src/cli/main.c`, `tests/test_banner.c`;
  `src/core/gguf_loader.c`, `include/core/moe_config.h`, `include/core/weights.h`,
  `src/core/weights.c`, `src/transformer/attention.c`, `src/transformer/qwen3moe_attention.c`
  (+header), `src/core/qwen3moe_run_state.c`, `include/core/run_state.h`, `CMakeLists.txt`,
  `tests/test_gguf_loader_qwen3moe.c`; `.github/ROADMAP.md`; `docs/ai/decision-log.md`,
  `docs/ai/mistakes.md`, `docs/ai/project-overview.md`, `docs/design/review-2026-07-15.md`.
- Verified: `make release && make test && make debug`, gcc and clang, clean tree each time — zero
  new warnings, ASan/UBSan clean, all tests pass including the two new test files. Real-model
  golden-output verification of the Qwen3-MoE fix against the actual reported file is a tracked,
  not-yet-done follow-up (see decision-log's 2026-07-24 entry) — disk/network constraints in this
  session.
- Branch: `claude/pr31-branding-cli-discussion-mwwe12`.

### 2026-07-18 — README front door: Bonsai-27B x86 comparison promoted to hero + quick start
- What: added a hero block under the intro — the sweep3 headline claim (4.2-4.8x vs PrismML's
  own fork, 2.97 vs 0.70 tok/s @ t4, config + drift caveat stated, links to captures and the
  full section) with a copy-paste quick start (clone/build/curl model/run); intro line now
  names the Bonsai result alongside the BitNet one; `#bonsai` anchor added. Also removed the
  "95% of DRAM bandwidth ceiling" clause from the hero bullets — that figure predates the
  2026-07-17 probe fix (bandwidth was measured ~3x low, so the utilization claim doesn't
  survive re-derivation; see CEILING_CALCULATION.md §2/§5).
- Why: the Bonsai comparison is the repo's strongest current, fully-documented result; the
  README's own claim+proof rules (2026-06-20/21 decisions) require the hero to be the best
  verifiable claim with config stated and no stale numbers.
- Areas: `README.md`.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-18 — 3-axis comparison sweep vs PrismML llama.cpp fork, screenshots + infographic
- What: 18 sequential same-session runs on Ternary-Bonsai-27B (60 tok, temp 0, one host):
  project-zero threads 0.96/1.73/2.50 + four t4 sentinels 2.73-3.12; SIMD@t4
  scalar/avx2/avx512f/vnni 2.82/3.05/2.88/3.08; classifier@t4 bf16/int8/int4 2.51/2.85/3.01.
  Fork (prism branch b1-79697f2): t1-t4 0.2/0.4/0.6/0.7 — project-zero 4.2-4.8x ahead at
  every thread count. Every run captured in a pty (screenshot PNG + raw bytes,
  `benchmark_results/sweep3_2026-07-18/`); capture tool gained opt-in PZ_CAPTURE_RAW_OUT.
  Infographic: `comparison.html` (published artifact) + `comparison_infographic.png`,
  README-linked. This host was host-A-class (L3 260 MiB, 41.9 GB/s fixed-probe).
- Fork gotchas recorded: rejects --no-conversation at runtime then loops printing prompts
  forever on closed stdin (produced a 5 GB log); needs -c 4096 (default 262K ctx = ~15 GB KV
  alloc, swap-thrash) and -st for one-shot runs.
- Process lesson (mistakes-grade, recorded here): a background llama-cli with no timeout sat
  swap-thrashing ~4 h while this session waited on its completion notification — long external
  runs need explicit timeouts and liveness checks (RSS/etime), and their output must be
  streamed/tee'd, never only piped to tail.
- Areas: `benchmark_results/sweep3_2026-07-18/**`, `tools/screenshots/cli/capture.mjs`,
  `README.md`.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-17 — Ceiling push round 2: instrumentation + two evidence-based reverts + KV-line fix
- What: wired step timing into the qwen35 hybrid path (steps 4-12 + new DeltaNet steps 22/23);
  attribution: ~91% Q2_0 matmul / 6.4% scalar recurrence / <3% rest. Tried and REVERTED on
  measurement: A3a wide non-VBMI unpack (kernel is load-bound, not compute-bound — neutral)
  and sub-64-row inline dispatch (pool splitting wins even at 48 rows — slightly negative).
  Fixed the "KV Strategy: Quantized I8" line for qwen35 (F32 reality) and the ceiling doc's
  KV bound it had corrupted (32 → 128 KB/pos). Standing result unchanged: 3.56 tok/s vs 6.0
  ceiling (59%); doc now states that materially exceeding ~60% single-stream requires
  batched/speculative decode, not more kernel tuning.
- Why: user asked to aim for the ceiling; both negative results are recorded per the plan's
  "a documented dead end is a deliverable" rule (full data in CEILING_CALCULATION.md §7,
  lessons in mistakes.md).
- Verification: gcc release 0 warnings, test_q2_0_matmul 35/35 after each change and after
  each revert; interleaved same-session end-to-end A/Bs for every perf-relevant decision.
- Areas: `src/transformer/qwen35_attention.c`, `include/core/step_timing.h`,
  `src/core/step_timing.c`, `src/math/matmul_q2_0_vnni.c` (net: comment only),
  `src/cli/main.c` (KV line), `docs/architecture/CEILING_CALCULATION.md`, `docs/ai/*`.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-17 — Ceiling spec + probe fix (~3x) + Q2_0 kernel optimization (+28% end-to-end)
- What: (1) `docs/architecture/CEILING_CALCULATION.md` — full spec/audit of the tok/s ceiling
  (probe methodology, 4 computation sites, per-model byte accounting, error bounds, bridging
  matrix). (2) Fixed `probe_dram_bandwidth()`'s ~3x accounting error (3 passes timed, 1 pass of
  bytes counted) + serialized volatile reads: same host 12.0 → 41.2 GB/s measured. (3) Optimized
  `dot_q2_0_row_vnni` (per-row float vector accumulator, F16C scales; `_ref` kept for A/B):
  micro-bench 1.32-1.68x (`tools/bench_q2_0`, `make bench-q2`), end-to-end 2.74/2.80 →
  3.56/3.54 tok/s (+28%), token-identical output, on the real downloaded 7.16 GB model.
  (4) calibration.c "Speed opt" line now prints computed values (was hardcoded "(INT8, ~+36%)").
  (5) Fixed stale 18B/64 block-layout comment in matmul_q2_0.h. Corrected README + RCA claims
  built on the broken probe numbers ("2.74 was at the bandwidth wall" → ~40% of true BW; honest
  ceiling ~6-7 tok/s, current utilization 59%).
- Why: user asked for detailed ceiling documentation, an independent review, and gap-bridging;
  the probe accounting bug surfaced during the planned probe improvement.
- Verification: clean gcc release/test 46/46 green (a mixed clang/gcc `build/` from earlier
  compiler cycling first produced a bogus `test_simd_vnni` link failure — the documented
  mtime-staleness Makefile trap; clean rebuild resolved it); clang compile-checks of changed
  TUs zero-warning; interleaved same-session model A/B ×2 rounds; `TN_STEP_TIMING=1` breakdowns.
- Areas: `src/core/hardware_profile.c`, `src/core/calibration.c`, `src/math/matmul_q2_0_vnni.c`,
  `include/math/matmul_q2_0.h`, `tools/bench_q2_0.c`, `Makefile`, `CMakeLists.txt`,
  `docs/architecture/CEILING_CALCULATION.md`, RCA report, README, mistakes.md.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-17 — Fix hardware profiler's hardcoded Data/token + ceiling; correct all ceiling-based claims
- What: `hardware_profile.c` computed `weight_bytes_per_tok`/`theoretical_ceiling` from
  compile-time BitNet-2B constants for every model (Ternary-Bonsai-27B: "1149 MB" instead of
  ~6.8 GB → ~6x-overstated ceiling; SmolLM2: 4.5x *under*stated). Added
  `tn_hardware_profile_set_model_bytes()` (recomputes bytes/ceiling/`model_fits_l3`/summary;
  `rebuild_summary()` extracted from the two duplicated snprintf blocks), called from `main.c`
  after GGUF weight load with real model sizes (embedding/head adjustment for Q2_0-native models,
  materialized-classifier aware; MoE overcounts, TODO). Startup box now labels its figure
  "(pre-load est.)"; corrected `[profile]` line prints post-load. Corrected the RCA report
  (§5 addendum: real ceilings, ~1.5–2x kernel headroom on host-B class, 2.74 tok/s was at host
  A's bandwidth wall) and the README's ceiling-derived claims.
- Why: found while answering "what changes reach the tok/s ceiling" — the target itself failed
  a dimensional sanity check; bug-fix policy requires fixing on discovery.
- Verification: gcc release + test 46/46 green; clang release/debug green (clang `make test`
  blocked by the container's missing clang ASan runtime — known, decision-log 2026-06-19);
  `make demo` end-to-end: corrected line prints, golden output ("Paris") intact.
- Areas: `src/core/hardware_profile.c`, `include/core/hardware_profile.h`, `src/cli/main.c`,
  `docs/ai/mistakes.md`, `docs/reports/RCA_QWEN_TOKS_DROP_2026-07.md`, `README.md`.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-17 — Consolidated RCA report for the 2.74 → ~1 tok/s Ternary-Bonsai drop
- What: Added `docs/reports/RCA_QWEN_TOKS_DROP_2026-07.md` — a single consolidated root-cause
  analysis of the post-classifier-work throughput drop, independently re-verifying the
  evidence already recorded in `mistakes.md`/`decision-log.md` (hot-path diff audit of
  `ce8e90d..HEAD`, the commit-bisection screenshots incl. the pre-VNNI control leg, and the
  host-A hardware profile recovered from git history via `git show 591333d:...pz_t4_peak.png`).
  Conclusion unchanged and now falsifiably documented in one place: host performance-profile
  change (L3 260→33 MiB, DRAM 16→~11-12 GB/s, first-touch stalls), not a code regression.
- Why: the question keeps being re-asked; scattered evidence made the answer look like an
  assertion rather than a proof. Report includes the restore/improve action plan
  (`--classifier int8/int4` same-session A/B; Q2_0 LM-head unpack profiling once a stable
  host is available).
- Areas: `docs/reports/RCA_QWEN_TOKS_DROP_2026-07.md`, README Qwen-section link.
- Branch: `claude/qwen-performance-drop-rca-pepnfp`.

### 2026-07-16 — Qwen 3.5/3.6 hybrid-attention support (Ternary-Bonsai-27B) + benchmark
- What: Full engine support for Qwen 3.5/3.6's hybrid Gated-DeltaNet/Gated-Attention
  architecture and PrismML's Q2_0 ternary GGUF packing — new
  `src/transformer/qwen35_attention.c`, `src/core/qwen35_run_state.c`, `src/math/matmul_q2_0.c`,
  `src/core/gguf_quant.c`'s `gguf_dequant_q2_0`, plus `MoEConfig`/`TransformerWeights`/`RunState`
  extensions (`has_linear_attn`, `q35_*` fields) mirroring the existing `has_mla` pattern. Ran the
  real downloaded `prism-ml/Ternary-Bonsai-27B-gguf` end-to-end and fixed 6 real bugs surfaced
  only by that real run (parser infinite loop, unexecuted macros, dotted-set namespace mutation,
  a whitespace-control lexer bug, an OOB cache write, and a transposed conv1d tensor read) — full
  writeup in `mistakes.md` (2026-07-16, "Qwen 3.6 integration: 6 real bugs"). Added
  `tests/test_chat_template.c` (18 cases). Built PrismML-Eng's `llama.cpp` fork (`prism` branch,
  group-128 Q2_0) as a benchmark/correctness comparator on the identical file.
- Why: task requested running this specific converted model in the engine, identifying other
  engines that can run it, and benchmarking against one of them.
- Areas: `src/transformer/`, `src/core/`, `src/math/`, `src/tokenizer/chat_template.cpp`,
  `include/`, `tests/test_chat_template.c`, `Makefile`, `CMakeLists.txt`, `docs/ai/mistakes.md`,
  `docs/ai/decision-log.md`.
- Result: gcc+clang release/test/debug all green; model produces coherent reasoning output
  matching the reference engine's continuation of the identical prompt token-for-token up to the
  point checked; project-zero's (unoptimized) Q2_0 kernel is markedly slower than the reference
  engine's on this 4-core host — see the benchmark artifact for the actual numbers.

### 2026-07-16 — Graceful --server shutdown (SIGINT/SIGTERM handler + flag-polling loop)
- What: `main.c`'s `--server` block used a bare `pause()` with no signal handler installed, so
  Ctrl+C (SIGINT) or `kill` (SIGTERM) used the default disposition — immediate process
  termination — meaning `pause()` never returned and `api_server_stop()` plus all of `main()`'s
  cleanup (`tokenizer_free`, `gguf_header_free`, `mapped_file_close`, `run_state_free`, etc.,
  several of them freshly added earlier the same day) was unreachable dead code in server mode.
  Added a `sigaction`-installed handler for SIGINT+SIGTERM that sets a
  `volatile sig_atomic_t g_shutdown_requested` flag (the only async-signal-safe action a handler
  can take), and replaced the bare `pause()` with `while (!g_shutdown_requested) pause();`.
  Scoped entirely inside the `--server` branch, installed right before the wait — REPL and
  one-shot `--prompt` Ctrl+C behavior (immediate kill) is unchanged.
- Making that dead code reachable for the first time immediately surfaced two more real,
  previously-latent bugs (documented in full in `docs/ai/mistakes.md`, 2026-07-16):
  1. `api_server_stop()`'s `close(ctx->server_fd)` alone doesn't reliably unblock the listener
     thread's concurrent blocking `accept()` on Linux — confirmed by an actual hang (process
     still running well past a generous timeout after SIGINT). Fixed by calling
     `shutdown(ctx->server_fd, SHUT_RDWR)` before `close()`.
  2. Testing the debug/ASan/UBSan build's server path for the first time (previously untestable —
     no reachable clean-shutdown path) surfaced a genuine unrelated UBSan finding: two GGUF
     metadata numeric-array reads in `src/tokenizer/tokenizer_gguf.c` (`scores`, `token_type`)
     did a raw pointer-cast-and-index over a zero-copy mmap pointer with no alignment guarantee
     — an unaligned load. Fixed via `memcpy` into a local, the same idiom already used elsewhere
     in this file (`str_cursor_next`) and in `gguf_reader.c`'s own scalar-field readers.
- Verification: all six gcc/clang × release/test/debug combinations green (clean `make clean`
  between each); golden "capital of France" output and tok/s unaffected; graceful SIGINT shutdown
  (exit code 0, "Shutting down..." printed, `api_server_stop()` actually runs) manually verified
  against gcc release, gcc debug (ASan/UBSan), and clang debug (ASan/UBSan) builds — no sanitizer
  errors in any of them, confirming the three fixes together.
- Affected files: `src/cli/main.c`, `src/api/http_server.c`, `src/tokenizer/tokenizer_gguf.c`,
  `docs/ai/mistakes.md`.

### 2026-07-16 — Animated GIF of the CLI banner reveal + shimmer for the README
- What: a single screenshot can't show the banner's slide-up reveal or its post-reveal shimmer,
  so added `tools/screenshots/cli/capture-gif.mjs` — a new sibling to `capture.mjs` that captures
  `script(1)`'s *timed* output (`--log-timing`, not just the final bytes) and replays it
  frame-by-frame through the same xterm.js/Playwright harness, screenshotting after every timed
  write, then encodes the frames into an animated GIF via the new `gifenc`/`pngjs` dev
  dependencies (pure-JS, no native deps — `pngjs` decodes each PNG screenshot back to raw RGBA
  for `gifenc` to palette-quantize and encode). Replay is trimmed to the chunk containing a
  configurable marker string (default `"Project Zero Engine"`, the line printed immediately
  after `tn_banner_print()` returns) so the GIF captures exactly the animation, not whatever
  prints next.
- Two real bugs found and fixed while building this (screenshot review caught both, same as the
  Phase 22.4 design-QA pattern): (1) the crop's row offset assumed `script(1)`'s own "Script
  started on ... [COMMAND=...]" header line always occupies exactly one terminal row — false;
  it wraps across a variable number of rows depending on the command's string length (long
  absolute paths push it past 100 columns easily), so a fixed "skip 1 row" offset cut into the
  banner. Fixed by detecting the banner's actual start row at runtime: do one non-timed full
  write of the whole animation slice into a throwaway page, then scan for the first row
  containing a `'#'` glyph column (the banner's own block-font content — the header line never
  contains one). (2) the crop's per-row pixel height was computed from `#term`'s own bounding
  box, which includes `capture.html`'s 16px CSS padding on all sides — dividing a padded box by
  the row count overestimates each row's height and throws the crop off by a visible row.
  Fixed by measuring `.xterm-screen` (the unpadded rendered-rows container) instead.
- Output: `docs/demo_banner_shimmer.gif` (14 frames: ~11 real animation frames at their actual
  45ms/90ms timings, plus 2 held frames on the settled state before the loop repeats), embedded
  in the README's UI/UX section above the existing static banner screenshot.
- Verification: visually confirmed by loading the actual generated GIF in a real (non-headless
  logic) browser page at multiple playback timestamps — an early frame shows the partial
  bottom-up reveal in progress, a late frame shows the fully legible, correctly-cropped
  "PROJECT ZERO" banner with no header-line leakage.
- Affected files: `tools/screenshots/cli/capture-gif.mjs` (new), `tools/screenshots/cli/
  package.json`, `tools/screenshots/cli/package-lock.json`, `docs/demo_banner_shimmer.gif` (new),
  `README.md`.

### 2026-07-16 — Fixed two real ASan leaks + added canonical bug-fix policy + web UI how-to guide
- What: the ~1.2MB ASan leak noticed during Phase 22.5 verification was fixed rather than left
  as a documented-but-unfixed finding. Root causes: (1) `tokenizer_free(&t)` in `src/cli/main.c`
  was gated on `args.tokenizer_path`, which is unset for the common GGUF-auto-load tokenizer
  path — now called unconditionally (safe: `t` is zeroed before either load path, and
  `tokenizer_free` no-ops cleanly on a zeroed struct). (2) `GGUFHeader`'s heap-allocated
  string-metadata copies had no free function at all — added `gguf_header_free()`
  (`src/core/gguf_reader.c`/`.h`), called from `main.c`'s cleanup path and both GGUF-parse-failure
  early returns. Verified clean (zero LeakSanitizer output) on both the one-shot `--prompt` and
  REPL paths after the fix, where both previously leaked on every run.
- Added `docs/ai/engineering-rules.md` § "Bug-fix policy" (any bug found gets fixed in the same
  pass, even pre-existing/unrelated ones — only large architectural fixes get deferred, and only
  with an explicit flag to the user). Synced to `CLAUDE.md`, `.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`. Recorded as a process
  decision in `docs/ai/decision-log.md` and the leak root-causes in `docs/ai/mistakes.md`
  (both 2026-07-16).
- Added `docs/WEBUI_GUIDE.md` — the how-to guide that was missing: starting the server/web UI,
  every web UI control (composer, Stop, Params sliders, theme toggle, image upload, model info
  panel), REPL commands (`/quit`, `/context`, `/think`, `/agent`, `/memory ...`), CLI flags
  (`--color`, `--web-ui`, `--static-dir`, `--cors*`, `--api-key`, `--metrics`), and an HTTP route
  reference table. Linked prominently from the top of the README's UI/UX section (previously
  that section was screenshots + short blurbs only, with no actual usage walkthrough).
- Verification: gcc + clang × release/test/debug all green; golden "capital of France" output
  and tok/s unaffected; fresh screenshots confirm the REPL (banner/spinner/shimmer) still
  renders correctly post-fix.
- Affected files: `src/cli/main.c`, `src/core/gguf_reader.c`, `include/core/gguf_reader.h`,
  `docs/ai/engineering-rules.md`, `CLAUDE.md`, `.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`,
  `docs/ai/decision-log.md`, `docs/ai/mistakes.md`, `docs/WEBUI_GUIDE.md` (new), `README.md`.

### 2026-07-16 — Phase 22.5 (continued): Live generation spinner + banner shimmer
- What: user asked for a "moving logo" like Claude Code's animated working indicator, on top of
  the one-time startup banner reveal. Added `tn_live_stats_spinner_frame()` (pure function of
  tick count, cycles through a standard 10-frame braille spinner) to `include/cli/live_stats.h`/
  `src/cli/live_stats.c`; `tn_live_stats_render()` gained a `color_enabled` parameter and now
  prepends the bold-cyan spinner glyph (distinct from the dim stats text and the green banner
  accent) before the `[N tok, X tok/s]` status line, advancing once per streamed token. Updated
  the one call site (`src/cli/repl.c`'s `ReplGenContext`/`repl_token_callback`) to thread
  `color_enabled` through. Also extended `tn_banner_print()` (`src/cli/banner.c`) with a bounded
  3-cycle dim/bold "shimmer" once the reveal finishes, rather than freezing instantly — capped at
  3 cycles (not indefinite) since the REPL blocks on stdin for the first prompt right after,
  so continuous animation would need a background thread, out of scope for a startup flourish.
- Added `tests/test_live_stats.c` (new — `live_stats.c` previously had no dedicated unit test)
  covering the new pure spinner-frame function: non-null/non-empty frames, period-10 cycling,
  variation across the cycle, and the existing tick/init counters. `tn_live_stats_render` itself
  remains manual/screenshot-verified only, same as the rest of this file's terminal-side-effect
  code.
- Verification: gcc + clang × release/test/debug all green (clean between each); golden
  "capital of France" output and tok/s unaffected; a pre-existing, unrelated ASan leak in
  `tokenizer_load_from_gguf`/`strdup` (~1.2MB, same byte/allocation count with and without this
  change) was confirmed present on the prior commit too — not introduced by this work. Screenshot
  at `docs/design/screenshots/07-cli-spinner-and-shimmer-2026-07-16T01-44-29Z.png` shows the
  spinner live during a streaming response; addendum in `docs/design/review-2026-07-15.md`.
- Affected files: `include/cli/live_stats.h`, `src/cli/live_stats.c`, `src/cli/repl.c`,
  `src/cli/banner.c`, `include/cli/banner.h`, `tests/test_live_stats.c` (new),
  `docs/design/review-2026-07-15.md`, `README.md`.

### 2026-07-16 — Phase 22.5: Animated ASCII-art CLI startup banner
- What: identified a remaining UI/UX gap — leading CLI/LLM tools (e.g. Claude Code) show an
  animated ASCII-art name banner on startup; project-zero had none. Added
  `include/cli/banner.h`/`src/cli/banner.c`: a hand-crafted 5-row block font (no figlet/external
  font dependency) covering the letters needed for "PROJECT ZERO", composed programmatically
  (`compose_banner`) so letter concatenation itself can't misalign — only the individual glyphs
  need to be correct. `tn_banner_print()` reveals the banner bottom-row-first over
  `GLYPH_ROWS` frames (~45ms/frame via `nanosleep`, not `usleep` — removed from POSIX.1-2008 and
  undeclared under this project's strict `_POSIX_C_SOURCE=200809L`), giving a "slide up into
  place" effect using only cursor-up + per-line clear (portable across real terminals and
  xterm.js). Wired into `src/cli/main.c`: shown for REPL and `--server` mode, suppressed for
  one-shot `--prompt` runs (matches Claude Code's own interactive-vs-scripted banner behavior),
  gated on `is_tty` so no escape codes ever leak into piped/redirected output. Registered
  `src/cli/banner.c` in `CMakeLists.txt`.
- Bug found and fixed during first real-terminal capture: the glyph separator was a literal
  `' '` character, but the row-printing function only treats `'.'` as blank (anything else prints
  as `'#'`) — every inter-letter gap rendered solid, garbling the banner. Fixed by using `'.'` as
  the separator. Full writeup in `docs/ai/mistakes.md` (2026-07-16).
- Verification: gcc + clang × release/test/debug all green (clean `make clean` between each,
  per the repeated stale-object lesson this project has hit before); golden "capital of
  France" output and tok/s unaffected (this is CLI-startup-only code, no generation-path
  changes); real terminal screenshot captured via `tools/screenshots/cli/capture.mjs` and
  hand-verified against the expected per-letter glyph layout — see
  `docs/design/screenshots/06-cli-startup-banner-2026-07-16T01-03-52Z.png` and the addendum in
  `docs/design/review-2026-07-15.md`.
- Affected files: `include/cli/banner.h`, `src/cli/banner.c` (new), `src/cli/main.c`,
  `CMakeLists.txt`, `docs/design/review-2026-07-15.md`, `README.md`.

### 2026-07-15 — Phase 22.4: Design QA, regression re-verification, README
- What: Wrote `docs/design/ui-ux-principles.md` (the checklist every UI/UX screenshot is graded
  against — best-practice checklist, explicit anti-patterns, a concrete "avoid the generic
  AI-generated look" list, symmetry/Gestalt principles). Built the screenshot capture tooling
  (`tools/screenshots/webui/chat.spec.mjs` — Playwright directly against a live server;
  `tools/screenshots/cli/capture.mjs` — `script(1)` + `@xterm/xterm` in a headless Playwright
  page, chosen over a hand-rolled ANSI-to-HTML converter so cursor movement/in-place updates
  render pixel-accurately). Captured and reviewed real screenshots of both the web UI (empty
  state, streaming, Stop button, sampling params, dark theme) and the CLI (color, markdown
  rendering, live tok/s) — full pass/fail writeup in `docs/design/review-2026-07-15.md`. Added a
  "UI/UX" section to `README.md` with the accepted screenshots.
- The design-QA pass caught two real issues, not just cosmetic nits — proof the gate does what
  it's meant to: (1) `webui/src/lib/ImageUpload.svelte` used a 📎 emoji as its only icon,
  violating the anti-pattern list I'd just written — replaced with an inline SVG. (2) A real bug:
  the REPL's live tok/s status line (`\r` + overwrite) was clobbering the actual streamed
  response text, since both wrote to the same terminal row with no forced newline between them —
  only visible in a real screenshot, not in any unit test. Fixed in `src/cli/live_stats.c` via
  save-cursor/jump-to-last-row/restore-cursor (full writeup in `docs/ai/mistakes.md`).
- Regression safety net (mandatory per the Phase 22 plan): re-ran golden-output checks (Paris/
  Berlin) on both the CLI and HTTP API paths after all Phase 22 structural changes (concurrency
  rearchitecture, vision-pipeline extraction, live-stats fix) — unchanged. Also caught and fixed
  a purely self-inflicted verification mistake twice during this phase: switching compiler/build
  type (e.g. `make debug CC=clang` → `make release CC=gcc`) without `make clean` in between
  leaves stale object files linked into what looks like a "release" build but is actually still
  ASan/UBSan-instrumented — always `make clean` between differing CC/build-type combinations.
- `npm audit` note carried over from 22.2 still applies (dev-dependency-only, SSR-specific
  advisories that don't apply to this static SPA).
- Verified: `make release/test/debug` green on gcc and clang after the `live_stats.c` fix; live
  end-to-end re-verification of the web UI (Playwright) and CLI (real terminal capture) with the
  fix applied, confirmed correct.
- Areas: `docs/design/{ui-ux-principles,review-2026-07-15}.md` (new), `docs/design/screenshots/`
  (new, 6 images), `tools/screenshots/**` (new), `src/cli/live_stats.c` (bug fix),
  `webui/src/lib/ImageUpload.svelte` (emoji→SVG fix), `README.md` (new UI/UX section),
  `docs/ai/mistakes.md`.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.2: Web chat UI (Vite+Svelte, embedded bundle) + image upload
- What: Added `webui/` — a Vite+Svelte SPA (chat window, streaming responses, sampling
  controls, dark/light theme, model info, image upload) built via `npm --prefix webui run
  build`. The build output is embedded into the C binary as a generated, git-committed TU
  (`src/api/webui_bundle_generated.c`, produced by the new standalone tool
  `tools/gen_webui_bundle.c`) so ordinary `make release`/`cmake --build` never need Node —
  only `make webui-bundle` (opt-in, never in `all`/`release`/`test`) does. `GET /` and
  `GET /assets/*` are served by the new `src/api/static_assets.c` (falls back to `index.html`
  for unknown top-level GETs, i.e. client-side SPA routing; supports `--static-dir <path>` to
  serve from disk in dev mode instead of the embedded bundle). `GET /` now serves the UI instead
  of the old health-check-alias JSON — `GET /health` is unchanged and remains the real check.
  New flags: `--web-ui <auto|on|off>` (default auto), `--static-dir <path>`.
- Image upload: extended `chat_request`'s JSON parser (`src/api/json_parse.c`) to accept the
  OpenAI "content parts" array form (`[{"type":"text",...},{"type":"image_url",...}]`) alongside
  plain-string content, added a new `src/api/data_url.c` base64 `data:` URL decoder (none
  existed in the codebase before), and extracted `main.c`'s ~150-line inline vision block (Phase
  34) into a reusable `src/multimodal/vision_pipeline.c` — both the CLI (`--image`) and the
  HTTP API now call the same `vision_pipeline_run()`. The server decodes an uploaded image to a
  temp file, runs the vision pipeline, and injects the result into the KV cache before
  generation, exactly mirroring the CLI's `--image` behavior; gracefully degrades to text-only
  with a logged warning if the server has no `--vision`/`--proj` configured.
- Deliberate deviations from the original plan, both justified and documented here rather than
  silently assumed: (1) the plan's "web UI framework: llama.cpp uses SvelteKit" was matched with
  a **plain Vite+Svelte SPA** (no SvelteKit/SSR) since this server only needs static files — SSR
  would add machinery with no benefit here. (2) "load the vision model once at startup" was
  **not** implemented as a persistent in-memory model; `vision_model_load_encoder/projector` are
  `mmap`-based (confirmed by reading `vision_weights_load.c`), so reloading per-request is cheap
  (an mmap() call + OS page cache), and keeping `vision_pipeline_run()`'s combined load+run
  interface (matching the CLI's existing, already-tested pattern) was simpler and lower-risk than
  splitting load/run across the connection-thread boundary.
- `npm audit` flags 7 moderate/high advisories in the pinned Svelte 4 / Vite 5 dev-dependency
  tree; all are either dev-server-only (esbuild, not exposed since this is a static build with no
  dev server shipped) or Svelte SSR-specific (this is a pure client-side SPA, no SSR at all) — a
  deliberate, documented trade-off against a costly Svelte 5 rewrite for zero applicable benefit.
- Verified: real end-to-end Playwright testing (not just unit tests) against a live server —
  page load, sending a message, streaming response, Stop button (mid-generation cancel), sampling
  params panel, dark/light theme toggle all confirmed working via screenshots. `curl`/`curl --raw`
  confirmed `GET /`, `/assets/*`, `--web-ui off` (404s), and `--static-dir` dev-mode serving.
  Golden output (Paris/Berlin) re-verified unaffected by the vision-pipeline extraction on both
  the CLI and (new) HTTP API paths. Image upload's graceful-degradation path (no
  `--vision`/`--proj` configured) verified end-to-end with a real base64 PNG; the full
  image-understanding path is *not* end-to-end verified in this environment (no vision.bin/
  projector.bin/vision-capable GGUF available to download) — the extracted logic is otherwise
  unchanged from the already-tested CLI code path (`tests/test_vision_components.c`,
  `test_vision_e2e.c`), so this is a scoped, disclosed gap, not a silent one.
- `make release/test/debug` green on gcc and clang throughout. New tests:
  `tests/test_static_assets.c` (hand-written fake manifest, not the real bundle — keeps
  `make test` Node-free), `tests/test_data_url.c`, and content-parts-parsing cases added to
  `tests/test_api_server.c`.
- Areas: `webui/**` (new), `tools/gen_webui_bundle.c` (new), `src/api/{static_assets,data_url,
  webui_bundle_generated}.c` + `include/api/{static_assets,data_url,webui_bundle}.h` (new),
  `src/multimodal/vision_pipeline.c` + `include/multimodal/vision_pipeline.h` (new, extracted
  from `main.c`), `src/api/{json_parse,http_server}.c`/`include/api/{chat_request,server_config}.h`
  (content-parts parsing, vision wiring), `src/cli/{args,main}.c`/`include/cli/args.h` (new
  flags), `CMakeLists.txt` (new sources), `Makefile` (`webui-bundle` target), `.gitignore`
  (`webui/node_modules/`, `webui/dist/`).
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.3: CLI/REPL polish (color, progress, live tok/s, markdown)
- What: Added `--color <auto|always|never>` (respects `NO_COLOR`), a coarse 4-stage model-load
  progress indicator (TTY in-place `\r` updates, plain one-line-per-stage otherwise), a live
  tok/s indicator updated per-token during REPL generation, and incremental markdown rendering
  (bold, inline code, fenced code blocks) for REPL output — handling constructs whose delimiters
  are split across separate streamed token pieces (e.g. `"**bo"` + `"ld**"`). Regrouped
  `--help` output into sections (Model & Generation / Hardware / Server / Multimodal / Memory &
  RAG / Output) with worked examples. Only the REPL path is affected — the one-shot `--prompt`
  path and the HTTP API's SSE callback are untouched, matching the plan's scoping.
- A manual TTY smoke test (via `script`) surfaced a real UX rough edge in the first cut of the
  markdown renderer: an unterminated code fence (common whenever `max_tokens` cuts generation off
  mid-block) buffered the ENTIRE rest of the response until the final flush, defeating live
  streaming. Fixed with a bounded safety valve (`MD_MAX_PENDING_UNCLOSED`, 4 KiB) — an opening
  marker that hasn't found its close within that many buffered bytes is flushed as plain text
  instead of waiting indefinitely, covered by a new test
  (`test_unclosed_fence_eventually_flushes_without_waiting_for_end`).
- Why: closes the CLI-polish gap vs. leading engines (colored output, progress bars, live stats,
  markdown rendering) identified in the original UI/UX audit.
- Areas: `src/cli/{color,progress,live_stats,md_render}.c` + matching headers (new),
  `src/cli/{args,main,repl}.c`/`include/cli/args.h` (new `--color` flag, progress-stage hooks,
  REPL composite callback), `CMakeLists.txt` (four new CLI sources registered),
  `tests/test_{color,progress,md_render}.c` (new).
- Result: `make release/test/debug` green on gcc and clang; new unit tests cover color
  resolution, progress-line formatting, and markdown rendering (including the split-delimiter
  and unclosed-construct edge cases); manual REPL smoke test under a real pty (`script`)
  confirmed live progress stages, live tok/s updates, and markdown styling all render correctly;
  one-shot `--prompt` golden output (Paris, 64 tok/s) unaffected.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.1: HTTP API hardening + concurrency rearchitecture
- What: Added CORS (`--cors`/`--cors-origin`), optional API-key auth (`--api-key`), `/metrics`
  (Prometheus text exposition, `--metrics`), `/docs` + `/openapi.json` (static OpenAPI 3.0 +
  hand-rolled docs page, no Swagger-UI dependency), and `POST /v1/chat/completions/cancel`
  (stop an in-flight generation). Rearchitected `http_server.c` from one serial listener thread
  to a detached-per-connection-thread model, with a `generation_mutex` serializing only the
  actual `generate_with_callback` calls (a second concurrent chat request gets `429` immediately
  rather than blocking). `generate_with_callback`'s `TokenCallback` now returns `int` (0 =
  continue, nonzero = stop early) so cancellation can actually halt the generation loop, not just
  the client's view of it. Raised `HTTP_MAX_BODY_BYTES` 512 KiB → 8 MiB ahead of Phase 22.2's
  image uploads.
- Real end-to-end testing (live server + `curl`/`curl --raw` against the SmolLM2-135M demo
  model) surfaced and fixed four pre-existing HTTP protocol bugs the "untested socket layer"
  label had been masking — see `docs/ai/mistakes.md` (2026-07-15 entry) for the full list
  (recv-loop hang on bodyless GETs, `Content-Length: 0` sent before the real non-streaming body,
  a false `Transfer-Encoding: chunked` claim over unframed bytes, and no `SIGPIPE` handling).
  Also fixed a bug introduced by the new cancel feature itself: the id registered internally was
  the raw id, but clients only ever see the `"chatcmpl-"`-prefixed id in the stream, so a client
  echoing it back to the cancel endpoint would never match — fixed by registering under the same
  public, prefixed id the client observes.
- Why: closes the biggest UI/UX gap vs. leading engines (CORS/auth/metrics/docs/cancel) and is a
  prerequisite for Phase 22.2's web UI (needs real CORS + a working stop button).
- Areas: `src/api/{server_config,cors,auth,metrics,openapi,cancel,http_server,sse_stream}.c` +
  matching headers, `src/transformer/generate.c`/`include/transformer/generate.h` (callback
  return type), `src/cli/{args.c,main.c}`/`include/cli/args.h` (new flags), `CMakeLists.txt`
  (six new API sources registered), `tests/test_{cors,auth,metrics,cancel,openapi}.c` (new).
- Result: `make release/test/debug` green on gcc and clang (ASan/UBSan); a ThreadSanitizer build
  of the full engine showed zero race warnings under concurrent traffic (metrics/models/health/
  docs requests firing while a generation held the mutex, correctly getting `429` for concurrent
  generation attempts); golden output ("Paris"/"Berlin") unchanged through both the CLI and the
  now-hardened API; manual verification of CORS allow/deny, auth 401/200, streaming, non-
  streaming, and mid-stream cancellation all confirmed working via live curl sessions.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.0: docs groundwork for Web UI & API/DX hardening
- What: Recorded the Phase 22 plan (web chat UI via Vite+Svelte embedded in the binary, HTTP API
  hardening with a concurrency rearchitecture, CLI/REPL polish, mandatory design-QA + regression
  screenshots) before any code changes. Justified and documented the `api` scope in
  `tool-sync-policy.md` (adapter files to be added once 22.1 lands).
- Why: `docs/ai/**` is canonical and updated before code per policy; this is a large multi-phase
  effort and needs the decision trail written down first.
- Areas: `docs/ai/decision-log.md`, `docs/ai/project-overview.md`, `docs/ai/tool-sync-policy.md`.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-06-19 — Portable `make dist` build + GitHub Release pipeline
- What: Added a portable distribution build and a release workflow that attaches a prebuilt
  x86-64 Linux binary to a GitHub Release. New `make dist` target compiles the bulk at
  `-march=x86-64-v2` with per-file SIMD ISA flags (AVX2/AVX-512/VNNI) so runtime `simd_dispatch`
  lights up the best tier on the host; `simd_dispatch.c` is compiled at the baseline with
  `-DTN_FORCE_DISPATCH_ALL` (new guard, no SIMD codegen there) so all branches are present;
  static `-static-libstdc++ -static-libgcc` leaves only libc/libm deps. Added a `--version`/`-v`
  flag (works without `--model`) and a `-DPZ_VERSION` build stamp (banner no longer hardcodes
  "Phase 16"). CMake gains an off-by-default `PZ_DIST` option mirroring the Makefile.
- Why: user asked for a prebuilt x86-64 binary on a GitHub Release, tested thoroughly; the
  existing `-march=native` release is not distributable on varied CPUs.
- Areas: `Makefile` (dist target, per-TU ISA rules, version stamp), `src/math/simd_dispatch.c`
  (`TN_FORCE_DISPATCH_ALL`), `src/cli/{args.c,main.c}` + `include/cli/args.h` (`--version`),
  `CMakeLists.txt` (`PZ_DIST`, `PZ_VERSION`), `.github/workflows/release.yml` (new),
  `.github/workflows/ci.yml` (dist build-check), `docs/RELEASING.md` (new).
- Result: gcc release/test(46)/debug/dist and clang release/debug/dist green; portable binary
  links only libc/libm; golden output (France→Paris, Germany→Berlin) correct across
  scalar/avx2/avx512f/vnni and T=1/2/8 on the SmolLM2-135M F16 model.
- Commit/PR: on branch `claude/x86-64-github-release-8xduj2`.

### 2026-06-14 — Docs reflect dense GGUF support (SmolLM2 + generic loader)
- What: README, ROADMAP, and project-overview said the engine runs only BitNet and
  DeepSeek-V2-Lite, but the benchmark docs (`.claude/BENCHMARK_SUMMARY.md`,
  `docs/PERFORMANCE_CEILING_REPORT.md`) already benchmark **SmolLM2-135M-Instruct F16**
  (dense GGUF) up to 83.79 tok/s, and `config_from_gguf()` in `src/core/gguf_loader.c` is
  architecture-agnostic. Added a third support tier: dense GGUF transformers (Llama-family)
  via the generic loader, with SmolLM2 as the verified model and other architectures flagged
  as loads-but-untested. MoE/MLA acceleration remains DeepSeek-V2-specific.
- Why: docs understated actual, already-tested capability; user asked for the correct picture.
- Areas: `README.md` (intro, new "Dense GGUF Models" section, footer), `.github/ROADMAP.md`
  (perf snapshot), `docs/ai/project-overview.md` (Purpose). Lean adapters (AGENTS/copilot/
  GEMINI/CLAUDE) left as-is per tool-sync-policy — they describe the targeted/special-cased
  architectures, not an exhaustive model list. Historical benchmark addenda left untouched.
- Branch: `claude/readme-llm-support-docs-3tg13v`.

### 2026-06-14 — README accuracy pass + repo best-practices + docs reorg
- What: (1) Corrected README intro to match canonical scope (BitNet + DeepSeek-V2-Lite
  GGUF + vision/agentic/RAG), kept "written in C", reframed Python as temporary
  dev/test tooling (zero-Python goal), added LLM-agnostic goal. (2) Reconciled the
  Phase 21 HTTP API claim to 🔄 partial/experimental across README, ROADMAP, and
  project-overview (it is real and wired but serial/loopback-only/untested-in-CI).
  (3) Added community-health files: `.github/CODEOWNERS`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`, `.editorconfig`, `CITATION.cff`. (4) Moved 27
  archival/design/report `.md` files out of the repo root into
  `docs/{architecture,phases,reports,weight-loading}/` and `docs/`, leaving 8 entry-point
  docs at root; rewrote all inbound markdown links path-aware and fixed 4 dangling links
  (verified 0 dangling repo-wide).
- Why: README/roadmap/overview contradicted each other and the tree; root had 35 `.md`
  files hurting discoverability; repo was missing standard GitHub best-practice files.
- Areas: `README.md`, `.github/ROADMAP.md`, `docs/ai/project-overview.md`, `.editorconfig`,
  `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/config.yml`,
  `CITATION.cff`, and `docs/{architecture,phases,reports,weight-loading}/**`.
- Branch: `claude/readme-accuracy-review-y9jk7u`.

### 2026-06-07 — Document branch-hygiene convention
- What: Added a "Version control & branch hygiene" section to `engineering-rules.md` (delete
  merged branches; enable auto-delete-head-branches; avoid flag/placeholder branches; don't
  commit artifacts/models/logs).
- Why: post-merge cleanup surfaced redundant branches; this sandbox's git proxy blocks ref
  deletion, so the convention + the repo auto-delete setting prevent future accumulation.
- Areas: `docs/ai/engineering-rules.md`. Canonical-only (adapters stay lean per tool-sync-policy).

### 2026-06-07 — Cross-tool AI development system
- What: Added `docs/ai/**` canonical docs + Claude/Copilot/Antigravity adapters.
- Why: one source of truth; continuity across Claude Code, GitHub Copilot, Antigravity.
- Areas: `docs/ai/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.github/`, `AGENTS.md`,
  `gemini/GEMINI.md`, `.agents/`.
- Commit/PR: (this change) — see commit checkpoints in PR to `master`.

### 2026-06-07 — Green CI + regression verification (PR #6, merged `cb9fa52`)
- What: Fixed the CI cascade and verified no regression across SmolLM2/BitNet/DeepSeek.
- Why: CI had never run to completion; ensure merge-safety and no regression.
- Areas: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`,
  `tests/test_vision_components.c`, `Makefile`, `src/core/run_state.c`,
  `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
- Result: all 7 CI checks green on PR #6 and on `master`; secrets scan clean (215 commits).
