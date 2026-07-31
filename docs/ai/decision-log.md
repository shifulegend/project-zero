# Decision Log — project-zero

> Timestamped architectural / tooling / workflow / process decisions. Newest first.
> Read at session start. Last updated: 2026-07-31.

### 2026-07-31 — Qwen3-8B brought up, benchmarked against llama.cpp, found slower, and root-caused instead of shipping a hand-wave

- Decision: user asked to run Qwen3-8B-Q4_K_M on pz and compare with llama.cpp. First attempt hit a real
  OOM (dequanting every Q4_K tensor to F32), then real garbage output (missing Qwen3 QK-norm, then a shared
  weight-type flag misdispatching Q4_K_M's mixed-per-layer quant types) — all three fixed and verified, not
  worked around. Once correct, pz measured 2.5-2.65 tok/s vs. llama.cpp's 4.0-4.8 on the same file/prompt/
  threads/host — user explicitly required a real fix, not documentation of the gap, so a deep RCA was run:
  read llama.cpp's actual AVX2 Q4_K kernel source directly (found it's nearly byte-identical to pz's own),
  measured T=1 vs T=4 scaling (3.5-4.5x, ruling out thread-pool overhead), and traced the real difference to
  llama.cpp's CPU backend transparently repacking Q4_K rows into interleaved 8/16-row super-blocks at load
  time with matching multi-row GEMV kernels — a genuinely different GEMV batching strategy, not an
  instruction-choice difference.
- Four attempts were tried in total, all honestly measured against the same 2.49-2.65 tok/s baseline:
  (1) software prefetch on the single-matrix dispatch — no measurable change; (2) a from-scratch,
  correctness-verified AVX-512 VNNI kernel — measured 33-40% *slower* (`dpbusd`'s raw output needs a
  separate scale-multiply this kernel's AVX2 path already fused for free), reverted; (3) a row-batched
  "grouped8" Q4_K repack mirroring llama.cpp's interleaved memory layout — measured *worse* (2.16 tok/s
  with prefetch, 1.72 without), reverted, because repacking only moved bytes around without adding real
  multi-row register interleaving; (4) a genuine bug fix — `attention.c`/`ffn.c` claimed in their own
  comments to quantize the shared input once and reuse it across Q/K/V (and gate/up), but that had only
  ever been wired up for the ternary path, so the Q4_K dense path (Qwen3-8B's actual path) was silently
  re-quantizing 3x/2x per layer — fixed and kept (verified via full gcc+clang release/test/debug, kept
  despite measuring no throughput change, since the eliminated O(n) work was always 2-3 orders of
  magnitude smaller than the O(d×n) matmul it fed). Real test coverage for Q4_K matmul was added
  regardless (`tests/test_q4k_matmul.c`) since it had none before this pass.
- User explicitly said "Fix it" after being shown the 4-attempt status, so a 5th attempt was made: a
  genuine multi-row interleaved Q4_K kernel (shared Q8K activation loads + independent per-row
  accumulator chains — the exact mechanism identified in the RCA, implemented directly rather than via a
  memory repack). Tried at 4-row granularity first: measured 2.00 tok/s, *worse* than baseline; disassembly
  showed real register spills even on this host's 32-register AVX-512VL file. Scaled back to 2-row (halving
  live-range pressure, confirmed via disassembly to spill much less): still measured *worse*, 2.19 tok/s. A
  clean same-session back-to-back A/B (baseline 2.72 -> 2-row 2.19 -> 4-row 2.00, all three within ~15
  minutes on the same host state) confirmed this was a real, monotonic regression, not noise. Reverted both
  variants (`git stash push` + `git stash drop`); working tree verified clean against the last pushed
  commit before and after.
- Status: RCA complete and evidence-backed across five attempts (one kept: the redundant-quant bug fix;
  four reverted: VNNI, grouped8, and both multi-row variants). The theory behind attempt 5 was directly
  validated by reading llama.cpp's source (shared loads + independent accumulators is genuinely what their
  kernels do) but the direct implementation of that idea still lost here — meaning the real win most likely
  needs BOTH halves of llama.cpp's design at once (repack + multi-accumulator together, not either alone,
  since both isolated attempts — grouped8 for the repack half, attempt 5 for the accumulator half — lost
  independently), verified on a host with real profiling tools (`perf`/`vtune`, neither available in this
  environment). That is a materially larger effort than anything attempted in this session and is flagged
  explicitly, not silently parked. Full chain in `docs/ai/mistakes.md` (2026-07-30/07-31 entries) and
  `README.md`'s Qwen3-8B section.
- Also fixed in the same pass: this host's clang-18 install was missing `libclang-rt-18-dev` (stale apt
  index — `apt-get update` resolved it), which had been silently skipping clang's ASan/UBSan test
  coverage. Not caused by this session's changes, but caught and fixed while verifying them, per the
  project's "fix any bug found in this pass" policy.

### 2026-07-24 — Full threads×SIMD×classifier sweep (52 configs); RCA for why INT4 classifier is slower than INT8/BF16
- Decision: after the F16/BF16 kernel fix (below) closed the single-sample pz-vs-llama.cpp gap,
  did the full sweep the user asked for: all 48 project-zero `--threads`(1-4)×`--simd`(4)×
  `--classifier`(3) combinations, plus llama.cpp's 4 thread counts (its only runtime-selectable
  axis) — in-repo report at `docs/design/reports/sweep-2026-07-24.html`, raw CSVs at
  `docs/design/reports/sweep_2026-07-24/`, screenshots at
  `docs/design/screenshots/sweep_2026-07-24/`.
- Methodology bug found and fixed mid-effort (full writeup in `mistakes.md`): the first pass used
  a short prompt + `--max-tokens 30`; the model hit EOS after 7 tokens, so every measurement was
  noise from a sub-200ms window. Reran all 52 configs with a long-form prompt and
  `--max-tokens 250` so every run generates 251 real tokens. This changed the numbers materially
  (peak dropped from a noise-inflated 84.7 to a real 69.0 tok/s) and changed the cross-engine
  conclusion from "one wide divergent gap" to "within ~3% at every thread count." A second bug
  (concurrent screenshot capture starving a running benchmark of CPU) was found and fixed the
  same way — never run two CPU-bound captures at once on this 4-core sandbox.
- RCA: INT4 classifier measured slower than INT8/BF16 at all 16 thread×SIMD combinations, holding
  even after the methodology fix (ruling out short-run noise as the cause) — this contradicts the
  classifier-quantization comment in `src/cli/main.c` estimating INT4 as faster. Root-caused
  against the actual kernel code (`src/math/parallel_matmul.c`'s `matmul_i4_task`, line ~570):
  1. INT4 weights are packed 2-per-byte, so every dot product needs a nibble-unpack step first.
     The fast unpack path (VBMI `vpermb`+`vpmultishiftqb`, 3 instructions/64 weights) needs
     AVX-512 VBMI, which this CPU doesn't have (confirmed via `/proc/cpuinfo`, same gap as the
     calibration SIGILL below) — so it falls back to the ~10-instruction SSE-interleave unpack
     (mask+shift+four `unpacklo`/`unpackhi`/`insert128` ops) on every 64 weights. INT8's kernel
     (`matmul_i8_task`) needs zero unpacking (1 byte = 1 weight already), so this is a real,
     large, INT4-only compute tax.
  2. That tax buys nothing on this model: SmolLM2-135M's classifier is 54.0 MiB at BF16, 27.0 MiB
     at INT8, 13.5 MiB at INT4, against a 33 MiB measured L3. INT8 already shrinks the classifier
     to fit in cache; INT4 shrinking it further saves bandwidth that was never the bottleneck
     (compute-bound, not bandwidth-bound, at this size) — so INT4 pays a real unpack cost to save
     bandwidth nobody needed.
  3. Confirmed the `--simd` flag doesn't affect this kernel at all — `simd_dispatch.c`'s dispatch
     table only covers the ternary-weight ops (rmsnorm/softmax/ternary matmul), not the classifier
     path, which explains why INT4 is uniformly slowest across all four SIMD rows in the heatmap
     rather than four separate coincidences.
  4. Separate, not-fully-confirmed qualitative finding: INT4 runs degenerate into looping
     repetition before the token budget is used up (BF16/INT8 stay coherent for all 251 tokens) —
     hypothesized as INT4's coarser per-row quantization (16 levels vs INT8's 256) pushing greedy
     decoding into a self-reinforcing loop, not yet verified against logit distributions.
- Status: root cause understood and documented; not fixed (a faster non-VBMI unpack path, e.g. a
  `pshufb`-based lookup table, is the natural fix but wasn't attempted this pass — flagged as a
  follow-up, added to `README.md`'s Help Wanted table rather than silently dropped).

### 2026-07-24 — RCA + fix: single-accumulator FMA dependency chain in the F16/BF16 GEMV kernels was latency-bound, not throughput-bound
- Decision: user noticed project-zero (54.95 tok/s) measured behind llama.cpp (58.8 tok/s) in a
  side-by-side SmolLM2-135M-Instruct F16 screenshot comparison and asked for a detailed RCA before
  any fix. Rejected the single-sample reading as a basis for either conclusion or action — repeated
  each engine 3x at T=1 and T=2 first. Result: at T=2 the two engines were statistically
  indistinguishable on this host (pz 56.20/56.81/57.92 vs llama.cpp 56.0/57.1/52.3 — pz's mean
  slightly *ahead*), so the original single-run "pz is behind" reading was itself run-to-run noise
  on a shared/virtualized host, not a reproducible deficit. At **T=1** (0 OS worker threads in
  project-zero's thread pool — see `threading/thread_pool.c`'s "caller-participates" design — so
  this isolates raw serial kernel throughput with zero dispatch/sync overhead from either engine),
  a real, consistent gap *did* reproduce: pz 29.85/28.21/29.98 (mean 29.35) vs llama.cpp
  32.4/34.0/33.0 (mean 33.13), ~13% every time, not noise.
- Root cause, found by reading llama.cpp's actual kernel source (cloned+built locally for this
  comparison) rather than guessing: `ggml_vec_dot_f16` (`ggml/src/ggml-cpu/vec.cpp`) unrolls
  **4 independent FMA accumulators** per SIMD width (`GGML_F16_STEP=32`/`GGML_F16_EPR=8` on AVX2 →
  4 accumulators of 8 each) specifically to keep multiple FMAs in flight and hide the ~4-5 cycle
  FMA latency (far longer than the FMA unit's 1-cycle throughput) — a standard, well-known
  technique. `parallel_matmul_f16` (`src/math/matmul_f16.c`) and `parallel_matmul_bf16`
  (`src/math/parallel_matmul.c`, the LM-head classifier kernel — the single largest matmul per
  token for most dense models, dim × vocab_size) both used a **single** accumulator per output row,
  making every FMA in the loop wait on the previous iteration's result: latency-bound, not
  throughput-bound. Same bug, same shape, in both kernels.
- Fix: rewrote both kernels' AVX2 and AVX-512 paths to use 4 independent accumulators (32-wide
  AVX2 / 64-wide AVX-512 outer loop, matching ggml's exact AVX2 layout), summed together only at
  the end, falling back to the original single-accumulator loop for the remainder and the existing
  scalar/8-wide tails unchanged. This is a pure reassociation of the same sum (same FMA count, same
  terms) — no algorithmic or precision-relevant change beyond normal floating-point reassociation.
- Verified, not assumed: re-ran the same T=1/T=2 x5-rep sweep after the fix.
  T=1: 34.20/34.64/32.75/32.16/34.93 (mean 33.74) — closes essentially all of the ~13% gap
  (llama.cpp's own T=1 mean: 33.13). T=2: 64.90/66.95/60.18/54.42/64.42 (mean 62.17) — now
  consistently *ahead* of llama.cpp's T=2 mean (55.13), not behind. Golden output unchanged
  ("The capital of France is Paris.") on every run before and after. Full `make release/test/debug`
  clean on **both gcc and clang** from a clean tree each time; additionally re-ran the real
  SmolLM2 model under the ASan/UBSan debug build specifically to stress-test the new
  4-accumulator load/FMA sequences for any out-of-bounds access — clean.
- Scope note: only the F16 and BF16 kernels were touched (the two formats this specific model and
  comparison actually exercise, and the two confirmed by evidence). The generic F32 kernel and the
  ternary/INT8/INT4 packed kernels were not inspected for the same single-accumulator pattern in
  this pass — flagged as a follow-up worth checking, not silently assumed fine.
- Status: ACCEPTED.

### 2026-07-24 — Added Qwen3-MoE (e.g. Qwen3-30B-A3B) GGUF loading support, fixing issue #32
- Decision: issue #32 (jpsoto) reported `[gguf_loader] missing tensor 'blk.0.ffn_gate.weight'`
  loading `Qwen3-30B-A3B-Q4_K_M.gguf`. Root cause: `weights_from_gguf()` only special-cased
  `deepseek2` and `qwen35`; `qwen3moe` (Qwen's actual MoE arch) fell through to the generic dense
  loader, which expects `blk.N.ffn_gate.weight` — MoE GGUFs instead have `blk.N.ffn_gate_inp.weight`
  (router) + stacked `blk.N.ffn_gate_exps.weight` (per-expert). This needed real new-architecture
  support, not a one-line patch — planned and implemented with a Plan-agent design pass first.
- Mid-implementation discovery (a real correctness risk, not an assumption to wave away): Qwen3's
  head_dim is independent of `dim/n_heads` (e.g. Qwen3-30B-A3B: dim=2048, n_heads=32 →
  dim/n_heads=64, but the real head_dim is 128), meaning Q/K/V projections are wider than `dim`
  (q_rows = n_heads*head_dim = 4096 ≠ dim). Reusing `attention_forward()`'s generic GQA branch —
  which sizes `s->q`/`s->xb`/`s->xb2` and the KV cache off `config_head_dim(cfg)`==dim/n_heads —
  would have silently overflowed those buffers. This is the *exact* bug class `mla_run_state.c`
  and `qwen35_run_state.c` already document and fix for their own independent-head-dim cases
  (confirmed by reading `qwen35_attention.c`'s `q35_full_attn_forward`, which uses fixed generous
  **static local buffers** and its **own** `q35_key_cache`/`q35_value_cache` for exactly this
  reason). Followed that established precedent rather than inventing something new:
  - New `MoEConfig.has_qk_norm` flag (like `has_mla`/`has_linear_attn`), reuses the existing
    `attn_head_dim` field (already there for qwen35) rather than adding another head-dim field.
  - New dedicated `qwen3moe_attention_forward()` (`src/transformer/qwen3moe_attention.c`) — plain
    GQA + per-head QK-norm before RoPE + full (non-partial) rotary, own static Q/K/V buffers, own
    output projection (input width q_rows, not dim — `attn_output.weight` is `[dim × q_rows]` for
    this arch, confirmed against qwen35's identical `wo` sizing pattern).
  - New `qwen3moe_run_state_alloc()/_free()` (`src/core/qwen3moe_run_state.c`) — own
    `qwen3moe_key_cache`/`qwen3moe_value_cache`/`qwen3moe_rope_freq`, correctly sized with
    `mc->attn_head_dim`; `main.c` passes `skip_kv_cache=mc.has_qk_norm` so the generic (wrongly
    sized) `key_cache`/`value_cache` is never allocated for this arch, mirroring qwen35's approach.
  - New `weights_from_gguf_qwen3moe()` loader: per-layer QK-norm tensors, GQA projections sized by
    `q_rows`/`kv_rows` (not `dim*dim`), and the deepseek2 branch's per-expert stride-slicing scheme
    reused verbatim (no shared experts, no leading dense layers for this arch — all layers routed).
  - Extracted `load_bf16_embedding_and_output()` out of `weights_from_gguf()`'s former
    deepseek2-only inline block so qwen3moe reuses it instead of duplicating ~35 lines.
- Testing (two-tier plan from the design pass, both required): (a) new
  `tests/test_gguf_loader_qwen3moe.c` — hand-builds a tiny synthetic in-memory `GGUFHeader`
  (`GGUFHeader`'s `tensors[]`/`meta[]` are plain fixed-size embedded arrays, confirmed in
  `include/core/gguf_reader.h`, so no real `.gguf` byte-serialization or multi-GB download is
  needed) and exercises `config_from_gguf`/`moe_config_from_gguf`/`weights_from_gguf` directly,
  asserts per-expert stride correctness (`moe_w1[0][0] != moe_w1[0][1]`, exact stride match), and
  runs one real token through `transformer_forward()` end-to-end (attention + MoE FFN + classifier)
  asserting finite logits — this is the first loader-level test in the repo (previously only
  `test_moe.c` existed, and it bypasses `gguf_loader.c` entirely). (b) Real-model verification
  against the actual `Qwen3-30B-A3B-Q4_K_M.gguf` (~18-19GB) from jpsoto's report, cross-checked
  against llama.cpp — **not run in this environment** (disk/network constraints for an 18GB+
  download were flagged up front); left as an explicit follow-up, tracked here rather than silently
  treated as done.
- Verification performed: `make release && make test && make debug`, **gcc and clang**, from a
  clean tree each time (`make clean` first — a prior pass in this same session briefly ran `make
  debug` before `make release`/`test` and hit a real but unrelated pre-existing Makefile fragility:
  `test_q2k_matvec`/`test_api_server` force `CFLAGS_RELEASE` for their own C++ link step regardless
  of global flags, so linking them against debug-instrumented `chat_template.o` — left over from
  the earlier `make debug` run — produced `undefined reference to __ubsan_handle_*` errors; the
  documented `release → test → debug` order avoids this). Zero new warnings, ASan/UBSan clean on
  both compilers, `test_gguf_loader_qwen3moe` passes on both including the real forward-pass token.
  This session's own test infrastructure bug caught along the way: the test initially crashed with
  a null-function-pointer SEGV (`pc=0x0`) because it never called `tn_simd_init()` —
  `tn_rmsnorm`/`tn_vec_dot`/etc. are function pointers that stay `NULL` until that runs (every real
  entrypoint calls it in `main.c`); fixed in the test, not a bug in the new loader/attention code.
- Risks/unknowns flagged honestly, not glossed over: the exact GGUF metadata key names
  (`qwen3moe.expert_count`, `.expert_used_count`, `.expert_feed_forward_length`,
  `.attention.key_length`) are inferred from the `deepseek2`/`qwen35` naming convention, not
  confirmed against a real file or llama.cpp's `qwen3moe` key table — this is exactly what the
  deferred real-model verification (b) above needs to confirm before fully trusting this fix
  against the actual reported file. Router top-k renormalization (`norm_topk_prob`) and QK-norm
  weight shape (assumed one `[head_dim]` vector shared across heads per layer) are likewise
  unconfirmed against a real file.
- Status: ACCEPTED for the synthetic-fixture-verified code path; real-model closure of issue #32
  is a tracked follow-up, not yet done.

### 2026-07-24 — Startup banner now always prints; piped/redirected output finally gets the plain text it was supposed to
- Decision: the 2026-07-16 entry below states the banner's intended invariant as "piped/redirected
  output (not a TTY) still gets plain text, so scripted/automated usage is unaffected" — but the
  actual code (`src/cli/banner.c:102`, `if (!is_tty) return;`) made `tn_banner_print()` a complete
  no-op for non-TTY output, printing nothing at all rather than plain text. This is why benchmark
  screenshots built from `tools/make_screenshot.py` (which renders a pre-captured *redirected*
  stdout `.txt` file, never a real TTY) never showed the "PROJECT ZERO" branding — the banner was
  never in the captured text to begin with. User explicitly asked that branding never depend on the
  CLI's invocation path (TTY or not, one-shot or REPL or `--server`).
- Fix: split `tn_banner_print()` following this codebase's own established idiom for TTY-dependent
  CLI rendering (`src/cli/progress.c`'s `tn_progress_format()` — a pure formatter taking `is_tty`,
  branching internally, no I/O of its own). Added `tn_banner_format_plain()` (`banner.c`/`banner.h`)
  — reuses the already-zero-I/O `compose_banner()` to render the same glyph content as a single
  plain block (`'#'`/`' '` characters, no ANSI, one trailing `\n` per row), with no cursor-movement
  escapes and no `nanosleep` pacing. `tn_banner_print()` now calls this plain path when `!is_tty`
  instead of returning early; the animated reveal+shimmer path (cursor-up `\x1b[%dA`, clear-line
  `\x1b[K`) is unchanged and still only reachable when `is_tty` is true, so no escape bytes can leak
  into non-terminal output. `src/cli/main.c`'s call site had its `if (stdout_is_tty) { ... }` wrapper
  removed — `tn_banner_print()` is now called unconditionally and branches internally.
- Verified, not just read: built with gcc, ran `./adaptive_ai_engine --model <bogus path> --prompt
  hi > out.txt 2>&1` (redirected, matching how benchmark captures are actually made) and confirmed
  the plain glyph block appears at the top of `out.txt` with zero `\x1b` bytes (`grep -c $'\x1b'` =
  0). Separately captured the same invocation through a real pty (`script -qec ...`) and confirmed
  56 ANSI escape sequences still present — the animated path is provably unchanged. New
  `tests/test_banner.c` (mirrors `tests/test_progress.c`'s pure-formatter-only convention) passes
  clean under gcc + ASan/UBSan: no ESC bytes, exactly 5 newlines, at least one glyph pixel, NULL/
  tiny-buffer safety.
- Status: ACCEPTED. Supersedes the "piped/redirected output ... still gets plain text" claim in the
  2026-07-16 entry below, which was aspirational rather than actually true until now — see
  `mistakes.md`'s existing writeup of the same gap (2026-07-16 entry there) for the full narrative
  of how the invariant was first stated.

### 2026-07-17 — Commit-bisected the 2.74 tok/s question instead of resting on a diff-based argument
- Decision: user asked for direct proof rather than accepting the earlier `git diff`-based
  reasoning — checked out the exact commit behind the 2.74 tok/s screenshot (`ce8e90d`) plus its
  parent (`34d3ac9`, pre-VNNI-kernel) in isolated git worktrees, rebuilt each, and reran the
  identical benchmark command, capturing real screenshots (not just log text) via the fixed
  capture tool. `ce8e90d` measured 1.40 tok/s today (not 2.74); `34d3ac9` measured 0.12 tok/s,
  matching the historical pre-VNNI baseline almost exactly — a built-in control proving the
  methodology does detect real code-driven gaps when they exist, which makes the *absence* of a
  gap between `ce8e90d` and current HEAD meaningful rather than a failure to look hard enough.
  Full evidence and screenshot paths in `mistakes.md`.
- Status: ACCEPTED — this closes the "is it code or host" question definitively for this benchmark.

### 2026-07-17 — Investigated why `auto` measured 1.19-1.24 tok/s vs the README's earlier 2.74 tok/s, instead of assuming noise
- Decision: user asked directly why the post-fix `auto` number was so much lower than the earlier
  benchmark's 2.74 tok/s for the identical default configuration. Ruled out the classifier fix as
  the cause via a real `git diff` (the default path is functionally unchanged), then got a live
  data point by re-running the exact original command rather than reusing old numbers or asserting
  "hardware noise" again. Found the re-run stalling for 5-10+ minutes at file-open and runtime-prep
  steps that historically take seconds, root-caused to first-touch-of-fresh-memory stalls (mmap
  populate, calloc zeroing) consistent with the underlying host being memory-pressured by other
  tenants — invisible to this guest's own memory/vmstat readings. Full evidence chain in
  `mistakes.md`. Confirmed final number: 1.24 tok/s, consistent with the classifier sweep's
  `auto` = 1.19 tok/s from the same session.
- Consequence: absolute tok/s numbers from this host are not comparable across separate sessions,
  only within a single, back-to-back sweep. Added this caveat to the README's benchmark section.
- Status: ACCEPTED.

### 2026-07-17 — Actually fixed the classifier no-op (materialization), not just warned about it
- Decision: the prior entry below (2026-07-16) fixed the classifier no-op bug with an explicit
  runtime warning, judging full per-format materialization too large a change to take on
  unprompted. User rejected that as insufficient ("Fix the no-op... No shortcuts, no honest
  explanations, no fooling around") and asked for the real fix. Implemented it: Q2_0-native models
  now materialize a BF16/INT8/INT4 classifier copy (dequantize Q2_0 → F32 → round to BF16 → reuse
  the existing `weights_build_classifier_quant()` for INT8/INT4), but **only when the user
  explicitly passes `--classifier`** (new `classifier_explicit` flag on `TnHardwareProfile`) — the
  default zero-copy path is untouched, so the multi-GB memory cost stays fully opt-in. Full detail
  in `mistakes.md`.
- Verified with real, differentiated numbers (strictly sequential, freshly recalibrated, idle host,
  4 threads): auto (zero-copy default) = 1.19 tok/s, bf16 = 1.13, int8 = 2.60, int4 = 2.62 tok/s.
  The fix is confirmed working — three formats now take three different code paths and produce
  three different results. Unexpected result: INT8/INT4 are ~2.2x faster than the zero-copy
  default despite reading more bytes, because the general VNNI int8/int4 kernel is more
  compute-efficient per element here than the specialized Q2_0 kernel — recorded in mistakes.md
  since it contradicts the naive "smaller format = faster" assumption.
- Verification: full gcc + clang release/test/debug, plus a from-scratch ASan/UBSan run with
  `$(LIB_OBJS)` genuinely instrumented (re-confirmed for this round too — the first attempt at this
  re-verification silently reused stale non-instrumented release objects via `make test` alone,
  caught by grepping the build log for the sanitizer flags on the changed files before trusting the
  green result) — all green, zero warnings, zero failures.
- Status: ACCEPTED.

### 2026-07-16 — Diagnosed the classifier BF16≈INT4 anomaly properly instead of accepting "noise" as the answer
- Decision: user rejected an initial "hardware measurement noise" explanation for why
  BF16/INT8/INT4 classifier runs measured identical tok/s, correctly pointing out that noise alone
  doesn't explain an exact match this precise. Traced it to a real bug: Q2_0-native models
  (Ternary-Bonsai-27B included) never actually read `--classifier` in `forward.c`'s dispatch — the
  flag is accepted, echoed, and even fed into a fake performance-ceiling calculation, but the
  actual LM-head matmul always uses the same zero-copy raw Q2_0 path regardless. Fixed (at the
  time) by adding an explicit runtime warning rather than implementing full per-format
  materialization — superseded by the real fix above, this warning-only version was not the final
  state. Separately confirmed the underlying virtualized host was *also* recycled mid-session
  (hardware genuinely changed between the thread-sweep and classifier-sweep measurements) — both
  facts are true, but the no-op dispatch bug is the actual reason the three classifier configs were
  indistinguishable, not the noise.
- Also changed the startup banner to match llama.cpp's actual (unconditional-in-a-TTY) behavior,
  after being asked why llama.cpp shows its banner in one-shot mode and pz didn't — read
  llama.cpp's source to confirm it has no interactive-vs-scripted distinction at all, then verified
  both the old and new pz behavior empirically via real pty captures rather than trusting the
  source read alone.
- Fixed a batch of pre-existing compiler warnings across `src/` and `tests/` on explicit reminder
  of the bug-fix policy (see mistakes.md for the full list) — all were genuinely pre-existing and
  unrelated to this session's main work, but "pre-existing" was correctly rejected as an excuse to
  leave them.
- Verification: full gcc + clang release/test/debug, plus a from-scratch ASan/UBSan run with
  `$(LIB_OBJS)` genuinely instrumented — all green, zero warnings, zero failures.
- Status: SUPERSEDED by the 2026-07-17 entry above.

### 2026-07-17 — Engineering highlights worth crediting explicitly (fact-checked before writing)
- Decision: user asked to record credit for specific technical work in this project's own docs.
  Declined to spawn research into social-media/psychological-persuasion framing for this (out of
  scope, and in direct conflict with this repo's own docs rule to stay factual and concise —
  `.claude/rules/docs.md`); instead verified each claim against the actual code/history before
  writing anything, and only recorded what checked out:
  1. **Q2_0/VNNI kernel connection (~29x speedup)**: Ternary-Bonsai-27B's GGUF tensors were tagged
     with a type ID mainline tooling reads as one format; computing real bytes-per-tensor against
     the file revealed PrismML's actual (non-canonical) packing, which turned out to be bit-for-bit
     compatible with this project's own AVX-512 VNNI ternary "w_enc bias trick" kernel. Connecting
     the two took Q2_0 matmul from ~1% of this host's DRAM bandwidth ceiling to ~29x faster. Fully
     documented in `mistakes.md` (2026-07-16, "Q2_0 matmul was 1% of this host's own bandwidth
     ceiling").
  2. **Activations quantized once per matmul call, not once per thread**: `parallel_matmul.c`'s
     dispatcher pre-quantizes the activation row a single time and passes the result to all worker
     threads (`parallel_matmul.c:145`), instead of each of T threads redundantly re-quantizing the
     same row — a real, verified efficiency choice, not new this session but genuine and worth
     citing when describing the kernel design.
  3. **Runtime ISA dispatch avoids cliff-edge failures on hardware lacking an extension**: the real
     fallback ladder is AVX-512VNNI → AVX-512 → AVX2 → scalar (ARM: dotprod → NEON), verified in
     `simd_dispatch.c` — not "SSE" as first described to me; corrected before writing it anywhere.
     The AVX-512VBMI CPUID-lying fix (`mistakes.md`, 2026-07-16) is part of this same dispatch
     discipline: verify a feature actually executes before trusting CPUID's claim of it.
  4. **Single binary, dual weight-format support** (native packed-ternary + dense/quantized GGUF,
     no per-model rebuild) is a real, inspectable property of this codebase's architecture — worth
     stating as a differentiator in the benchmark writeup, scoped as an existing engine capability
     rather than something delivered in this session.
  - **Correction (2026-07-17), on user pushback:** originally declined to credit "our VNNI ternary
    kernel is tighter than bitnet.cpp's own reference implementation," reasoning no benchmark
    against bitnet.cpp existed *in this specific session*. User correctly pushed back — the bar for
    "is this true and ours" isn't "did I personally re-run it this chat," it's "does real, verified
    evidence exist." It does: the 2026-06-21 decision-log entry above documents a controlled,
    same-SIMD/same-thread/same-precision measurement (methodology in `BENCHMARK_REPORT.md`
    Addendum AP) showing Project Zero beats Microsoft's own `bitnet.cpp` on BitNet b1.58 by
    +19-37% at every thread count, BF16 head-to-head — a real, apples-to-apples result, not the
    broader uncontrolled "up to 5.4x" headline number (which mixes each engine's own best SIMD
    pick and is a different, less rigorous claim). That controlled +19-37% number is the accurate,
    defensible form of claim #1 and is now credited on that basis.
- Status: ACCEPTED — see README.md's benchmark section for where this is surfaced publicly.

### 2026-07-16 — Fixed a real SIGILL crash found while benchmarking classifier precision (bf16/int8/int4)
- Decision: while collecting a `--classifier` throughput sweep for benchmark documentation, every
  run crashed with `SIGILL` on this host. Root-caused (via `gdb`) to AVX-512VBMI code executing
  despite this specific virtualized (Firecracker) host's CPUID advertising VBMI support it cannot
  actually retire. Fixed rather than working around it (e.g. by just disabling calibration or the
  classifier sweep), per the "any bug found gets fixed in the same pass" policy — this crash would
  hit any user on similarly-virtualized hardware on their very first run, not just this benchmark.
  Full technical detail in `docs/ai/mistakes.md`.
- Fix shape: added a one-time, execution-verified runtime check for AVX-512VBMI
  (`sigaction`+`sigsetjmp` self-test in `cpu_features.c`) instead of trusting CPUID alone, and
  converted the two consumer call sites (`bitunpack2_vnni.h`'s Q2_0/ternary unpack,
  `parallel_matmul.c`'s INT4 classifier unpack) from compile-time-only `#if TN_HAS_AVX512VBMI`
  branching to compiling both the VBMI and non-VBMI paths always, dispatching via the verified
  runtime flag.
- Status: ACCEPTED. Verified end to end: cleared the calibration cache and re-ran
  `--classifier int4` (the exact crashing scenario) — calibration now completes both phases and
  generation proceeds normally.

### 2026-07-16 — Reversed the "flag, don't fix" call on both remaining items after user pushback
- Decision: the prior entry below deferred 2 items (Q2_0 batch/MoE VNNI path, byte-level BPE
  detokenizer) as "flagged instead of fixed." User directly challenged that ("Why did u not fix
  the remaining 2?"). Re-examined both from scratch rather than re-justifying the original
  deferral, and implemented both — see `docs/ai/mistakes.md`'s top entry for full technical
  detail. Both original blockers turned out to be weaker than assessed: the batch path had *no
  real caller at all* (not just "off the hot path"), and the detokenizer's assumed need for
  cross-call state didn't actually exist (byte-level concatenation is inherently stateless).
- While re-verifying under genuinely sanitizer-instrumented library objects (a gap in this
  session's own prior verification — `make test` alone does not actually instrument
  `$(LIB_OBJS)`, only the test files; see mistakes.md), found and fixed 2 more independent,
  pre-existing memory-safety bugs: a heap-buffer-overflow in the AVX-512BW LUT ternary kernel
  (`ternary_matmul_lut_avx512bw.c`, an unnecessary 64-byte SIMD load where only 32 bytes were ever
  used) and a 4x buffer under-allocation in `test_moe.c`'s test harness (`moe_gate_w` sized for
  `tn_i8` when the router reads it as `float`). Both fixed per the "any bug found gets fixed in
  the same pass" policy.
- Verification: `make release`, `make test` (release-mode), a full ASan+UBSan test run with
  `$(LIB_OBJS)` and all test binaries freshly compiled under sanitizer flags in one invocation
  (not the ordering-dependent `make debug` step, which is a no-op here — see mistakes.md), and
  `make debug` (main engine binary) from a clean tree — all green, for both gcc and clang.
- Status: ACCEPTED. Both items closed; no known open shortcomings remain from the Qwen 3.6
  benchmark work as of this entry.

### 2026-07-16 — Closed the Q2_0 perf gap and the remaining chat-template gaps on explicit request
- Decision: given a direct follow-up request to "fix all shortcomings" from the earlier Qwen 3.6
  benchmark (this same day), implemented the deferred Q2_0 VNNI kernel and the contained
  chat-template gaps (`|items` filter, tuple-unpacking for-loops, `loop.previtem`/`nextitem`,
  method calls on literals) rather than leaving them as documented-but-unaddressed. Left two
  items deliberately out of scope, each re-flagged explicitly rather than silently fixed or
  silently dropped:
  1. The Q2_0 *batch* matmul path (`parallel_matmul_q2_0_batch`, used for MoE per-expert
     routing) still uses the portable decode+FMA kernel. Ternary-Bonsai-27B is dense (MoE
     disabled), so this was never on the measured hot path this session, and extending the same
     VNNI trick to it wasn't verified against a real MoE+Q2_0 model.
  2. The byte-level-BPE detokenizer's incomplete GPT-2 byte-unicode reverse table (mojibake on
     non-ASCII output) — a real, pre-existing, unrelated-to-Qwen-3.6 bug found via a screenshot
     review, but a stateful rewrite of a function on every model's hot decode path, with zero
     existing non-ASCII test coverage to verify against; see mistakes.md for the full reasoning.
- Benchmark results (before -> after the VNNI kernel), same prompt, same host, 4 threads,
  greedy decoding: project-zero generation throughput went from ~0.11 tok/s to **~3.24 tok/s**
  (~29x) on the real model. Verified as a genuine speedup and not a correctness regression by
  comparing the exact sampled-token-ID sequence at matching generation steps before and after —
  identical for every step checked (both engines' greedy argmax agreed token-for-token), and a
  full 256-token completion was re-captured end-to-end producing the same coherent
  reasoning-then-answer response as before the optimization.
- Correctness verification method for the new VNNI kernel specifically: a dedicated unit test
  (`tests/test_q2_0_matmul.c`) comparing it against an independent reference decode path
  (`gguf_dequant_q2_0`, unrelated to either matmul kernel) across single-block/multi-block/
  real-model-dimension/non-block-aligned/all-zero-activation cases, run under full ASan+UBSan
  instrumentation of every object involved (not just the test file — see mistakes.md's note on
  why the first version of this test had false failures and how that was root-caused with a
  one-hot activation probe before concluding the kernel itself was correct).
- Status: ACCEPTED. gcc+clang release/test/debug all green (including the 2 new test binaries);
  real-model output re-verified coherent and consistent with the pre-optimization baseline.

### 2026-07-16 — Qwen 3.5/3.6 hybrid-attention support: architecture, benchmark engine, and scope
- Decision: implement full Qwen 3.5/3.6 (Gated DeltaNet + Gated-Attention hybrid) support,
  targeting `prism-ml/Ternary-Bonsai-27B-gguf` specifically, following the existing `has_mla`
  precedent (extend shared `MoEConfig`/`TransformerWeights`/`RunState` structs with
  `q35_*`/`has_linear_attn`-gated fields rather than new per-arch types).
- New tensor format: added `GGML_TYPE_Q2_0`/`gguf_dequant_q2_0`/`parallel_matmul_q2_0` for
  PrismML's ternary Q2_0 packing. Confirmed empirically (bytes-per-tensor computed from the real
  downloaded file, cross-checked against llama.cpp discussion #22019) that this specific file uses
  the **group-128** variant (34B/128 elems, 2.125 bpw), not mainline ggml's canonical **group-64**
  `block_q2_0` (18B/64 elems) — both share GGUF type ID 42, so the group size had to be derived
  from the file, not the type. Zero-copy: token embedding and LM head stay Q2_0-raw (dequantizing
  either fully to F32 would need ~5GB), dequanted per-row/per-token on demand.
- Benchmark engine choice: built **PrismML-Eng/llama.cpp** (the `prism` branch), not mainline
  ggml-org/llama.cpp, specifically because mainline's CPU/Metal Q2_0 kernel only covers the
  group-64 packing and cannot load this project's actual downloaded group-128 file without a
  second multi-GB download of a different GGUF variant. The fork's `QK2_0` constant (128) was
  checked in `ggml-common.h` before building to confirm compatibility with the exact file already
  on disk, letting both engines benchmark the identical `.gguf`.
- Full-model verification method: rather than trusting code review alone against the reference
  source (`src/models/qwen35.cpp`/`delta-net-base.cpp` in the same clone), correctness was proven
  by running the identical prompt through both engines and requiring matching, coherent output —
  this is what actually surfaced the final remaining bug (a transposed conv1d kernel read; see
  `mistakes.md` 2026-07-16 item 6) that line-by-line math comparison against the reference source
  had missed. Tokenization/chat-template rendering were separately confirmed byte-identical via
  `llama-tokenize -f <rendered_prompt.txt> --ids` against project-zero's own `--verbose` dump.
- Scope boundary: Q2_0 matmul performance (currently scalar-decode-to-stack-buffer +
  AVX2 FMA per 128-element block) was explicitly left unoptimized — the task was correctness +
  running the model + benchmarking existing behavior, not a performance pass. This shows up
  directly in the benchmark: project-zero's Q2_0 kernel is markedly slower than the reference
  engine's (see the benchmark artifact for numbers on this specific 4-core host). Flagged as a
  known, deliberately out-of-scope follow-up rather than silently accepted.
- Status: ACCEPTED. Full architecture verified end-to-end against the real 27B model: gcc+clang
  release/test/debug all green (incl. new `tests/test_chat_template.c`, 18 cases), coherent
  reasoning output produced and cross-checked token-for-token against the reference engine's
  continuation of the same prompt.

### 2026-07-16 — Process decision: bugs get fixed on discovery, not just logged
- Decision: any bug found during a task — including pre-existing ones unrelated to what's being
  worked on, surfaced incidentally via ASan/UBSan/ThreadSanitizer, manual/screenshot review, test
  failures, or code reading — gets fixed in the same pass. "Pre-existing" or "unrelated to this
  change" is no longer an acceptable reason to leave a confirmed real defect in place.
- Trigger: a ~1.2MB ASan leak (49k+ allocations) surfaced while verifying the Phase 22.5 CLI
  banner/spinner work was initially treated as a pre-existing, out-of-scope finding and merely
  documented. On follow-up, traced to two real, small, fixable bugs — (1) `tokenizer_free()` in
  `src/cli/main.c` was gated on `args.tokenizer_path`, which is unset for the common GGUF-auto-
  load tokenizer path, so cleanup silently never ran; (2) `GGUFHeader`'s heap-allocated
  string-metadata copies (`parse_meta_entry` in `src/core/gguf_reader.c`) had no corresponding
  free function at all. Both fixed same-session; see `docs/ai/mistakes.md` (2026-07-16) for the
  full writeup.
- Canonical rule added: `docs/ai/engineering-rules.md` § "Bug-fix policy" — only exception is a
  fix requiring a large architectural change, which must be flagged to the user explicitly
  rather than silently fixed or silently left. Synced to all adapters (`.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`).

### 2026-07-15 — Phase 22: Web UI & API/DX hardening — tech + scope decisions
- Decision: project-zero has no UI/UX beyond a raw JSON API and a plain-text CLI (confirmed:
  no `ui/`/`web/`/`static/`/`.html`/`package.json` anywhere pre-Phase-22; `src/api/http_server.c`
  exposed only `GET /v1/models`, `POST /v1/chat/completions`, `GET /health`). Closing the gap vs.
  leading engines (llama.cpp server, Ollama, LM Studio) requires a web chat UI, HTTP API
  hardening (CORS/auth/metrics/OpenAPI), and CLI/REPL polish (color/progress/markdown).
- Web UI tech: **Vite + Svelte** SPA (a lighter analogue of llama.cpp's SvelteKit server UI — no
  SSR needed since the C server only serves static files), built via an npm step used only by
  *contributors* touching `webui/src`. The built `webui/dist/` bundle is embedded into the C
  binary as a generated, **git-committed** byte-array TU (`src/api/webui_bundle_generated.c`),
  so ordinary `make release`/`cmake --build` and CI stay 100% Node-free by default — mirrors
  llama.cpp's own build-then-embed pattern. `make webui-bundle` (opt-in, never in `all`/`test`)
  regenerates it; a `webui.yml` CI job scoped to `webui/**` fails on drift.
- Scope decision (concurrency): the HTTP server's single-threaded/serial `handle_connection()`
  model is **rearchitected** to per-connection threads + a `generation_mutex` (only one
  `generate_with_callback` in flight at a time; second concurrent chat request gets `429`), so
  the web UI's static assets and Stop/Cancel button aren't blocked behind an in-flight
  generation. This is a real concurrency change to `http_server.c`, not pure presentation —
  verified with ThreadSanitizer in addition to the usual ASan/UBSan pass.
- Scope decision (image upload): included now. Requires extracting `main.c`'s inline
  vision-loading block into a reusable `multimodal/vision_pipeline.c` function callable from
  both the CLI and `--server` mode, plus raising `HTTP_MAX_BODY_BYTES`/`CHAT_MAX_CONTENT` to fit
  base64 images.
- Mandatory verification additions (user-specified, apply project-wide going forward for this
  kind of change): (1) any refactor of existing non-UI code done in service of a UI change (the
  two items above) must be verified with a before/after run of every currently-tested model,
  timestamped raw-output screenshots, and a text+tok/s diff against baseline — extends the
  existing golden-output/A/B practice, does not replace it; (2) every UI/UX screenshot (web +
  CLI) is graded against a written design-principles reference (`docs/design/ui-ux-principles.md`)
  via independent image review before being accepted, looping fix→recapture on failure.
- Capture tooling: investigated `docs/tty_bitnet.png`/`demo_bitnet.gif` — confirmed **manually
  captured**, no script/tool/CI step produces them (repo-wide search for vhs/asciinema/ttyrec/
  `.tape` files found zero hits). Phase 22 introduces the first real capture tooling: Playwright
  (headless Chromium, pre-installed in this environment) for the web UI, and a scripted terminal
  capture for the CLI/REPL (see `docs/design/ui-ux-principles.md`/Phase 22.4 notes for the exact
  tool, since `vhs` requires `ttyd`+`ffmpeg` which are not present in this environment).
- Full design: see the Phase 22 plan; sub-phases 22.0 (docs) → 22.1 (API hardening) → 22.2 (web
  UI) → 22.3 (CLI polish) → 22.4 (design QA/regression/README).

### 2026-06-21 — Claim corrected to same-SIMD/same-thread/same-precision; hero = beats bitnet.cpp
- Decision: The README headline claims only what holds apples-to-apples. Fresh three-engine
  measurement on one Xeon (same SIMD, per-thread, matched precision) shows: Project Zero
  beats Microsoft `bitnet.cpp` on BitNet at every thread (+19…+37%, BF16 head; 1.80× / 95%
  DRAM ceiling tuned), and on dense SmolLM2 beats `llama.cpp` at 1–3 threads but trails at the
  4-thread peak (−12%). So **drop any blanket "beats llama.cpp"**; hero = beats bitnet.cpp +
  only single no-dep binary running ternary AND dense. DeepSeek MoE 7× gap stays visible.
- Wording: marketing copy/PNGs must **not** use the word "honest" (reads as justifying);
  state facts confidently instead. (Internal docs may still discuss honesty.)
- Methodology: TG steady-state; PZ via `[gen]` over 128 tok; competitors via `llama-bench
  -n 128 -r 5`; warm cache; one test at a time; full results in BENCHMARK_REPORT.md Addendum AP.
- Gotcha (recorded): the container's host CPU migrated mid-session (Xeon 2.10→2.80 GHz; the
  2.80 lacks `avx_vnni`), which SIGILLs native-built competitor binaries and cripples
  bitnet.cpp's i2_s kernel (0.58 tok/s). PZ is unaffected (runtime SIMD dispatch). Comparison
  numbers are from the 2.10 GHz host where both engines had their VNNI kernels; live tty/video
  demos are PZ-only on the current host and labelled as such.
- Assets added: `docs/benchmark_bitnet.png`, `docs/benchmark_smollm2.png` (bar charts),
  `docs/tty_bitnet.png`, `docs/tty_smollm2.png` (live terminal captures), `docs/demo.webm`.

### 2026-06-20 — README repositioned as a "claim + proof" star-conversion page
- Decision: Lead the README with an honest performance claim ("beats `llama.cpp`/`bitnet.cpp`
  in some configs"), an above-the-fold benchmark table, visual proof, a one-command `make demo`,
  and exposed audit/QA links. Detailed technical content (CLI, architecture, DeepSeek, limits)
  is kept below, nothing removed.
- Honesty rule (binding): every headline win states its config; the losses stay visible —
  DeepSeek-V2 MoE ~7× behind llama.cpp (expert scatter) and SmolLM dense losing at T=1–2.
  Audience is systems devs who will scrutinize; an overclaim that doesn't survive `make demo`
  costs more credibility than it gains.
- Added: `make demo` target (downloads SmolLM2-135M GGUF, runs golden prompt; tokenizer is
  embedded so no `--tokenizer`), `docs/GROWTH_STRATEGY.md` (distribution playbook), and three
  committed proof images under `docs/` (fresh-Xeon terminal card + two OpenBenchmarking result
  screenshots, rendered via headless Chromium / Playwright).
- Reproduction note: the cloud host is itself an Intel Xeon 2.10 GHz / AVX-512 VNNI, so the
  fresh run (BitNet 40.42 tok/s, SmolLM2 142.39 tok/s, INT4) is legitimately comparable to the
  documented Xeon numbers. Project Zero reads BitNet only from native `.bin` (no ternary-GGUF
  support), so the BitNet model was converted from HF safetensors; bf16 tensors were re-encoded
  to f32 first because `convert_hf_bitnet.py` uses numpy (no bf16 slicing) — a one-off
  workaround, the committed tool was not modified.

### 2026-06-19 — Prebuilt x86-64 binary via a tagged GitHub Release
- Decision: Ship a portable prebuilt `adaptive_ai_engine` as a GitHub Release asset, built by a
  new `make dist` target and published by `.github/workflows/release.yml` on `v*` tag push.
  First release tagged `v0.1.0` (pre-1.0 → `prerelease`).
- Portability decision (key): `make release`'s `-march=native` is NOT distributable (the
  AVX-512-VNNI CI runner would bake in AVX-512 → SIGILL on older CPUs). Use **per-TU
  multiversioning**: bulk at `-march=x86-64-v2`, each SIMD kernel TU carries its own ISA flag,
  and `simd_dispatch.c` is compiled at the baseline with `-DTN_FORCE_DISPATCH_ALL` so AVX2/
  AVX-512/VNNI are selected at RUNTIME (the design `simd_dispatch` already implements). Static
  `-static-libstdc++ -static-libgcc` → only libc/libm at runtime.
- Min-CPU envelope: starts + runs BitNet ternary at x86-64-v2; quant/dense GGUF (Q4_K/F16) need
  AVX2 (those kernels are compile-time AVX2-or-scalar, called directly). Documented in
  `docs/RELEASING.md`.
- Supply chain: avoid third-party Actions (cut the release with the `gh` CLI), least-privilege
  `permissions` (only the publish job gets `contents: write`); SHA-pinning noted as a follow-up.
- Status: ACCEPTED; verified gcc release/test(46)/debug/dist + clang release/debug/dist green,
  golden output (France→Paris, Germany→Berlin) correct across scalar/avx2/avx512f/vnni and
  T=1/2/8. (clang `make test` is blocked only by a missing ASan runtime in the local container.)

### 2026-06-07 — Adopt a cross-tool AI development system with one source of truth
- Decision: `docs/ai/**` is canonical; Claude Code, GitHub Copilot, and Google Antigravity
  files are thin adapters that summarize and link here. All such files are dynamic and updated
  proactively every session.
- Rationale: prevent three drifting instruction systems; preserve context across tools.
- Sync rule: on a durable change, update `docs/ai/**` first, then adapters, then record in
  `change-trace.md`. See `tool-sync-policy.md`.

### 2026-06-07 — `master` is the canonical branch; `docs/...` branch archived, not merged
- Decision: Do not merge the unrelated-history `docs/readme-footer-openbenchmarking` branch.
  Verified `master` is a strict superset of its code; archived the branch as a git bundle and
  (pending) remove it from the public remote.
- Rationale: unrelated histories add no code; the branch holds stale tests and messy history.

### 2026-06-07 — Production OOM guard in `run_state_alloc`
- Decision: Add a deterministic size guard (reject buffers >32× available RAM) instead of
  relying on `calloc` returning NULL.
- Rationale: cross-platform determinism (macOS over-commits). 32× headroom never rejects a
  runnable config; all three models re-verified unchanged.
- Status: ACCEPTED, merged via PR #6.

### 2026-06-07 — CI fixes are test/build-only except the one guard above
- Decision: Keep production engine behavior unchanged; fix CI by fixing test harness + Makefile.
- Rationale: "no new development / no regression" constraint; the failures were test-side.

### 2026-06-07 — Regression measured by A/B on the same hardware
- Decision: Because the cloud CPU differs from documented i5-11300H/Xeon baselines, prove "no
  regression" by building HEAD vs a known-good commit on the same host and comparing tok/s +
  golden outputs, rather than comparing to the documented absolute numbers.
- Rationale: absolute tok/s isn't portable across CPUs; relative A/B is.
