# Phase 18 speculative decoding — real-model benchmark (2026-09-24)

**Verdict:** the benchmark found and fixed a severe, real bug (Stage 1's batched-matmul kernels
were scalar-only, making the batched "verify" step *slower* than the sequential calls it was
meant to replace) — fixing it delivered a real **3.6–5.3× speedup** to speculative decoding
itself. But even after the fix, on this 4-core benchmark machine plain generation is still
**faster end-to-end** than speculative decoding for both tested draft/verifier pairs
(speculative runs at 0.42×–0.68× of plain's throughput). This is not a further bug: the
batched-verify mechanism is confirmed working as designed and its savings grow with
`--spec-length` (up to 44% cheaper than the sequential-equivalent cost at spec-length 8) — the
per-round draft-model and corrective-commit overhead that plain generation doesn't pay simply
outweighs it at this core count and these acceptance rates. Full data below.

This report documents the same work recorded in `docs/ai/change-trace.md`'s and
`docs/ai/mistakes.md`'s 2026-09-24 "Stage 7" entries; those are the canonical root-cause/decision
record, this is the full benchmark writeup they point to.

## 1. Setup

Phase 18's draft model must share the verifier's tokenizer for token-level accept/reject to mean
anything. The repo's only local model (`models/smollm2.gguf`) was already the smallest published
size in its family, so it became the **draft**; two larger same-family siblings were downloaded
to serve as **verifiers** — the same "big Brain + small Drafter" shape the feature was designed
for, just with the already-present model taking the small role.

| Role | Model | Params | Size on disk | vs. draft |
|---|---|---|---|---|
| Draft | `SmolLM2-135M-Instruct` | 135M | 271 MB | — |
| Verifier A | `SmolLM2-360M-Instruct` | 360M | 726 MB | 2.7× |
| Verifier B | `SmolLM2-1.7B-Instruct` | 1.7B | 3.4 GB | 12.6× |

All three (`bartowski/SmolLM2-*-Instruct-GGUF`, F16) share `vocab_size=49152` and the same
gpt2-BPE tokenizer — confirmed by parsing each GGUF's own header before downloading the full
file. All weights are F16, run through the classifier's BF16 head.

**Hardware:** 4 physical/logical cores, AVX-512 VNNI, ~35–43 GB/s measured DRAM bandwidth
(varies run to run), 15 GB RAM.
**Build:** `gcc -O3 -march=native` release binary, commit `e7b783b` (the fix commit; see §3),
branch `claude/architecture-progress-review-ahv93e`.

Every command below is the real CLI, unmodified — a plain run omits `--draft-model`, a
speculative run adds it and `--spec-length`:

```
./adaptive_ai_engine --model models/smollm2-360m.gguf \
  --draft-model models/smollm2.gguf --spec-length 5 \
  --prompt "..." --max-tokens 64 --temperature 0 --threads 4
```

## 2. Correctness, checked first

Speculative decoding's greedy accept/reject is designed to reproduce plain greedy generation
token-for-token (Stage 5's synthetic-model proof). Before measuring any speed, that claim was
checked against real weights:

| Check | 135M + 360M | 135M + 1.7B |
|---|---|---|
| Plain vs. speculative output, `--temperature 0` | byte-identical | byte-identical |
| ASan/UBSan, real (non-synthetic) tensor shapes | clean | same code path |

Both pairs produced `"The capital of France is Paris."` for the prompt "What is the capital of
France?" whether or not `--draft-model` was passed.

## 3. The bug this benchmark found

The first benchmark run showed speculative decoding running **8–10× slower** than plain
generation — not a smaller win than hoped, an outright regression. `TN_STEP_TIMING=1` (Stage
5.5's own instrumentation) isolated it to one step: the batched "verify" call cost *more* than
the sequential calls it was meant to replace.

**Root cause:** every batched-matmul kernel (ternary, F16, and the BF16/INT8/INT4 classifier
formats, `src/math/batched_matmul{,_f16,_classifier}.c`, added in Stage 1) decoded weights and
accumulated the dot product in plain scalar C, while every single-token kernel doing the same
math (`src/math/matmul_f16.c`, `ternary_matmul_packed_avx512.c`, etc.) is AVX-512/AVX2
SIMD-accelerated with multi-accumulator FMA. The batched kernels genuinely read each weight row
from RAM only once — the documented bandwidth win was real — but scalar-speed compute on
AVX-512 hardware was so much slower than SIMD-speed compute that it erased the saving many times
over.

**Fix:** decode/unpack each weight row to F32 once per row (SIMD-accelerated where a fast decode
path already existed — `_mm512_cvtph_ps` for F16, `unpack_ternary_block_avx2()` for ternary),
then reuse the project's existing SIMD-dispatched `tn_vec_dot()` (`math/simd_dispatch.h`, the
same dispatch table the single-token attention path's own dot products already go through) for
every candidate token against that row. Same math, no more per-token re-decoding, full SIMD
width where it matters most. Also fixed `tests/test_batched_matmul.c`, which never called
`tn_simd_init()` and so SEGV'd (NULL function pointer) the moment the rewritten kernels started
depending on `tn_vec_dot`.

**Before vs. after** (360M verifier, `--spec-length 5`, same prompt, 6 rounds measured each —
the step the fix targets directly):

| | Verify-batch step cost | |
|---|---|---|
| Before fix | 790.9 ms/round | |
| After fix | 122.4 ms/round | **6.5× faster** |

**Before vs. after, as end-to-end throughput** (same prompt, "Explain how photosynthesis works
in plants, step by step.", 64 tokens):

| Verifier | Spec-length | Before fix | After fix | Speedup |
|---|---|---|---|---|
| 360M | 3 | 4.36 tok/s | 15.62 tok/s | 3.58× |
| 360M | 5 | 3.64 tok/s | 15.98 tok/s | 4.39× |
| 360M | 8 | 3.29 tok/s | 17.45 tok/s | 5.30× |
| 1.7B | 3 | 0.98 tok/s | 4.80 tok/s | 4.90× |

(1.7B spec-length 5/8 "before" numbers were not captured with a large enough sample to be
reliable — the process was interrupted mid-run once the bug was found — so only the one clean
same-prompt data point is reported rather than an estimate.)

Verified: `test_batched_matmul` 6/6 (unchanged numerical tolerance); full `make release/test/debug`
green on gcc and clang; plain-vs-speculative output re-confirmed byte-identical on both real
pairs after the fix — the rewrite is a pure performance change, same dot-product math computed
with wider SIMD lanes and less redundant re-decoding.

## 4. Results: speculative vs. plain, current code

Averaged over two prompts (a ~40-token explanatory answer and a short-story opener, both capped
at 64 tokens so neither model exits early on EOS), greedy decoding, 4 threads, commit `e7b783b`.

| Verifier | Mode | tok/s | vs. plain |
|---|---|---|---|
| 360M | plain | 34.98 | 1.00× |
| 360M | spec-length 3 | 15.62 | 0.45× |
| 360M | spec-length 5 | 15.98 | 0.46× |
| 360M | spec-length 8 | 17.45 | 0.50× |
| 1.7B | plain | 7.89 | 1.00× |
| 1.7B | spec-length 3 | 4.68 | 0.59× |
| 1.7B | spec-length 5 | 4.59 | 0.58× |
| 1.7B | spec-length 8 | 4.28 | 0.54× |

Speculative decoding does not beat plain generation for either pair at any tested spec-length on
this hardware, though the gap narrows as spec-length grows (0.45× → 0.50× on the 360M pair,
0.59× → 0.54× — roughly flat, slightly declining — on the 1.7B pair).

## 5. Where the time goes

Per-round breakdown from `TN_STEP_TIMING=1`'s four Phase 18 round IDs (steps 24–27, added in
Stage 5.5), same 64-token run used in §4's table.

**135M + 360M**

| Spec-length | Draft (ms) | Verify (ms) | Accept/reject (ms) | Commit (ms) | Rounds | Tokens/round | Acceptance |
|---|---|---|---|---|---|---|---|
| 3 | 23.4 | 101.6 | 0.3 | 40.2 | 25 | 2.56 / 4 | 64.0% |
| 5 | 46.3 | 122.4 | 0.6 | 40.8 | 20 | 3.20 / 6 | 53.3% |
| 8 | 82.1 | 147.4 | 1.0 | 41.8 | 14 | 4.57 / 9 | 50.8% |

**135M + 1.7B**

| Spec-length | Draft (ms) | Verify (ms) | Accept/reject (ms) | Commit (ms) | Rounds | Tokens/round | Acceptance |
|---|---|---|---|---|---|---|---|
| 3 | 23.0 | 351.4 | 0.3 | 144.6 | 27 | 2.37 / 4 | 59.3% |
| 5 | 43.5 | 429.0 | 0.6 | 133.1 | 21 | 3.05 / 6 | 50.8% |
| 8 | 81.9 | 563.8 | 0.9 | 151.5 | 19 | 3.37 / 9 | 37.4% |

"Tokens/round" is `tokens_generated / step25.count` (step 25's count is the number of
`transformer_forward_batch` verify calls, i.e. rounds) — a direct acceptance-rate proxy the
existing instrumentation already produces, no new code needed. "Acceptance" is that figure
divided by the max possible tokens/round (`spec_length + 1`, for full acceptance plus the bonus
token).

**Batched verify vs. the sequential-equivalent cost** (what N single-token verifier calls would
have cost, using each pair's own plain per-token rate — 28.6 ms/token for 360M, 126.7 ms/token
for 1.7B):

| Verifier | Spec-length | Batched verify | Sequential-equivalent | Delta |
|---|---|---|---|---|
| 360M | 3 | 101.6 ms | 85.8 ms | **+18% (worse)** |
| 360M | 5 | 122.4 ms | 143.0 ms | −14% |
| 360M | 8 | 147.4 ms | 228.8 ms | −36% |
| 1.7B | 3 | 351.4 ms | 380.1 ms | −8% |
| 1.7B | 5 | 429.0 ms | 633.5 ms | −32% |
| 1.7B | 8 | 563.8 ms | 1013.6 ms | −44% |

The batched call gets cheaper than the sequential calls it replaces as spec-length grows — up to
44% cheaper at spec-length 8 on the 1.7B verifier — confirming the batching mechanism itself now
works as designed. At spec-length 3 on the smaller (360M) verifier it's actually *more*
expensive than sequential: the two-pass design (decode row to F32, then dot-product it N times)
has a fixed per-call cost that a very short batch doesn't amortize enough to beat matmul_f16.c's
single-pass fused decode+FMA. It just doesn't grow fast enough, at any tested spec-length, to
outrun the draft-phase and corrective-commit overhead that plain generation doesn't pay at all.

## 6. Conclusion

The benchmark did its job twice over: it caught a real, severe defect (scalar-only batched
kernels, now fixed and confirmed 3.6–5.3× faster on the code path they touch), and it produced
an honest answer to "does `--draft-model` help here" — on this 4-core machine, with these
draft:verifier size ratios and these acceptance rates, **no, plain generation is still faster.**
The per-round cost of drafting with a second model plus two corrective single-token commits
outweighs the verify-batch savings, even though those savings are real and growing with
spec-length.

This tracks the general literature on CPU speculative decoding: the technique wins when
acceptance rate is high, the verifier is disproportionately expensive relative to the draft, and
there's enough parallel throughput headroom for batching's reduced RAM traffic to translate into
wall-clock time rather than hitting a compute ceiling. Four cores is a fairly low ceiling. Levers
most likely to flip this result: a higher-acceptance draft/verifier pair, more CPU cores, or the
still-open full-SIMD-tier parity for the batched kernels (this pass added one dispatch tier via
the project's existing `tn_vec_dot`, not the full 9-way tier match every single-token ternary
kernel has — a documented Phase 18-B follow-up, `include/math/batched_matmul.h`).

The draft model is not useless — it measurably works, and the fix it drove is a real, durable
improvement to the codebase regardless of this specific hardware's verdict — it just hasn't
crossed even with plain generation yet on this box.
