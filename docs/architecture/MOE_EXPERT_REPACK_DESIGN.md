# MoE Expert Weight Repack — Design (Discussion #1, Q1)

**Status:** design draft (no engine code yet)
**Date:** 2026-10-07
**Companion to:** [`MOE_RESEARCH_AND_FIX_PLAN.md`](MOE_RESEARCH_AND_FIX_PLAN.md) · [Discussion #1](https://github.com/shifulegend/project-zero/discussions/1)

> Answering Discussion #1 §1: *"What's the right granularity for interleaving,
> and how does llama.cpp do it?"* The decisive constraint is not the layout —
> it is that any layout we choose must cost **~zero extra runtime heap**.

---

## the constraint that shapes every option

64 experts/layer × 26 MoE layers, 6 active per token. The expert tensors are
stored sequentially in the 8.9 GB GGUF, so the 6 active experts for a token are
scattered across the file → **156 scattered DRAM streams/token**; the hardware
prefetcher tracks ~8–10 → effective bandwidth collapses 11.7 → 2–3 GB/s.

`MOE_RESEARCH_AND_FIX_PLAN.md` already established the trap: **a load-time copy
of the expert tensors needs 6+ GB of extra heap**, which OOMs an 8 GB machine.
So the design goal is a layout with contiguous top-k access *without* a second
full copy resident in RAM.

## design space (ranked)

### 1. Offline repack artifact — recommended
A one-time pass (`tools/repack_experts.py`, run in the conversion pipeline) reads
the GGUF and writes a side-car `*.pzrepack` file in which, for each MoE layer, the
expert weight blocks are reordered so a token's top-k are **contiguous**. The
engine `mmap`s the side-car read-only and points the 64 expert pointers at it.
The original GGUF stays untouched, so `--model orig.gguf` keeps working.

- **granularity: whole Q4_K super-blocks, within row-major expert blocks.**
  A super-block (32 elements) carries its own fp16 `d`/`dmin`; interleave at
  *whole-super-block* granularity so decode never splits a block. Not per-element
  (breaks the block-scale contract); not whole-expert (that is already the current
  layout, and it is the problem).
- cost: one-time disk, **zero extra runtime heap** (mmap), reversible.

### 2. In-place permutation index — supplement only
Keep the GGUF; build a 64-entry per-layer order table used to gather. Alone this
does **not** fix the prefetcher (the gather is still non-contiguous); it helps only
TLB/locality when combined with (1).

### 3. `madvise` / prefetch hints — supplement only
`madvise(MADV_WILLNEED)` on the selected experts before compute was tried (P4) and
did not move the needle, because page faults are not the dominant term once the
streams are scattered. `posix_fadvise` is a no-op on an mmap. Keep as an optional
hint on top of (1), never a replacement.

### 4. Huge pages — orthogonal, cheap
`madvise(MADV_HUGEPAGE)` over the expert region (or `MAP_HUGETLB`) cuts TLB misses.
Independent of layout; safe to enable on its own.

## how llama.cpp does it (verify against the current tree)

llama.cpp holds every tensor as a `ggml_tensor` in one model buffer and dispatches
at compute time on `tensor->type`; for CPU MoE it depends on per-tensor layout plus
cache/prefetch handling, and on some backends a CPU **repack** step produces a
blocked copy. The transferable principle (GOLDEN_RULES Rule 7) is the important
part: **never dequantize upfront — pass the raw quantized pointer and dispatch the
kernel at compute time.** PZ's remaining MoE bottleneck is exactly an F32-dequant
path that violates this; the repack (this doc) and the native Q4_K kernel (Q2 of the
discussion) are the two halves.

> ⚠️ `MOE_RESEARCH_AND_FIX_PLAN.md` flagged that a cited `repack.cpp` could not be
> located in the then-current llama.cpp tree. Do **not** cite a specific file path
> here until it is re-verified against the pinned checkout (GOLDEN_RULES Rule 7).

## performance update — the bandwidth model, and what the repack recovers

The whole problem is one inequality: a token's top-k access must present **fewer
concurrent streams than the prefetcher can track** (~8–10).

```
per-token scattered reads  = 6 experts × 26 MoE layers      = 156
concurrently live streams  = 6 experts × 4 threads          = 24   (prefetcher: ~8–10)
→  prefetcher overwhelmed  →  BW 11.7 GB/s (practical)  →  2–3 GB/s (scattered regime)
```

The repack collapses the per-layer access into **one sequential run**, so the
prefetcher stays ahead and effective bandwidth climbs back toward the ceiling.

| regime | effective BW | DeepSeek-V2-Lite tok/s (T=1..4) |
|---|---|---|
| now — scattered, F32 dequant path | ~2–3 GB/s | 1.10 / 1.26 / 1.32 (measured) |
| repack + native Q4_K kernel | → toward practical ceiling (~11.7 GB/s) | → toward the **~9.8 analytical ceiling** |
| llama.cpp, same class of host (reference) | — | 7.73 / 13.44 / 19.72 |

Why the ceiling is ~9.8 and not higher on this model: the ~8.9 GB of expert weights
must stream once per token **regardless of layout** — the repack removes the *waste*
(miss-serviced lines), not the floor. The remaining gap to llama.cpp at T=4 is the
second half of the work — the **native Q4_K matmul** (Discussion #1, Q2) — not the
layout.

**targets (acceptance):** L3 miss **< 60%** (from 85–86 %); effective BW **≥ 6 GB/s**;
**≥ 9 tok/s** on the DeepSeek-V2-Lite Q4_K_S row; all **27 layers active** (no dead
layers from NaN). Measured as an A/B on the same host (see below).

## the golden-hash regression (borrowed discipline)

Pin a fixed prompt → first-N logits (or the emitted token stream) to a hash
asserted in CI, so "garbled vs coherent" becomes a failing test rather than a human
eye. This is the same cross-machine golden the sibling ternary lane
(`8b-is-engine`, `crates/ternary-lane`) proves bit-identical on aarch64 / x86-64 /
wasm.

## verification plan (per GOLDEN_RULES Rule 5 + decision-log 2026-06-07)

- A/B on the **same host**: HEAD (no repack) vs the repacked path — compare tok/s
  and the golden token-stream hash. Absolute tok/s is not portable across CPUs;
  relative A/B is.
- Record the exact command with **all** flags, the exact output, and tok/s
  (GOLDEN_RULES Rule 8).
- Acceptance for the repack step alone: the **performance-update targets** above
  (L3 miss < 60 %, effective BW ≥ 6 GB/s, ≥ 9 tok/s, all 27 layers active).

## reproduce the baseline (Rule 8 — record every flag, including defaults)

The measurement that motivates this doc, in the exact command form the repo
requires (all flags explicit; `--simd`/`--classifier` default to `auto`):

```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 30 --temperature 0.0 --threads 4 \
  --simd auto --classifier auto
# record: exact output text · tok/s · and (perf stat) IPC + L3-miss
```

Record the *output text + tok/s + IPC + L3-miss* together — a tok/s number
without its flags is the exact failure GOLDEN_RULES Rule 8 exists to prevent.
The same run, with the repack in place, is the A/B baseline.

## next artifact

The repack itself is `tools/repack_experts.py` (offline, writes the `*.pzrepack`
side-car described above) plus the load-time pointer swap in `gguf_loader.c` —
neither is in this docs PR; this doc is the design that gates them.

*the CPU cousin of the ternary lane · 0 + 1 · fine touch from within · vaked.dev*
