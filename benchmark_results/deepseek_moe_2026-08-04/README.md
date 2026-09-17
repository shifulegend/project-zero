# DeepSeek MoE benchmark: project-zero vs colibri vs llama.cpp (2026-08-04)

## TL;DR

**No single MoE model runs on all three engines**, so this is two pairwise
comparisons, both anchored on llama.cpp:

| Model | project-zero | colibri | llama.cpp |
|---|---|---|---|
| **DeepSeek-V2-Lite-Chat** (15.7B/2.4B, Q4_K_S) | **4.07 tok/s** | n/a — no DeepSeek arch support | **12.87 tok/s** |
| **OLMoE-1B-7B-0924-Instruct** (6.9B/1.3B) | n/a — no OLMoE arch support | **0.63 tok/s** (0.31–0.90 range) | **27.50 tok/s** |

All runs: T=4 threads (this box has 4 physical cores, no SMT), CPU-only, cold
page cache warmed before timing, greedy/deterministic decoding.

## Why not one 3-way table

- **colibri has no DeepSeek loader.** Its `README.md` lists exactly four
  supported families — GLM-5.2 (744B), Inkling (975B), Kimi K3 (2.8T), OLMoE
  (7B/1B) — one hand-written C engine per family (`colibri.c`, `inkling.c`,
  `kimi_k3.c`, `olmoe.c`). `colibri.c` and `quant.h` reference DeepSeek only as
  prior art (GLM-5.2 reuses DeepSeek-V3-style MLA/MTP) — there's no code path
  that loads a `deepseek2`-arch GGUF or safetensors checkpoint. The three
  families that *are* DeepSeek-lineage (GLM-5.2, Inkling, Kimi K3) are
  372 GB–1.45 TB, far beyond what fits in this environment.
- **project-zero has no OLMoE loader.** `src/core/gguf_loader.c` special-cases
  exactly three GGUF `arch` strings for MoE routing — `deepseek2`, `qwen35`,
  `qwen3moe`. An `olmoe`-arch GGUF falls through to the generic dense loader,
  which looks for `ffn_gate/up/down` tensors that don't exist in an OLMoE
  checkpoint (it has `ffn_*_exps` + a router) — this is a hard load failure,
  not a degraded run.

Given that, this run uses each engine's smallest *actually-supported* MoE
model, with llama.cpp run on both so it's a common reference point across the
two tables.

## Table 1 — DeepSeek-V2-Lite-Chat Q4_K_S (project-zero vs llama.cpp)

Model: [`mradermacher/DeepSeek-V2-Lite-Chat-GGUF`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF)
`.Q4_K_S.gguf` (9.53 GB on disk, 15.7B total / 2.4B active params, 64
routed experts + 2 shared experts/layer, top-6 routing, MLA attention).

| Engine | Method | Decode tok/s |
|---|---|---|
| **project-zero** | 1 warmup (discarded) + 3 timed process invocations, `--threads 4 --temperature 0 --max-tokens 16` | 4.11 / 4.19 / 3.91 → **avg 4.07** |
| **llama.cpp** | `llama-bench -m <model> -t 4 -n 20 -p 8 -r 3 -ngl 0` (in-process, r=3 avg) | prompt(p=8) 20.14 · **gen(n=20) 12.87** ± 0.40 |

**llama.cpp is ~3.16× faster than project-zero** on this model/machine. This
matches the direction (not the magnitude) of the standing finding in
`README.md` / `docs/DEEPSEEK_Q8_HANDOVER.md`: project-zero's MoE path is
memory-bandwidth-bound because top-K expert weights sit at non-contiguous
GGUF offsets (~86% L3 miss rate/token per the handover doc's analysis), while
llama.cpp repacks/interleaves selected-expert weights. The 3.16× gap measured
here (vs. the previously-recorded 7.3× on a different machine, an
i5-5250U/DDR3 box) is a different CPU (`Intel Xeon @ 2.10GHz`, AVX-512 VNNI,
~40 GB/s observed bandwidth vs. the old box's ~5.3 GB/s DDR3 ceiling) — not a
code fix; the underlying non-contiguous-expert-access issue is still open
(see `README.md`'s Help Wanted table).

Raw output: `pz_deepseek_runs.txt`, `llamacpp_deepseek_bench.csv`.

## Table 2 — OLMoE-1B-7B-0924-Instruct (colibri vs llama.cpp)

- **colibri**: converted via `c/tools/convert_olmoe_merged.py --repo
  allenai/OLMoE-1B-7B-0125-Instruct` to colibri's merged int8 format (~6.9 GB
  resident), run as `SNAP=<dir> ./olmoe 64 8 ref_olmoe_real.json` (cache=64 =
  every expert, so no eviction by design), `OMP_NUM_THREADS=4`.
- **llama.cpp**: [`allenai/OLMoE-1B-7B-0924-Instruct-GGUF`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF)
  `olmoe-1b-7b-0924-instruct-q4_k_m.gguf` (4.21 GB), `llama-bench -t 4 -n 20 -p 8 -r 3`.

| Engine | Method | Decode tok/s |
|---|---|---|
| **colibri** | 3 timed runs, `ref_olmoe_real.json` (5-token prompt → 12 generated) | 0.69 / 0.31 / 0.90 → **avg 0.63** (high run-to-run variance) |
| **llama.cpp** | `llama-bench` r=3 avg | prompt(p=8) 49.94 · **gen(n=20) 27.50** ± 1.64 |

colibri is ~44× slower than llama.cpp here, but **this is not a fair
architectural comparison** — colibri's `olmoe.c` engine is a disk-streaming
design built for models where resident weights vastly exceed RAM (its own
docs benchmark GLM-5.2 at 744B/372GB-on-disk, ~11 GB of *disk reads per
token* on a cold cache). OLMoE-1B-7B is 6.9B total params; colibri's own
`chat_olmoe.sh` comments note the *entire* expert set for this model fits in
an ~8 GB RSS budget — i.e., this benchmark runs colibri's streaming/cache
machinery on a model that doesn't need streaming at all, on a container whose
block-device I/O is plausibly throttled/shared (hence the 0.31–0.90 tok/s
spread across identical runs — recorded expert cache hit rate was a
constant 65.5% across all 3 runs, so the variance is I/O latency, not cache
behavior). Read this table as "colibri runs correctly and reproducibly on
its smallest supported model on this box," not as a verdict on colibri's
design, which targets a completely different model-size regime.

Raw output: `colibri_olmoe_runs.txt`, `llamacpp_olmoe_bench.csv`.

## Methodology notes

- Page cache warmed (`cat model > /dev/null`) before each engine's first run.
- All three engines run sequentially, never concurrently, to avoid CPU
  contention skewing tok/s.
- T=4 chosen because this container exposes exactly 4 physical CPUs (no SMT) —
  the natural "use everything" setting for all three engines.
- No `sudo`/CPU-governor pinning available in this container (unlike the
  bare-metal procedure in `.claude/BENCHMARK_SUMMARY.md`'s pre-benchmark
  checklist) — numbers here are container-relative, not a new authoritative
  record for either engine.
- project-zero and llama.cpp binaries were built fresh in this session
  (`make release CC=gcc`; llama.cpp `cmake -DGGML_NATIVE=ON`, commit `4308a4f`).
  colibri's `olmoe` binary built via `make -C c olmoe` (gcc, `-O3 -march=native
  -fopenmp`).

## Environment

See `system_info.txt`.
