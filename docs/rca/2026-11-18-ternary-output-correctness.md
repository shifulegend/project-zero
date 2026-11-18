# RCA: wrong / degraded BitNet output (bitnet.cpp and pz), 2026-11-18

Status: **fixed** — pz activation quantiser (#75), LM-head kernels (#76), correctness gate in the benchmark scripts (#74, #77). The bitnet.cpp finding is reported upstream (microsoft/BitNet #602). Still open: § 8 item 4 (SmolLM2-1.7B Q4_K_M, pz perplexity 0.46% below llama.cpp, not traced to a tensor).
Evidence: `benchmark_results/rca_ternary_output_20261118/`, scripts in `tools/bench/verify/`.

## 1. Trigger
The head-to-head screenshots of bitnet.cpp on `bitnet-b1.58-2B-4T-i2_s.gguf` show the same two
sentences repeated for 500 tokens (also in the published 2026-10-29 sweep). A benchmark is only
meaningful if every engine computes the model correctly, so the cause had to be found, not explained away.

## 2. Method (nothing assumed)
1. Same prompt, same token IDs, greedy, in every engine (token IDs compared, § 3).
2. Bisect bitnet.cpp by build configuration (SIMD tier, repack, llamafile, batching).
3. Compare the first layer tensor by tensor against an **independent numpy computation written
   from the GGUF bytes only** (`tools/bench/verify/bitnet_layer0.py`; no pz or llama.cpp code used;
   weight layout taken from bitnet.cpp's quantiser `ggml-bitnet-mad.cpp:97-150`).
4. Confirm each hypothesis by changing exactly one thing and measuring.
5. Score every engine with perplexity on one fixed text (the standard llama-perplexity method;
   pz has `--perplexity` with the same method).

## 3. Things ruled out (measured)
| Hypothesis | Result |
|---|---|
| Tokenizer | pz and bitnet.cpp (default pre-tokenizer) give the **same 33 token IDs**. With the llama3 pre-tokenizer forced (`--override-kv tokenizer.ggml.pre=str:llama3`) 1 split changes (`real`,`-`,`world` becomes `real`,`-world`): 32 tokens, same wrong text. Not the cause. |
| Sampler / raw prompt | Plain prompt "The capital of France is": bitnet.cpp " a small city, and the capital of France is a small city…"; pz, ik_llama.cpp " Paris. Paris is…". Not a sampling artefact. |
| SIMD tier | bitnet.cpp rebuilt with AVX2 only (no AVX-VNNI/AVX-512): identical wrong text. |
| Prompt batching, CPU repack | `-b 1 -ub 1` and `GGML_CPU_REPACK=OFF`: identical wrong text. |
| Threads | `-t 1` identical. |

## 4. Finding 1 (bitnet.cpp): FFN activation is SiLU, the model needs squared-ReLU
**Code**: `3rdparty/llama.cpp/src/models/bitnet.cpp:128-133`
```cpp
cur = build_ffn(cur, ffn_up, ..., ffn_gate, ..., LLM_FFN_SILU, LLM_FFN_PAR, il);   // line 133
```
`llama_model_bitnet_b158` (`models/models.h:1318`, `llama-model.cpp:199-200`) derives from
`llama_model_bitnet` and overrides only `load_arch_hparams` (`models/bitnet.cpp:173`), so the
`bitnet-b1.58` architecture (BitNet 2B-4T: `hidden_act = relu2`) is built by the SiLU graph.
ik_llama.cpp's graph uses `LLM_FFN_RELU_SQR` (`src/graphs/build_bitnet.cpp:234`); pz sets
`act_type = 1 /* ReLU2 */` for `bitnet*` (`src/core/gguf_loader.c:270-273`).

**Evidence (layer 0, BOS token, from the same GGUF):**
| tensor (first 3 values) | numpy ReLU² | numpy SiLU | bitnet.cpp | pz default |
|---|---|---|---|---|
| Vcur-0 | -0.1914 -0.0891 0.0672 | same | -0.1914 -0.0891 0.0672 | -0.1972 -0.1329 0.1373 (Finding 2) |
| ffn_down-0 | **0.46 24.03 -36.88** | -33.49 86.64 -81.20 | **-35.54 88.23 -80.71** | |
| l_out-0 | 0.49 24.37 -37.18 | -33.45 86.99 -81.50 | -35.51 88.58 -81.00 | 3.57 21.98 -36.16 |
Everything up to the FFN activation matches the numpy reference (Vcur-0 to 4 digits); bitnet.cpp's
`ffn_down-0` equals the **SiLU** variant (also visible in `eval-callback`: op `SWIGLU`, and `-0.0`
where squared-ReLU would give `+0.0`).

**Confirmed by experiment**: changing that one line to `LLM_FFN_RELU_SQR`
(`benchmark_results/rca_ternary_output_20261118/bitnet_cpp_ffn_relu_sqr.diff`) gives
"The capital of France is Paris. Paris is a city that is known for its rich history, culture…"
and perplexity 31.40 (was 48.67).

Upstream: already reported as microsoft/BitNet #602 and #588 (see `2026-11-18-fix-and-gate-plan.md` § 3); `main` is still the commit tested here. The model's `config.json` says `hidden_act: relu2`.

Secondary bug seen on the way: building bitnet.cpp with `GGML_LLAMAFILE=OFF` fails to compile
(`ggml-cpu.c:1495: use of undeclared identifier 'src1_cont'`), so the non-llamafile I2_S path is dead code.

## 5. Finding 2 (pz): the activation quantiser truncates instead of rounding
**Code** (every x86/scalar tier, all feed the ternary int8 kernels):
- `src/math/quantize_i8.c:29` `q[i] = (int8_t)(int)v;  /* truncation toward zero (same as llama.cpp) */`
- `src/math/quantize_i8.c:96` `_mm256_cvttps_epi32` (AVX2) and `:111` scalar tail
- `src/math/quantize_i8_avx512.c:60` `_mm512_cvttps_epi32` (the default VNNI tier) and `:72`
Callers: `ternary_matmul_packed_vnni.c:89`, `ternary_matmul_packed_vnni256.c:67`,
`ternary_matmul_packed_dotprod.c:71` (ARM), `matmul_q2_0_vnni.c:234,341`, `parallel_matmul.c:149,179` (preq path).

The comment is wrong: bitnet.cpp's `quantize_row_i8_s` (`quants.c:1322`) uses `roundf`, and so does
the BitNet reference. Truncation shrinks every activation towards zero by 0.5 LSB on average, a
bias correlated with the sign of the activation, so it does not average out across 2560 terms.

**Evidence (layer 0, BOS, V projection, 640 outputs, relative L2 error vs exact float):**
| variant | error |
|---|---|
| numpy int8, round-to-nearest (= bitnet.cpp's method) | 0.38 % |
| pz `--simd avx2` (float activations) | 0.03 % |
| numpy int8, **truncate toward zero** | **3.71 %** |
| pz default (`--simd vnni`) | **3.70 %** (0.17 % from the truncating numpy model) |
pz's default int8 path is ~10x noisier than an 8-bit quantiser should be, and the truncating model
reproduces it, so truncation is the cause.

**Impact (perplexity, BitNet 2B, same text, 6 chunks of 512):**
| engine | PPL |
|---|---|
| bitnet.cpp, ReLU² fix, int8 round-to-nearest | 31.40 |
| pz `--simd avx2` (float activations) | 31.32 |
| **pz default (int8, truncating)** | **34.39 (+9.7 %)** |
| bitnet.cpp as built (SiLU) | 48.67 (+55 %) |
Two independent correct implementations agree within 0.3 %; pz's default path is measurably worse.
pz's output is fluent, which is why this survived: the damage is a ~10 % perplexity loss, not gibberish.

**Why no test caught it**: `tests/test_simd_vnni.c:125` asserts "max quantization error <= scale
(truncation toward zero loses at most 1 ULP)", i.e. it encodes truncation as expected (a correct
quantiser is within 0.5 scale). All tiers truncate identically, so tier-vs-tier tests agree.

## 6. Everything else measured
| model | llama.cpp PPL | pz PPL | verdict |
|---|---|---|---|
| SmolLM2-135M F16 | 8.0257 | 8.0252 | match (0.006 %) |
| Mamba-130M F16 | 12.5667 | 12.5801 | match (0.11 %) |
| SmolLM2-1.7B Q4_K_M | 5.9495 | 5.9221 | pz 0.46 % lower; not understood (see § 7) |
| BitNet 2B | (bitnet.cpp, see § 5) | 34.39 | see § 5 |
Greedy text: SmolLM2-135M identical for 32 tokens; Mamba and 1.7B agree for the first sentence then
take a different branch at a near-tie (PPL agrees for Mamba).
Smaller observations: pz holds the embedding table as BF16 (`TMP_x_in` for BOS: 0.15625 vs the F16
file's 0.156616, ~0.2 %); pz also loads this GGUF with `pre=default` (same as bitnet.cpp; llama3 splits
`-world` as one piece).

## 7. Web interface
pz's chat API on this model answers correctly ("What is the capital of France?" -> "The capital of
France is Paris.", and a coherent two-sentence ML explanation); the web UI is a front end for the same
API and kernels. It is therefore **not broken, but it carries Finding 2**: the same truncating
quantiser runs in the server, so BitNet quality there is ~10 % worse in perplexity than it should be.

## 8. Recommended root-cause fixes (not applied)
1. pz: one shared rounding definition for activation quantisation (round-to-nearest-even in AVX-512,
   AVX2, NEON and scalar; `cvtps_epi32` / `vcvtnq_s32_f32` / `nearbyintf`), correct the comment, change
   the test to demand <= 0.5 scale error, add a layer-level regression against `bitnet_layer0.py`, update
   golden outputs that move, A/B the speed (no extra cost expected: same instruction count).
2. Benchmark harness: a **correctness gate** before timing, since speed of a wrong computation is
   meaningless: per model, perplexity of every engine on a fixed text must agree with the reference
   engine within a stated tolerance, otherwise the cell is reported INVALID instead of timed.
3. bitnet.cpp reference: either (a) report its BitNet cells as INVALID (wrong function at the pinned
   commit) and keep the patched build as a separate, clearly labelled "bitnet.cpp + ReLU² fix"
   variant, or (b) only the patched variant. Needs a decision, since it changes what the published
   numbers compare. The upstream bug is worth reporting to microsoft/BitNet.
4. Understand the 1.7B Q4_K_M 0.46 % gap by per-tensor comparison before calling it benign.

## 9. Reproduce
```
python3 tools/bench/verify/bitnet_layer0.py A                 # numpy layer 0 (needs models/bitnet-b1.58-2B-4T-i2_s.gguf)
./pz --model <gguf> --perplexity <text> --ppl-ctx 512 --ppl-chunks 8 [--simd avx2|vnni --no-calibrate]
llama-perplexity -m <gguf> -f <text> -c 512 --chunks 8        # llama.cpp / bitnet.cpp builds
llama-eval-callback -m <gguf> -p "The capital of France is" -n 1   # bitnet.cpp per-op tensors
```
