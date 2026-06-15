# DeepSeek-V2-Lite-Chat Performance Handover
## Q4K × Q8 Fused Kernel — Complete Developer Handover Document

> **Status:** Plan approved, implementation NOT started  
> **Prepared:** 2026-03-24 (Copilot session 6b85c14c)  
> **Target:** Close the 7.3× gap between project-zero (1.90 tok/s) and llama.cpp (13.79 tok/s)  
> **Read before touching any code:** `GOLDEN_RULES.md`

---

## 0. Who Should Read This

This document is written for **any developer or AI agent continuing this work** — including
someone who has never seen this repository before. It explains the project from scratch,
documents every failed approach, identifies exactly what needs to change, and provides
step-by-step implementation instructions.

Do not skip sections. The graveyard in Section 5 contains approaches that *looked* correct
but caused regressions — re-discovering them wastes days.

---

## 1. What Is project-zero?

**project-zero** is a from-scratch, high-performance LLM inference engine written in pure C.
It runs large language models on commodity CPUs with no GPU, no Python, no external framework.
A single compiled binary (`adaptive_ai_engine`) can run BitNet ternary models, F16 dense models
(e.g. SmolLM2-135M), and quantized GGUF models (e.g. DeepSeek-V2-Lite-Chat Q4_K_S).

**Why C, not Python?** Python ML frameworks have unavoidable overhead — Python interpreter,
framework dispatch, CUDA driver. For CPU-only inference at maximum throughput, everything
must be in tight C loops with direct SIMD intrinsics. The engine achieves this.

**How to build:**
```bash
cd /Users/<AUTHOR>/Documents/project-zero
make -j4           # Release build: -O3 -march=native
make debug -j4     # Debug build: -g -O0 -fsanitize=address
make clean         # Remove all build artifacts
```

**How to run (DeepSeek example):**
```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 12 --temperature 0.0 --threads 4
```

**Key CLI flags for this work:**
- `--threads N` — number of compute threads (T=4 is optimal on this 4-logical-core machine)
- `--classifier auto/bf16/int8/int4` — LM head quantization (use `int8` for DeepSeek)
- `--simd auto/avx2/scalar` — SIMD selection (does NOT affect Q4K path — see Section 5.1)
- `--verbose` — print per-step timing and expert hit statistics

---

## 2. What Is GGUF? What Is Q4_K?

### 2.1 GGUF Format

GGUF is the binary file format used by llama.cpp (and now project-zero) to store model weights.
A `.gguf` file contains:
1. A header with metadata key-value pairs (model architecture, vocabulary, hyperparameters)
2. A tensor descriptor table (name, shape, quantization type, byte offset for each weight matrix)
3. Raw weight data (quantized bytes in packed format)

project-zero reads GGUF with `src/core/gguf_reader.c` (parses the binary header) and
`src/core/gguf_loader.c` (maps tensor descriptors to the `TransformerWeights` struct and
`Config` struct at load time).

### 2.2 Q4_K Quantization

Q4_K is a 4-bit quantization format. Each weight value is stored as a 4-bit integer (0-15)
instead of a 32-bit float. This reduces model size by ~8×. A 16 GB model becomes ~2 GB.

**Why not just 4-bit? Why "K"?** The "K" variants (Q4_K, Q5_K) use a two-level scale
structure called a "superblock" for better accuracy. Each superblock covers 256 elements
and uses 8 sub-block scales.

**Q4_K superblock layout (256 elements → 144 bytes):**
```
Bytes  0- 1:  d     (float16) — superblock scale
Bytes  2- 3:  dmin  (float16) — superblock minimum correction scale
Bytes  4-15:  sc12  (12 bytes) — 8 sub-block scales + 8 sub-block mins, packed 6 bits each
Bytes 16-143: qs    (128 bytes) — 256 × 4-bit nibbles, two nibbles per byte
               └── qs[i] lo nibble = element 2i, hi nibble = element 2i+1
```

**How a weight value is decoded:**
```
For element i in sub-block g (g = 0..7, each sub-block = 32 elements):
  raw_nibble = qs[i/2] & 0xF  (for even i)  OR  qs[i/2] >> 4  (for odd i)
  true_value = raw_nibble * sc[g] * d - mn[g] * dmin
```

Where `sc[g]` and `mn[g]` are decoded from the packed `sc12` field.

This is exactly what `dot_q4k_row()` in `src/math/matmul_q4k.c` does today — but with
F32 activations. The fix replaces F32 activations with int8 (Q8) activations.

---

## 3. What Is DeepSeek-V2-Lite? (Architecture for Newcomers)

DeepSeek-V2-Lite-Chat is a 15.7B parameter language model from DeepSeek AI. It uses two
advanced techniques: **MLA** (Multi-head Latent Attention) and **MoE** (Mixture of Experts).

### 3.1 Standard Transformer vs DeepSeek

A standard transformer (e.g. SmolLM2-135M which project-zero already runs well) has:
- **Attention:** N attention heads, each reading from KV cache — reads/writes `2 × dim × n_layers` per step
- **FFN:** Two linear projections (gate + up) → activation → multiply → down projection

DeepSeek uses:
- **MLA Attention:** Low-rank KV compression — instead of caching full K and V matrices
  (which would be huge at 16 heads × 5120 dim), it compresses to a 512-dimensional "latent"
  and re-expands at query time. This massively reduces KV cache size and bandwidth.
- **MoE FFN:** Instead of one FFN per layer, there are 64 "expert" FFNs. For each token,
  only 6 are selected (top-6 routing). This gives the model capacity of a huge FFN
  but computation cost of a small one.

### 3.2 DeepSeek-V2-Lite Key Dimensions

Sourced from `include/core/moe_config.h` and `BENCHMARK_REPORT.md` addenda:

```
dim           = 2048    (token embedding dimension)
n_layers      = 27      (total transformer layers)
n_heads       = 16      (attention heads)
n_kv_heads    = 16      (KV heads — GQA ratio 1:1 for Lite)
kv_lora_rank  = 512     (MLA KV compression dimension — reduces KV cache 10×)
qk_nope_dim   = 128     (per-head non-position-encoded Q/K dimension)
qk_rope_dim   = 64      (per-head RoPE Q/K dimension)
v_head_dim    = 128     (per-head V dimension)
vocab_size    = 102,400 (large vocabulary)

num_experts   = 64      (total routed experts per MoE layer)
top_k         = 6       (experts selected per token)
n_shared_exp  = 2       (shared "always-on" experts)
expert_hdim   = 1408    (expert FFN hidden dimension)
first_moe_layer = 1     (layer 0 is dense; layers 1-26 are MoE)
```

### 3.3 Why DeepSeek Is Harder to Optimize Than SmolLM2

SmolLM2-135M has dim=576, 30 layers, all F16 weights → total ~204 MB per decode step.
DeepSeek-V2-Lite Q4_K_S has ~8.9 GB of weights. At 1.90 tok/s, it reads ALL weights
every ~0.53 seconds. The machine's DDR3 bandwidth is only 5.1–5.3 GB/s. The Q4K kernel
must minimize bytes read per multiply — that's the entire optimization story.

---

## 4. How a Decode Step Works in project-zero (DeepSeek)

Understanding the call chain is essential before changing anything.

```
main() [src/cli/main.c]
  └─ generate() [src/transformer/generate.c]
       └─ transformer_forward(token, pos) [src/transformer/forward.c]
            ├─ embed_token() — lookup token embedding
            └─ for each layer 0..26:
                 ├─ attention_forward()  [src/transformer/attention.c]
                 │    └─ (has_mla=true) → mla_attention_forward()
                 │         ├─ MLA_MATMUL(kv_compress)   ← calls parallel_matmul_q4k()
                 │         ├─ MLA_MATMUL(kv_expand)     ← calls parallel_matmul_q4k()
                 │         ├─ MLA_MATMUL(q_proj)        ← calls parallel_matmul_q4k()
                 │         ├─ RoPE, softmax, attention
                 │         └─ MLA_MATMUL(output_proj)   ← calls parallel_matmul_q4k()
                 └─ ffn_forward() [src/transformer/ffn.c]
                      └─ (moe_layer_is_moe) → moe_ffn_forward()
                           ├─ moe_router_forward() — gate scores
                           ├─ top-k expert selection
                           ├─ for each selected expert e:
                           │    ├─ dequant_expert_weight(w1[e]) → F32 → matmul  ← SLOW PATH
                           │    ├─ dequant_expert_weight(w3[e]) → F32 → matmul  ← SLOW PATH
                           │    ├─ SiLU activation
                           │    └─ parallel_matmul_q4k(w2[e])  ← already on fast path
                           └─ shared expert (2 always-on):
                                ├─ parallel_matmul_q4k(shared_w1)  ← fast path
                                ├─ parallel_matmul_q4k(shared_w3)  ← fast path
                                └─ parallel_matmul_q4k(shared_w2)  ← fast path
```

**Current hot paths** (most time spent):
1. `mla_attention_forward`: 5 calls to `parallel_matmul_q4k()` — dominates at ~60% of step
2. `moe_ffn_forward` expert w1/w3: still calling `dequant_expert_weight()` → F32 → float matmul
3. `moe_ffn_forward` expert w2 + shared: already using `parallel_matmul_q4k()`

**All `parallel_matmul_q4k()` calls are bottlenecked by the same issue:** the inner loop
`dot_q4k_row()` reads F32 activations (4 bytes/element).

---

## 5. Root Cause — Why We Are 7.3× Slower Than llama.cpp

### 5.1 Memory Bandwidth Is the Bottleneck (Proven)

Hardware: Intel i5-5250U, 8 GB DDR3, measured bandwidth ~5.1–5.3 GB/s.

For DeepSeek-V2-Lite at 1.90 tok/s, each decode step reads essentially the entire model:
- Model size (Q4_K_S): 8.9 GB → at 1.90 tok/s: 8.9 GB × 1.90 = 16.9 GB/s **needed**
- But DDR3 ceiling is 5.3 GB/s — so how is it running at all?

The answer: only a fraction of the model is active per token (MoE selects 6/64 experts).
But the MLA attention reads ALL 27 layers' projection weights on every step.

**Bandwidth per matmul element:**

| What is read | project-zero (current) | llama.cpp |
|---|---|---|
| Weight (Q4K nibble) | 0.5 bytes | 0.5 bytes |
| Activation per multiply | **4.0 bytes** (F32) | **1.0 byte** (Q8 int8) |
| **Total per element** | **4.5 bytes** | **1.5 bytes** |
| **Ratio** | **3× MORE bandwidth** | baseline |

This 3× activation bandwidth penalty, combined with slightly lower instruction efficiency
(F32 FMA needs 2 conversion instructions vs `maddubs`'s zero conversions) explains the 7.3× gap.

### 5.2 IPC Evidence (from perf measurements in prior sessions)

IPC (Instructions Per Cycle) at T=1:
- project-zero: **1.63** IPC
- llama.cpp: **2.33** IPC

Lower IPC from project-zero means: the CPU is frequently stalling, waiting for data from RAM.
This is the signature of a memory-bandwidth-bound workload. Adding compute (wider SIMD,
more threads, etc.) does not help — the stall is on the load units, not the arithmetic units.

### 5.3 What llama.cpp Does Differently

**Key reference:** `~/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` line 1742
**Function:** `ggml_vec_dot_q4_K_q8_K()`

Before calling this function, llama.cpp calls `quantize_row_q8_K_ref()` to convert the
F32 activation to Q8_K format (int8 per element, with a scale per 256 elements). Then:

```c
// Per superblock (256 elements, 128 bytes of Q4 weights):
const __m256i m4 = _mm256_set1_epi8(0xF);
__m256i sumi = _mm256_setzero_si256();

for (int j = 0; j < QK_K/64; ++j) {
    __m256i q4bits = _mm256_loadu_si256(q4);  // 32 bytes → 64 nibbles
    __m256i q4l    = _mm256_and_si256(q4bits, m4);
    __m256i q4h    = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

    __m256i q8l    = _mm256_loadu_si256(q8);   // 32 bytes, int8 activations
    __m256i p16l   = _mm256_maddubs_epi16(q4l, q8l);  // uint8 × int8 → int16
    p16l           = _mm256_madd_epi16(scale_l, p16l); // scale → int32
    sumi           = _mm256_add_epi32(sumi, p16l);
    // ... repeat for hi nibbles
}
// Final: acc += d * q8_scale * hsum32(sumi)
```

The critical instruction is `_mm256_maddubs_epi16`:
- Multiplies 32 unsigned int8 (Q4 nibbles: 0–15) by 32 signed int8 (Q8 activations: -127..127)
- Produces 16 int16 sums — no conversion, no F32, no intermediate allocation
- One instruction processes 32 weight elements

**Why we can use it:** Q4K nibbles are always in [0, 15] — they fit in unsigned int8. Q8 activations
are in [-127, 127] — they fit in signed int8. `maddubs` is specifically designed for this pattern.

---

## 6. What Already Exists in the Codebase (Do Not Re-Implement)

Before writing any new code, understand what is already built. These are the building blocks
for the fix.

### 6.1 The Dispatch Is Already Correct

`mla_attention.c` contains this macro (lines ~100–115):
```c
#define MLA_MATMUL(out, in, wptr, sptr, n_in, n_out)        \
    do {                                                     \
        if (w->has_mla_quant) {                              \
            parallel_matmul_q4k(out, in, wptr, n_in, n_out, tp); \
        } else if (is_float) {                               \
            parallel_matmul_float32(out, in, wptr, n_in, n_out, tp); \
        } else {                                             \
            parallel_ternary_matmul_packed(...);             \
        }                                                    \
    } while (0)
```

`w->has_mla_quant` is set to `true` in `gguf_loader.c` by the `DS_LOAD_MLA_PROJ_HEAP` macro
whenever the MLA projection weights are Q4_K type. This means all 5 MLA calls already go to
`parallel_matmul_q4k()`. **No change needed here.**

Similarly, `moe_ffn.c` shared expert path already calls `parallel_matmul_q4k()` for shared
expert w1/w2/w3 when the type is `GGUF_TYPE_Q4_K`. **No change needed here either.**

### 6.2 Activation Pre-Quantization Infrastructure Already Exists

`include/math/quantize_i8.h` declares:
```c
float quantize_row_to_i8(const float *x, int8_t *q, int n);          // scalar
float quantize_row_to_i8_avx2(const float *x, int8_t *q, int n);     // AVX2 accelerated
float quantize_row_to_i8_avx512(const float *x, int8_t *q, int n);   // AVX-512 accelerated
int32_t sum_i8(const int8_t *arr, int n);
```

These already exist in `src/math/quantize_i8.c`. They quantize a float vector to int8 with a
**single scale for the whole vector**. We need a variant with **per-256-element-block scaling**
to match Q4K's superblock granularity. That variant does not yet exist — we add it.

### 6.3 The Pre-Quantized Activation Pattern Already Exists (for Ternary Matmul)

`include/math/parallel_matmul.h` defines:
```c
typedef struct {
    const int8_t *q_x;       /* points into caller's buffer */
    float         act_scale;  /* scale to reconstruct F32 */
    int32_t       sum_qx;     /* sum of all int8 values (for bias correction) */
    int           valid;      /* 1 = successfully quantized, 0 = fallback */
} TnPreqActivation;

int tn_preq_prepare(TnPreqActivation *preq, int8_t *buf, const float *x, int n);
void parallel_ternary_matmul_packed_preq(float *out, const float *x, const tn_u8 *w,
    int n, int d, float scale, const TnPreqActivation *preq, ThreadPool *tp);
```

The pattern: **quantize activation once → pass pre-quantized activation to all rows**.
This is exactly what we need for Q4K. The ternary implementation pre-quantizes `s->xb`
once before the gate and up projections (two matmul calls), saving 1 redundant quantization.

**Critical limitation:** `tn_preq_prepare()` only activates on `TN_HAS_AVX512VNNI`:
```c
int tn_preq_prepare(TnPreqActivation *preq, int8_t *buf, const float *x, int n) {
    preq->valid = 0;
#if TN_HAS_AVX512VNNI
    // ... actual quantization ...
    preq->valid = 1;
#else
    (void)buf; (void)x; (void)n;  // ← does nothing on AVX2 machines
    return 0;
#endif
}
```

The test machine (i5-5250U) has **AVX2 only** — no AVX-512 VNNI. So `preq->valid` is always 0
and the ternary fast path is never taken on this machine. This is why the analogous Q4K fast path
has never been implemented — there was no proven template to follow on AVX2.

**For our Q4K fix:** We add a separate `tn_q8k_act_quantize()` function that works on AVX2.
It uses a different quantization granularity (per-256-block instead of per-vector) and does not
need the `sum_qx` bias correction term used by VNNI ternary.

### 6.4 Weight Loading Is Already Correct

`src/core/gguf_loader.c` uses `DS_LOAD_MLA_PROJ_HEAP` to load MLA weights:
```c
// Copies raw Q4_K bytes to a heap buffer at model load time.
// Sets w->has_mla_quant = true.
// The heap copy costs ~30 MB of RAM but amortizes cold SSD page faults.
DS_LOAD_MLA_PROJ_HEAP(w->mla_wq,     "blk.%d.attn_q.weight",    q_rows * dim,    &w->has_mla_quant);
DS_LOAD_MLA_PROJ_HEAP(w->mla_wkv_a,  "blk.%d.attn_kv_a.weight", kva_rows * dim,  &w->has_mla_quant);
DS_LOAD_MLA_PROJ_HEAP(w->mla_wkv_b,  "blk.%d.attn_kv_b.weight", kvb_rows * lora, &w->has_mla_quant);
DS_LOAD_MLA_PROJ_HEAP(w->wo,         "blk.%d.attn_output.weight", dim * n_heads_v, &w->has_mla_quant);
```

The heap buffers contain raw Q4_K bytes in the GGUF `block_q4_K` layout (2+2+12+128 = 144 bytes
per 256 elements). **This is identical to llama.cpp's `block_q4_K` struct.** No format conversion
needed — we can pass these bytes directly to the fused kernel. **Do not change this.**

---

## 7. What "libggml Is Already Linked" and "Fast Path Already Configured" Mean

These phrases were stated in a prior session. After exhaustive code search:

**"libggml is already linked":**  
`grep -rn "libggml\|ggml.h\|TN_GGML\|GGML_BACKEND" Makefile src/ include/` returns zero hits.  
There is **no GGML linkage** in the codebase. This recollection is likely from planning discussions
about a potential temporary bridge, which was never implemented. **We do not need to link libggml.**
The algorithm has been fully read from `ggml_vec_dot_q4_K_q8_K()` and will be implemented natively
in pure C. Linking libggml would add a macOS-only build dependency and violate the engine's
zero-external-dependency design principle.

**"Fast path already configured":**  
This is **architecturally true** at the dispatch level:
- `MLA_MATMUL` macro already routes to `parallel_matmul_q4k()` when `has_mla_quant=true` ✅
- `moe_ffn.c` shared expert path already calls `parallel_matmul_q4k()` for Q4K type ✅
- `TnPreqActivation` / `quantize_row_to_i8` infrastructure already exists ✅
- Weight heap copies (raw Q4K bytes) are already correct ✅

What is NOT yet done: the inner kernel `dot_q4k_row()` still reads **F32 activations**.
The "fast path" is wired at the dispatch level but the actual speed improvement lives in the
kernel, one level below dispatch. That kernel is what we are implementing.

---

## 8. The AP Graveyard — Seven Failed Approaches (Do Not Retry)

These approaches were tried in prior sessions and either caused regressions or had no effect.
Every item below is documented in `BENCHMARK_REPORT.md` Addendum AP (lines 902–1118).

### 8.1 SIMD mode switching with `--simd` flag (AP.1) — NO EFFECT

**What was tried:** Running `--simd scalar`, `--simd avx2`, `--simd avx512f` to see if changing
SIMD mode improved DeepSeek performance.

**Why it had no effect:** The `--simd` flag only controls the **ternary matmul** dispatch table
in `src/math/simd_dispatch.c`. It does NOT affect:
- `matmul_f16.c` (compile-time `#elif TN_HAS_AVX2` — no runtime switch)
- `matmul_q4k.c` (compile-time `#elif TN_HAS_AVX2` — no runtime switch)

This was confirmed by reading the source: `src/math/simd_dispatch.c` only populates a function
pointer for `tn_ternary_matmul`. The Q4K and F16 paths use compile-time selection. The `--simd`
flag is irrelevant to DeepSeek performance. Do not waste time testing it again.

### 8.2 INT8/INT4 classifier precision (AP.2) — ALREADY DONE, MINIMAL IMPACT

**What was tried:** Changing lm_head (vocabulary projection) from BF16 to INT8/INT4.

**Result:** +85% on lm_head alone. But lm_head is approximately 1.5% of decode time for DeepSeek
(because the model spends 95%+ of time in MLA and MoE matmuls). So the full model speedup was
negligible (~1.02×). This is already implemented and committed. Do not revisit.

### 8.3 Thread count tuning above T=4 (AP.3) — T=4 IS CONFIRMED OPTIMAL

**Finding:** The machine has 2 physical cores / 4 logical threads. T=4 is optimal.
T=5+ degrades performance (memory bandwidth shared across more threads, no additional parallelism).
Both llama.cpp and bitnet.cpp **hang entirely** at T=5+ on this machine (never complete in 2+ minutes).
project-zero runs at T=5–8 but degrades to 6–7 tok/s. T=4 is the ceiling. Do not test T>4.

### 8.4 DS_LOAD_MLA_PROJ zero-copy mmap path (AP.4) — 2.5× REGRESSION

**What was tried:** Using the mmap pointer directly (no heap copy) for MLA weight pointers.
This is the `DS_LOAD_MLA_PROJ` macro in `gguf_loader.c` (still present, but inactive).

**Result:** 0.69 tok/s vs 1.75 tok/s baseline — a 2.5× regression.

**Why:** On macOS with 8 GB RAM and a 8.9 GB model, the mmap region is not resident in RAM.
Each Q4K weight access triggers an SSD page fault. The 8 KB OS pages are evicted before
the next token's access. The "zero-copy" becomes "zero-cache" — extremely slow.

The heap copy (DS_LOAD_MLA_PROJ_HEAP) amortizes this: copy the Q4K bytes once at load time
into a heap buffer that fits within available RAM. The buffer stays hot in the OS page cache.

**IMPORTANT:** Do NOT re-enable `DS_LOAD_MLA_PROJ` without first implementing the Q8 fused
kernel AND verifying the model fits in RAM. The zero-copy approach is architecturally sound
on a machine with sufficient RAM — but it is wrong for this machine and for the current kernel.

### 8.5 Wider SIMD on existing F32 activation path (AP.7) — BANDWIDTH-BOUND

**What was tried:** Using AVX-512 16-wide operations in the existing decode-then-multiply loop
in `dot_q4k_row()`.

**Why it failed:** The kernel is **memory-bandwidth-bound** (proven by IPC=1.63). The CPU has
more arithmetic units available (could do more FMA instructions per cycle) but they sit idle
waiting for data from RAM. Making the arithmetic wider does not make RAM faster. The bottleneck
does not move. This was confirmed by perf measurements showing IPC did not increase.

The only fix for a bandwidth-bound kernel is to **reduce bytes read**, not to process more per cycle.

### 8.6 vmtouch / Transparent Hugepage tuning (AP.5) — MEASUREMENT HYGIENE ONLY

**Finding:** Without pinning the model in page cache (vmtouch on Linux) or warming it
(`cat model > /dev/null` on macOS), measured throughput shows 0.38–0.57 tok/s — 7.3× below
the real baseline. This is not a regression; it is mmap page-fault overhead during the measurement
window. **These steps are measurement prerequisites, not optimizations.** Always warm the cache
before benchmarking. Do not report these numbers as performance data.

Note: vmtouch is Linux-only. macOS equivalent: `cat models/model.gguf > /dev/null` before the
first benchmark run. This fills the OS page cache. Subsequent runs read from RAM, not SSD.

### 8.7 "scalar --simd beats avx2 --simd" finding from session AQ (AP.8 context) — MISLEADING

**What was reported:** In session AQ (Addendum AQ), using `--simd scalar` appeared to produce
better DeepSeek performance than `--simd avx2`. This was confusing and caused wasted investigation.

**Root cause:** As explained in 8.1, `--simd` does not affect the Q4K kernel. The apparent
difference was caused by run-to-run variance in a single-measurement, cold-cache test. The
methodology was flawed (no warmup, no multiple runs). Once fair v3 methodology was established
(warm cache + 3-run average), the `--simd` flag has zero measurable effect on DeepSeek throughput.

---

## 9. The Correct Fix — Precise Implementation Plan

### 9.1 Scope: ONE File

**All code changes go in `src/math/matmul_q4k.c`.** No other source file needs modification.

- `mla_attention.c` — no change (dispatch already correct)
- `moe_ffn.c` — no change in Phase 1; Phase 2 changes expert w1/w3 dispatch
- `include/core/weights.h` — no change (`has_mla_quant` flag already exists)
- `src/core/gguf_loader.c` — no change (`DS_LOAD_MLA_PROJ_HEAP` already correct)
- `include/math/matmul_q4k.h` — no change (internal functions only, caller API unchanged)

The `parallel_matmul_q4k()` function signature stays identical. All callers continue to pass
`const float *x` as the activation. Pre-quantization happens transparently inside the function,
before threads are dispatched. Callers see no difference.

This satisfies **Golden Rule 1B (Maximum Modularity):** the fix stands on its own.

### 9.2 What Changes Inside matmul_q4k.c

**Three additions:**

**Addition 1 — `tn_q8k_act_quantize()` (activation quantizer)**

New static function. Converts F32 vector to int8 with per-256-element-block scaling.

```c
/*
 * Quantize float32 activation x[n] to int8 per Q4K superblock granularity.
 * qs_buf[n]:          output int8 values (caller allocates n bytes)
 * scales_buf[n/256]:  output per-block F32 scales (caller allocates n/256 * 4 bytes)
 * n must be a multiple of 256 (Q4K_SUPER).
 *
 * Per block of 256 elements:
 *   scale = max(|x[i]|) / 127.0f   (symmetric, avoids -128)
 *   q[i]  = clamp(round(x[i] / scale), -127, 127)
 *   scales_buf[b] = scale
 *
 * Uses quantize_row_to_i8_avx2() per block when TN_HAS_AVX2.
 * Falls back to scalar per block otherwise.
 *
 * Why per-256-block (not per-vector)?
 *   A 5120-element vector spans 20 superblocks. If one superblock has very
 *   different magnitude than another (e.g. attention logit scale vs residual scale),
 *   a single vector scale under-quantizes high-magnitude blocks (losing precision)
 *   and over-quantizes low-magnitude blocks (saturating to ±127 everywhere).
 *   Per-block scaling matches Q4K's superblock granularity exactly.
 */
static void tn_q8k_act_quantize(int8_t *qs_buf, float *scales_buf,
                                 const float *x, int n) {
    int n_blocks = n / Q4K_SUPER;  /* Q4K_SUPER = 256 */
    for (int b = 0; b < n_blocks; b++) {
        const float *xb  = x      + (size_t)b * Q4K_SUPER;
        int8_t      *qb  = qs_buf + (size_t)b * Q4K_SUPER;
#if TN_HAS_AVX2
        scales_buf[b] = quantize_row_to_i8_avx2(xb, qb, Q4K_SUPER);
#else
        scales_buf[b] = quantize_row_to_i8(xb, qb, Q4K_SUPER);
#endif
    }
}
```

Note: `quantize_row_to_i8_avx2()` is already implemented in `src/math/quantize_i8.c` and
declared in `include/math/quantize_i8.h`. We are reusing existing infrastructure.

**Addition 2 — `dot_q4k_row_q8act()` (the fused kernel)**

New static function. Replaces `dot_q4k_row()` when Q8 activation is available.

```c
/*
 * Fused Q4K row dot product with pre-quantized int8 activation.
 *
 * row_q4k:  raw Q4K bytes for one weight row (n_super × 144 bytes)
 * q8act:    int8 activation values (n elements, quantized per 256-block)
 * q8scales: float scale per 256-element block (n/256 values)
 * n:        inner dimension (multiple of 256)
 *
 * Algorithm (AVX2 path):
 *   For each superblock b (256 elements):
 *     1. Decode d, dmin from F16 bytes 0-3
 *     2. Decode 8 (scale, min) pairs from sc12[12 bytes]
 *     3. For each of 4 groups g of 64 elements (32 lo + 32 hi nibbles):
 *        a. q4bits = _mm256_loadu_si256(qs + g*32)
 *        b. q4l = q4bits & 0xF;  q4h = (q4bits >> 4) & 0xF
 *        c. q8l = _mm256_loadu_si256(q8act + b*256 + g*64)
 *        d. q8h = _mm256_loadu_si256(q8act + b*256 + g*64 + 32)
 *        e. p16l = _mm256_maddubs_epi16(q4l, q8l)  ← KEY INSTRUCTION
 *        f. p16l = _mm256_madd_epi16(scale_i16_l, p16l) → int32
 *        g. sumi += p16l (int32 accumulate)
 *        h. Also accumulate q8_sum_l, q8_sum_h (for min correction)
 *     4. dot += (d * q8scales[b]) * hsum32(sumi)
 *                - dmin * q8scales[b] * q8_sum_all (min correction)
 *
 * Why maddubs works here:
 *   q4 nibbles [0..15] fit in unsigned uint8 (left operand of maddubs)
 *   q8 activations [-127..127] fit in signed int8 (right operand of maddubs)
 *   maddubs(uint8 a, int8 b): computes a[i]*b[i] + a[i+1]*b[i+1] → int16
 *   This is exactly Q4K × Q8 with no conversion overhead.
 */
static float dot_q4k_row_q8act(const uint8_t *row_q4k,
                                const int8_t *q8act,
                                const float *q8scales, int n);
```

The full AVX2 implementation follows the same structure as the existing `dot_q4k_row()` AVX2 path
(same superblock loop, same scale decode), but replaces:
```c
// OLD: load F32 activation → convert to float → fmadd_ps
__m256 xv_lo = _mm256_loadu_ps(xl + j);          // 32 bytes
acc_lo = _mm256_fmadd_ps(lo_f, xv_lo, acc_lo);   // F32 multiply
```
With:
```c
// NEW: load Q8 activation → maddubs directly
__m256i q8l  = _mm256_loadu_si256((const __m256i*)(q8 + offset));  // 32 bytes
__m256i p16l = _mm256_maddubs_epi16(q4l, q8l);                     // INT16 multiply
```

The min correction changes from `sum_xl` (sum of F32 activations) to
`(q8scales[b] * sum_q8l)` where `sum_q8l` is the sum of the int8 activation values
in that sub-group. This is equally accurate and avoids reading F32 values.

**Addition 3 — Extend `MatmulQ4KArgs` and `parallel_matmul_q4k()`**

```c
typedef struct {
    float         *out;
    const float   *x;          /* original F32 activation (kept for fallback) */
    const uint8_t *w;
    int            n, d;
    size_t         row_bytes;
    /* Phase 1 additions — pre-quantized activation (NULL = use F32 path) */
    const int8_t  *q8_act;     /* NULL or n bytes of int8 activation */
    const float   *q8_scales;  /* NULL or (n/256) float scales */
} MatmulQ4KArgs;
```

Modified `parallel_matmul_q4k()`:
```c
void parallel_matmul_q4k(float *out, const float *x, const uint8_t *w_q4k,
                           int n, int d, ThreadPool *tp) {
    int n_blocks = n / Q4K_SUPER;

    /* Pre-quantize activation once (shared read-only across all threads).
     * Stack allocation: n bytes (≤ 5120 for DeepSeek) + n_blocks*4 bytes.
     * Safe: thread stacks are 8 MB; max usage here is ~5.2 KB.
     * Guard: fall back to F32 path for unusually large n. */
    int8_t *q8_buf   = NULL;
    float  *q8_scales = NULL;
    int8_t  q8_stack[8192];      /* covers up to n=8192 on stack */
    float   sc_stack[32];         /* covers up to n=8192/256=32 blocks */

    if (n <= 8192 && n % Q4K_SUPER == 0) {
        q8_buf    = q8_stack;
        q8_scales = sc_stack;
        tn_q8k_act_quantize(q8_buf, q8_scales, x, n);
    }
    /* For n > 8192: q8_buf remains NULL → uses F32 path (correctness preserved) */

    MatmulQ4KArgs args = {
        .out       = out,
        .x         = x,
        .w         = w_q4k,
        .n         = n,
        .d         = d,
        .row_bytes = (size_t)n_blocks * Q4K_BYTES,
        .q8_act    = q8_buf,
        .q8_scales = q8_scales,
    };
    if (!tp) {
        matmul_q4k_task(&args, 0, 0, d);
        return;
    }
    threadpool_dispatch(tp, matmul_q4k_task, &args, d);
}
```

Modified `matmul_q4k_task()`:
```c
static void matmul_q4k_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    MatmulQ4KArgs *a = (MatmulQ4KArgs *)arg;
    for (int i = start; i < end; i++) {
        if (a->q8_act) {
            /* Fast path: pre-quantized int8 activation with maddubs kernel */
            a->out[i] = dot_q4k_row_q8act(
                a->w + (size_t)i * a->row_bytes,
                a->q8_act, a->q8_scales, a->n);
        } else {
            /* Fallback: original F32 activation path (unchanged correctness) */
            a->out[i] = dot_q4k_row(
                a->w + (size_t)i * a->row_bytes, a->x, a->n);
        }
    }
}
```

**Why stack allocation is safe:**
- DeepSeek max activation dimension: `dim=2048` → 2048 bytes Q8 + 8 floats scales = 2080 bytes
- MLA KV compress input: 2048 → 2048 bytes
- MoE shared expert input: 2048 → 2048 bytes  
- All well within the 8192-byte guard and far below the 8 MB thread stack limit

**Why threads safely share `q8_buf` (read-only):**
The q8_buf is pre-filled before `threadpool_dispatch()` is called. All threads only read it;
no thread writes to it. The stack buffer is alive for the duration of `parallel_matmul_q4k()`'s
stack frame, which encompasses the entire `threadpool_dispatch()` call. Safe.

### 9.3 Same Change for Batch Variant

`parallel_matmul_q4k_batch()` handles multiple experts sharing the same input `x`. The same
pre-quantization applies — quantize `x` once, pass to all rows across all experts:

```c
void parallel_matmul_q4k_batch(float * const *outs, const float *x, ...) {
    /* Same tn_q8k_act_quantize() call as parallel_matmul_q4k() */
    /* MatmulQ4KBatchArgs gets q8_act / q8_scales fields */
    /* matmul_q4k_batch_task() dispatches dot_q4k_row_q8act() when q8_act != NULL */
}
```

---

## 10. Phase 2 — MoE Expert w1/w3 (After Phase 1 Validated)

Phase 1 fixes MLA projections and shared expert matmuls (all of which go through
`parallel_matmul_q4k()`). Phase 2 fixes the routed expert w1/w3 paths in `moe_ffn.c`.

**Current state (Phase 2 target):**
```c
// In moe_ffn.c sequential expert loop (has_expert_quant=true, Q4K type):
float *fw1 = dequant_expert_weight(w->moe_w1[layer][e], Q4_K, w13_elems);
parallel_matmul_float32(s->hb, s->xb, fw1, dim, expert_hdim, tp);  // ← F32 path
```

**After Phase 2:**
```c
// w1/w3 go through parallel_matmul_q4k() which now uses Q8 activation internally
if (w->expert_w13_quant_type == GGUF_TYPE_Q4_K) {
    parallel_matmul_q4k(s->hb, s->xb, (const uint8_t*)w->moe_w1[layer][e],
                         dim, expert_hdim, tp);
    parallel_matmul_q4k(s->hb2, s->xb, (const uint8_t*)w->moe_w3[layer][e],
                         dim, expert_hdim, tp);
}
```

Note: the batched variant `parallel_matmul_q4k_batch()` exists exactly for this use case —
processing k experts' gate and up projections with a shared activation. Phase 2 wires this up.

**Why Phase 2 is separate from Phase 1:**
Expert w1/w3 pointers are currently set to F32 dequanted buffers at load time (via
`dequant_expert_weight()` which lazily dequantizes). To use `parallel_matmul_q4k()` for w1/w3,
the raw Q4K pointers must be stored instead of the F32 buffers. This requires a small change
in `gguf_loader.c` to set `moe_w1[layer][e]` to the raw Q4K pointer. That change touches
weight loading — higher risk, different PR. Phase 1 (pure kernel change) should be validated
first to prove correctness before introducing weight pointer changes.

---

## 11. Model Download — Machine Constraints

### 11.1 Hardware

```
CPU:      Intel Core i5-5250U (Broadwell, 2 physical / 4 logical cores, 1.6 GHz base)
SIMD:     AVX2 + F16C only (no AVX-512 of any kind, no VNNI, no NEON)
RAM:      8 GB DDR3-1600 (measured bandwidth: 5.1–5.3 GB/s)
Storage:  SSD (read: ~500 MB/s sequential)
OS:       macOS
```

### 11.2 Model Size vs Available RAM

| Quant | Est. size | Fits in 8 GB RAM? | Use for |
|-------|-----------|-------------------|---------|
| Q2_K  | ~2.4 GB   | ✅ (easily)        | Quick correctness checks only — quality degrades |
| Q3_K_M | ~5.1 GB  | ✅ (recommended)   | **Development + benchmarking on this machine** |
| Q4_K_S | ~8.9 GB  | ⚠️ exceeds physical RAM, uses SSD-backed virtual memory | Final comparison with prior AN results |
| Q4_K_M | ~9.1 GB  | ⚠️ same as above  | Skip for now |

**Recommendation:** Download Q3_K_M for this machine. Prior benchmarks (Addenda AN, AP) used
Q4_K_S on a 16 GB Linux machine. When comparing, document the quant difference clearly.

### 11.3 Download Command

```bash
pip install huggingface_hub

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/DeepSeek-V2-Lite-Chat-GGUF',
    filename='DeepSeek-V2-Lite-Chat-Q3_K_M.gguf',
    local_dir='models/'
)
"
```

---

## 12. Testing Protocol — Golden Rules Compliance

### Rule 2: Test After EVERY Change

After adding `tn_q8k_act_quantize()`, after adding `dot_q4k_row_q8act()`, and after wiring
them into `matmul_q4k_task()` — run the correctness check EACH TIME:

```bash
make -j4 && ./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q3_K_M.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 12 --temperature 0.0 --threads 4
# MUST contain: "Paris"

./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q3_K_M.gguf \
  --prompt "What is the capital of Germany?" \
  --max-tokens 12 --temperature 0.0 --threads 4
# MUST contain: "Berlin"
```

If either test fails → **REVERT IMMEDIATELY** before adding more code:
```bash
git checkout src/math/matmul_q4k.c
```

### Rule 4: Expert Layer Activity (Silent Corruption Detection)

Q8 quantization errors can cause NaN propagation. The symptom is that only L00–L01 show
expert activity while L02–L26 are silent (NaN propagated from the first MoE layer silences all
subsequent attention scores). Check with:

```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q3_K_M.gguf \
  --prompt "Explain the French Revolution in detail." \
  --max-tokens 64 --temperature 0.0 --threads 4 --verbose 2>&1 | grep "layer\|expert\|L[0-9]"
```

Expected: all 27 layers show expert activity with multiple different experts selected.
Failure: only L00 or L01 active → NaN in Q8 kernel. Revert.

### Rule 5: Performance Regression Check

After Phase 1:
```bash
cat models/deepseek-v2-lite-chat-Q3_K_M.gguf > /dev/null  # warm page cache (macOS)
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q3_K_M.gguf \
  --prompt "Explain the history of the Roman Empire from founding to fall in great detail." \
  --max-tokens 64 --temperature 0.0 --threads 4 2>&1 | grep "tok/s"
```

Performance thresholds:
- Phase 1 minimum acceptable: **≥ 0.9 tok/s** (must not regress from current 1.90 on Q4_K_S)
- Phase 1 target: **≥ 5 tok/s** (expected from 3× bandwidth reduction)
- Phase 2 target: **≥ 10 tok/s** (MoE expert paths fixed too)
- Regression gate: >5% drop from previous commit → revert

### Rule 5: SmolLM2 Regression Check (Do Not Break F16 Path)

Changes to `matmul_q4k.c` must not affect the F16 matmul path (completely separate code path).
Verify anyway:

```bash
./adaptive_ai_engine \
  --model models/SmolLM2-135M-Instruct-f16.gguf \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 5 --temperature 0.0 --threads 4
# Must output: "Paris"
```

### Rule 8: Document Exact Commands

Every benchmark result must record ALL flags. From GOLDEN_RULES.md:
```
Date: YYYY-MM-DD HH:MM UTC | Commit: <SHA>
Command: ./adaptive_ai_engine --model <path> --prompt "<text>" \
         --max-tokens N --temperature T --threads N --simd <mode> --classifier <mode>
Output:  <exact text>
tok/s:   X.XX tok/s (N tokens)
```

### Final Fair Benchmark (run sequentially, same terminal session)

```bash
# 1. Warm macOS page cache
cat models/deepseek-v2-lite-chat-Q3_K_M.gguf > /dev/null

# 2. PZ T=1..4, 3 runs each (warmup run already done above)
PROMPT="Tell me everything you know about the history of ancient Rome, from founding through the fall of the Western Empire. Include Julius Caesar, Augustus, Constantine, and major battles."
for t in 1 2 3 4; do
  echo "=== T=$t ==="
  for r in 1 2 3; do
    ./adaptive_ai_engine --model models/deepseek-v2-lite-chat-Q3_K_M.gguf \
      --prompt "$PROMPT" --max-tokens 64 --temperature 0.0 --threads $t \
      --classifier int8 2>&1 | grep "tok/s"
  done
done

# 3. llama.cpp reference (in-process r=3, sequential)
~/llama.cpp/build/bin/llama-bench \
  -m models/deepseek-v2-lite-chat-Q3_K_M.gguf \
  -t 1 -t 2 -t 3 -t 4 \
  -n 64 -p 20 -r 3 -ngl 0
```

---

## 13. File-by-File Summary of All Changes

| File | Phase | Change | Why |
|------|-------|--------|-----|
| `src/math/matmul_q4k.c` | 1 | Add `tn_q8k_act_quantize()`, `dot_q4k_row_q8act()`, extend `MatmulQ4KArgs`, modify `parallel_matmul_q4k()` and `parallel_matmul_q4k_batch()` | Core fix — replaces F32 with Q8 activation |
| `include/math/matmul_q4k.h` | 1 | No change | Caller API unchanged |
| `src/transformer/mla_attention.c` | 1 | **No change** | MLA_MATMUL macro already routes to parallel_matmul_q4k() |
| `src/transformer/moe_ffn.c` | 2 | Route expert w1/w3 to `parallel_matmul_q4k()` for Q4K type | Removes F32 dequant for MoE experts |
| `src/core/gguf_loader.c` | 2 | Store raw Q4K pointer for expert w1/w3 (not F32 dequant) | Enables zero-copy expert weight access |
| `include/core/weights.h` | 1 | **No change** | has_mla_quant flag already exists |
| `BENCHMARK_REPORT.md` | both | Append new addendum after each benchmark run | Rule 6 |
| `DEBUGGING_JOURNAL.md` | both | Dated entry for each change | Rule 6 |
| `.claude/BENCHMARK_SUMMARY.md` | both | Update with new numbers | GitNexus freshness |

---

## 14. Pre-Commit Checklist

```
[ ] make -j4 succeeds with no errors or new warnings
[ ] France → Paris output correct (DeepSeek Q3_K_M, T=4, temp=0)
[ ] Germany → Berlin output correct
[ ] All 27 transformer layers show expert activity (not just L00-L01)
[ ] tok/s >= previous commit baseline at T=4 (within 5%)
[ ] SmolLM2 F16 still outputs "Paris" correctly
[ ] No hardcoded dimension constants (all derived from n, Q4K_SUPER, at runtime)
[ ] AVX2 path has #if TN_HAS_AVX2 guard with scalar #else fallback
[ ] Stack buffer guard: if (n > 8192) fallback to F32 path (q8_buf=NULL)
[ ] DEBUGGING_JOURNAL.md updated: date, command, output, tok/s, what changed
[ ] Commit message: "<subsystem>: <one-line summary>" then details
[ ] Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## 15. Session History (for context continuity across AI sessions)

| Session | Machine | Key event | Result |
|---------|---------|-----------|--------|
| 2025 early | Linux i5-11300H 16GB | Initial DeepSeek Q4K integration | 0.9 tok/s |
| 2026-03-22 | Linux i5-11300H 16GB | DS_LOAD_MLA_PROJ_HEAP heap copy MLA fix | **1.90 tok/s** (AN) |
| 2026-03-22 | Linux i5-11300H 16GB | DS_LOAD_MLA_PROJ zero-copy attempt | 0.69 tok/s regression — REVERTED |
| 2026-03-23 | macOS i5-5250U 8GB | SmolLM2 fair benchmark v3 + GitNexus | PZ beats llama+bitnet at T=3,4 |
| 2026-03-24 | macOS i5-5250U 8GB | F16 prefetch investigation | No prefetch is correct (reverted) |
| 2026-03-24 | macOS i5-5250U 8GB | This document written; plan approved | — |
| **Pending** | macOS i5-5250U 8GB | Phase 1: Q8 fused kernel | Target ≥ 5 tok/s |
| **Pending** | macOS i5-5250U 8GB | Phase 2: MoE expert Q4K wiring | Target ≥ 10 tok/s |

---

## 16. Glossary (for developers new to this domain)

| Term | Meaning |
|------|---------|
| **tok/s** | Tokens per second — how fast the model generates text. Higher is better. |
| **Q4_K** | 4-bit quantization with K-type (superblock) scale structure. 144 bytes per 256 weights. |
| **Q8** / **Q8_K** | 8-bit (int8) quantization. 1 byte per weight. Used for activations, not stored weights. |
| **MLA** | Multi-head Latent Attention — DeepSeek's compressed KV cache attention mechanism. |
| **MoE** | Mixture of Experts — 64 specialist FFNs per layer; only 6 are activated per token. |
| **gemv** | General matrix-vector product. Each decode step is a batch of gemv operations. |
| **bandwidth-bound** | Performance limited by how fast data can be read from RAM, not by CPU compute speed. |
| **maddubs** | `_mm256_maddubs_epi16` — AVX2 instruction: multiplies 32 uint8×int8 pairs → 16 int16 sums. |
| **superblock** | 256 consecutive weight elements with shared scale metadata in Q4_K format. |
| **SIMD** | Single Instruction Multiple Data — CPU feature that processes N values per clock cycle. |
| **AVX2** | Intel/AMD SIMD extension — 256-bit registers, processes 8 F32 or 32 int8 at once. |
| **VNNI** | Vector Neural Network Instructions — specialized int8 multiply-accumulate (not on this machine). |
| **IPC** | Instructions Per Cycle — ratio of instructions completed to clock cycles elapsed. Low IPC = stalled on memory. |
| **GGUF** | GGML Universal Format — binary file format for quantized LLM weights (used by llama.cpp). |
| **mmap** | Memory-map — treat a file as virtual memory. Fast when file fits in RAM; slow when it doesn't. |
| **preq** | Pre-quantized activation — computing the int8 quantization of the activation vector once and reusing it across multiple matmul calls that share the same input. |
| **DS_LOAD_MLA_PROJ_HEAP** | Macro in `gguf_loader.c` that heap-copies raw Q4K bytes for MLA weight tensors. |
| **has_mla_quant** | Boolean flag in `TransformerWeights` — true when MLA weights are stored as raw Q4K bytes. |

---

*This document must be updated after every implementation milestone.*  
*AI agents: read Sections 5 through 12 in order. The graveyard in Section 8 is mandatory reading.*  
*Do not implement anything from Sections 9–10 without first reading GOLDEN_RULES.md.*  
*Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>*
