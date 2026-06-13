# Kernel Internals — Project Zero

A technical reference for contributors who want to understand how the fast paths work before diving into the code. Reading this alongside `src/math/` will save you several hours of reverse-engineering.

---

## 1. The Ternary Weight Problem

BitNet b1.58-2B-4T weights are **ternary**: each weight is one of {−1, 0, +1}. This creates an unusual arithmetic situation — the dominant operation in a transformer forward pass (dense matrix-vector multiply) reduces to:

```
output[i] = sum_j (w[i][j] * x[j])   where w[i][j] ∈ {-1, 0, +1}
```

Multiplication by ternary values is not multiplication at all: it's either a negation, a no-op, or a zero. The naive approach (dequantize ternary weights to FP32, run standard FMA) wastes the dominant fraction of memory bandwidth on loading floats that encode only 1.58 bits of information.

### Packing

Project Zero packs 4 ternary weights per byte using a 2-bit encoding:
```
0b00 = −1
0b01 =  0
0b10 = +1
0b11 = (unused / padding)
```

A 2048-wide weight row consumes 512 bytes instead of 8192 bytes (FP32) or 2048 bytes (INT8). The LM head uses INT4 quantization for its larger weight matrix.

---

## 2. The VBMI Kernel (AVX-512, the Fast Path)

The hot path on AVX-512 VBMI machines (Ice Lake Server, Sapphire Rapids, Tiger Lake, Zen 4) runs three instructions per 64-element block instead of the unpack→multiply→accumulate sequence.

**File:** `src/math/ternary_matmul_packed_vbmi.c`

### Step 1: `vpermi2b` — table lookup decode

```c
// Build a 64-byte LUT indexed by the 2-bit ternary encoding
// lut[0x00] = -1, lut[0x01] = 0, lut[0x02] = +1
__m512i lut = _mm512_set1_epi8(...);   // packed {-1, 0, +1, 0} × 16

// Expand packed ternary bytes → 64 decoded int8 values in one instruction
__m512i decoded = _mm512_permutex2var_epi8(lut_lo, packed_weights, lut_hi);
```

`vpermi2b` performs a 64-way byte-granularity table lookup across two 512-bit registers. It decodes 64 ternary weights (packed in 32 bytes) to 64 signed bytes in a single instruction, with latency of 3 cycles on Sapphire Rapids.

### Step 2: `vpternlogd` — 3-input bitwise

```c
// Compute sign mask from decoded weights (negative weights → flip sign of activation)
__m512i sign_mask = _mm512_ternarylogic_epi32(decoded, zero_mask, ones, 0x?);
```

`vpternlogd` is a 3-input bitwise operation that can express any boolean function of three inputs in a single instruction (the 8-bit immediate encodes the truth table). Used here to compute the sign-flip mask and zero mask from the decoded ternary values without branching.

### Step 3: `vpdpbusds` — INT8 VNNI accumulation

```c
// Accumulate: add (decoded_weights × activations) into int32 accumulator
// vpdpbusds: dst += (a_unsigned × b_signed) horizontally in groups of 4
acc = _mm512_dpbusd_epi32(acc, decoded_unsigned, activations_i8);
```

`vpdpbusds` (or the non-saturating `vpdpbusd`) performs 16 parallel 4-wide dot products between unsigned and signed int8 pairs per cycle on a 512-bit vector — 64 int8 MACs per instruction, accumulated into 16 int32 lanes. The signed×unsigned asymmetry requires the `w_enc = w + 1` bias trick: weights are stored as {0, 1, 2} rather than {−1, 0, +1}, and the dot product result is corrected by subtracting `sum(activations)`.

### Why this is faster than FP32 FMA

On the Xeon (Sapphire Rapids, 4 cores):
- FP32 FMA throughput: 2 `vfmadd` per cycle × 16 floats × 4 cores = 128 FP32 MACs/cycle
- INT8 VNNI throughput: 2 `vpdpbusd` per cycle × 64 int8 MACs × 4 cores = 512 INT8 MACs/cycle

The memory bandwidth story is the bigger win: 512 bytes of packed ternary weights vs 8192 bytes of FP32 weights for the same computation. At 36 tok/s we are consuming ~11.7 GB/s of DRAM bandwidth — approximately 95% of the measured ceiling on that hardware.

---

## 3. The Thread Pool

**File:** `src/threading.c`

The thread pool uses C11 atomics with a spin-then-sleep pattern to avoid futex syscall overhead on hot dispatch cycles.

```c
// Worker spin loop (simplified)
while (running) {
    // Spin for ~500ns checking for work
    for (int i = 0; i < SPIN_COUNT; i++) {
        if (atomic_load_explicit(&queue->head, memory_order_acquire) != tail) {
            // got work — process without any syscall
            break;
        }
        _mm_pause();  // PAUSE instruction: reduces power + avoids memory order machine clears
    }
    // Fall through to futex sleep only if no work arrived during spin window
    futex_wait(&queue->head, tail);
}
```

The key insight: for short generation steps (5–50ms per token), the thread scheduling latency from a naive `pthread_cond_signal` / `futex_wake` is measurable as a fraction of token generation time. Spinning for a brief window before sleeping captures the common case (next token dispatch arrives within microseconds) with no syscall.

Tested on Tiger Lake: spin-then-sleep reduces per-dispatch overhead from ~15μs (naive futex) to ~2μs at 8 threads with minimal CPU idle penalty.

---

## 4. KV Cache Layout

**File:** `src/kv_cache.c`

Layout: `[layer][head][position][head_dim]`

This is the opposite of the "natural" transformer layout `[position][layer][head][dim]`. The reason: during attention, all positions for a single head in a single layer are read together. Contiguous layout along the `[position]` dimension maximizes cache line reuse during the O(T²) attention step, especially for long contexts.

With FP16 KV cache at 2048 head_dim, 64 heads, 32 layers, 4096 context:
- Sequential read per attention: 4096 × 128 bytes = 512 KB per layer (fits in 4MB L3)
- Scattered layout would cause 4096 cache line loads (64B each) per head instead of ~512 sequential cache lines

---

## 5. The MoE Expert Scatter Problem (HELP WANTED)

**Status:** 13× slower than llama.cpp. Root cause identified. Fix planned but not implemented.

**Model:** DeepSeek-V2-Lite-Chat-Q4_K_S — 26 MoE layers, 64 experts per layer, 6 experts activated per token.

### What's happening

Each MoE layer selects 6 of 64 experts based on router logits. In a GGUF file, expert weights are stored in expert-index order: expert_0 weights, expert_1 weights, ..., expert_63 weights. For any given token, the 6 activated experts are scattered at non-contiguous DRAM offsets with gaps of ~140MB between each.

```
Expert 0: offset 0x0000_0000 (unused)
Expert 1: offset 0x0880_0000 (unused)
Expert 7: offset 0x3980_0000 ← activated
Expert 23: offset 0xBB80_0000 ← activated
Expert 31: offset 0xF980_0000 ← activated
... etc.
```

Intel's hardware prefetcher tracks approximately 8–10 independent stream prefetches. Accessing 6 experts across 26 MoE layers means requesting ~156 non-contiguous streams simultaneously — the prefetcher gives up and DRAM latency dominates.

### Profiling data (perf stat, same Xeon as BitNet benchmark)

| Metric | Project Zero | llama.cpp |
|---|---|---|
| IPC | 0.94 | 1.61–1.85 |
| L3 miss rate | 86% | ~52% |
| Page faults per run | ~2,000,000 | ~645 |
| Effective DRAM bandwidth | 2–3 GB/s | ~9 GB/s |
| tok/s | ~1.0 | 13.79 |

### The fix: expert weight repacking at load time

At model load, sort and interleave expert weights so that any set of top-K activated experts for a typical token selection pattern are physically adjacent in memory. llama.cpp implements this in `llama_model_load_internal()`.

The challenge with Q4_K: each "superblock" in Q4_K quantization contains 256 weights with a shared scale+offset. Repacking must preserve superblock boundaries or recompute scales. See `ggml_type_size(GGML_TYPE_Q4_K)` and the `block_q4_K` struct in ggml-quants.h for the layout.

**Discussion with more context:** https://github.com/shifulegend/project-zero/discussions/1

If you've worked on ggml Q4_K layout or llama.cpp's expert repacking, your input there would be extremely valuable.

---

## 6. SIMD Dispatch Table

**File:** `src/math/cpu_features.c`, `src/math/dispatch.c`

Runtime detection at startup via CPUID (x86) and `getauxval(AT_HWCAP)` (ARM). Checked once and cached. The dispatch table selects the appropriate kernel at init time:

| Tier | Instruction set | Hardware | MACs/cycle (512b) |
|---|---|---|---|
| AVX-512 VBMI + VNNI | vpermi2b + vpdpbusd | SPR, ICX, TGL, ZEN4 | 64 |
| AVX-512 VNNI (no VBMI) | unpack + vpdpbusd | CLX, ZEN4 without VBMI | 64 (slower decode) |
| AVX-VNNI (256-bit) | vpdpbusd (YMM) | ADL, RPL, ZEN3 | 32 |
| AVX2 (no VNNI) | vpmaddubsw chain | Haswell through Skylake | ~16 |
| ARM NEON+dotprod | vdotq_s32 | Apple M1–M4, A14+, Snapdragon 8 | 16 |
| Scalar fallback | standard C | Any | 1 |

The VBMI path is only ~15% faster than the VNNI-only path on identical hardware; the bulk of the speedup vs bitnet.cpp comes from the VNNI accumulation itself plus the memory layout, not specifically VBMI.

---

## 7. Auto-Calibration

**File:** `src/calibration.c`

On first run, the engine probes:
1. **DRAM bandwidth** — writes and reads a ~256MB buffer and measures wall-clock time
2. **L3 cache size** — binary search for the buffer size where throughput drops significantly
3. **Physical core count** — reads `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` (Linux) or `sysctl -n hw.physicalcpu` (macOS) to distinguish physical from hyperthreaded logical cores

Results are used to:
- Set `--threads` default (physical cores only — HT doubles thread count but halves per-thread AVX-512 frequency on many Intel CPUs)
- Choose classifier quantization level (INT4 if bandwidth-constrained, BF16 if compute-constrained)
- Set KV cache strategy thresholds

Results are cached to `~/.project_zero_calibration` and reused on subsequent runs unless `--recalibrate` is passed.

---

## Appendix: Build Targets

```bash
make release    # -O3 -march=native -flto + PGO (recommended)
make debug      # -O0 -g -fsanitize=address,undefined
make pgo        # Profile-Guided Optimization two-pass build (best performance)
make test       # Run full test suite (required before any PR)
```

PGO adds ~8–12% throughput on top of `release` on the Xeon benchmark due to branch prediction improvements in the token sampling and rope rotation paths.
