# BitNet.cpp Implementation Guide for Project Zero

## Overview
This document provides implementation-ready details for closing the 2.5x instruction gap between Project Zero (2.27B instr/token) and BitNet.cpp (0.90B instr/token).

**Target:** Achieve 24.67+ tok/s on i5-11300H (same as BitNet.cpp)

---

## Issue 1: Per-Token Activation Quantization (Biggest Bottleneck)

### Current Problem (Project Zero)

Project Zero calls `quantize_row_to_i8()` once per worker thread per token:

```c
// Per-worker quantization (EXPENSIVE, called every token)
void quantize_row_to_i8(const float * x, int8_t * y, int n) {
    float max = 0;
    // Find max absolute value
    for (int i = 0; i < n; i++) {
        if (fabs(x[i]) > max) max = fabs(x[i]);  // ~n comparisons + branches
    }
    
    float scale = 127.0f / max;
    
    // Quantize all elements
    for (int i = 0; i < n; i++) {
        y[i] = (int8_t)roundf(x[i] * scale);  // ~n multiplies + rounds + stores
    }
}

// Called from: per_worker_compute_token()
for (int layer = 0; layer < num_layers; layer++) {
    // Activation quantization per thread (SYNCHRONIZED BARRIER)
    quantize_row_to_i8(hidden_state, hidden_state_i8, hidden_dim);  // ~200-300M instr
    
    // MatVec: weight × quantized_activation
    // ...
}
```

**Cost Breakdown per token:**
- 32-dim loop (max reduction): ~50 comparisons + branches
- × num_layers (24 for 2B model): 1200 ops
- × num_threads (8 threads): 9600 ops synchronously
- **Total: 150-300M instructions (15-30% of per-token budget)**

**DRAM Cost:** Full activation vector touched (2B float = 8GB of memory bandwidth)

### BitNet.cpp Solution

Replace with **single-pass SIMD max reduction**:

```c
// Global quantization scale (FAST, called once per token, all threads wait here)
inline float compute_quant_scale_simd(const float* x, int n) {
    __m256 max_vec = _mm256_set1_ps(0.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    
    // Vectorized max reduction (n/8 iterations instead of n)
    for (int i = 0; i < n; i += 8) {
        __m256 vec_x = _mm256_loadu_ps(x + i);
        __m256 vec_abs = _mm256_andnot_ps(sign_mask, vec_x);  // |x|
        max_vec = _mm256_max_ps(max_vec, vec_abs);
    }
    
    // Horizontal max (tree reduction)
    __m128 max_lo = _mm256_castps256_ps128(max_vec);
    __m128 max_hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 max_128 = _mm_max_ps(max_lo, max_hi);
    
    max_128 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));  // [max, max, -, -]
    max_128 = _mm_max_ss(max_128, _mm_movehdup_ps(max_128));         // [max, -, -, -]
    
    float max_val = _mm_cvtss_f32(max_128);
    return 127.0f / max_val;  // Single scale value
}

// This gets called ONCE per token (not per layer or thread!)
float quant_scale = compute_quant_scale_simd(hidden_state, hidden_dim);

// Then use this scale everywhere (no extra quantization needed)
for (int layer = 0; layer < num_layers; layer++) {
    // No barrier here! Scale is already computed
    
    // MatVec with pre-computed scale
    // If using LUT: LUT entries are built using this scale once
    // ...
}
```

**Cost:** ~50M instructions instead of 150-300M
**Savings:** 100-250M instr/token (10-25% improvement)

**Implementation in Project Zero:**

1. Move `quantize_row_to_i8` call to **token inference start** (not per-layer)
2. Store result as **single float** (not vector)
3. Pass scale to all worker threads via shared memory (no lock needed)

```c
// In main inference loop
struct TokenComputeContext {
    float* activations;     // Full activation buffer
    float quant_scale;      // Global scale (computed once)
    int8_t* quantized;      // Pre-allocated
    int hidden_dim;
};

// Before worker threads start
ctx.quant_scale = compute_quant_scale_simd(ctx.activations, ctx.hidden_dim);

// Each worker thread can then use ctx.quant_scale (read-only, no synchronization)
```

---

## Issue 2: Weight Unpacking in Hot Loop

### Current Problem (Project Zero)

PZ unpacks 2-bit weights **inside the accumulation loop**:

```c
// From ggml_vec_dot_i2_i8_s_1x1()
for (int row = 0; row < nrc; row++) {
    __m256i accu = _mm256_setzero_si256();
    
    const uint8_t * x_row = x + row * bx / 4;
    
    // THIS IS INSIDE THE LOOP (expensive!)
    for (int i = 0; i < group32_num; i++) {
        const uint8_t *px = x_row + i * 1024;      // Strided weight access
        const int8_t  *py = y + i * 4096;          // Activation access
        __m256i accu32 = _mm256_setzero_si256();
        
        for (int j = 0; j < 32; j++) {
            // Unpack 4×2bit weights from single 32B register
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);    // Shift
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);    // Shift
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);    // Shift
            
            xq8_3 = _mm256_and_si256(xq8_3, mask);          // Mask
            xq8_2 = _mm256_and_si256(xq8_2, mask);          // Mask
            xq8_1 = _mm256_and_si256(xq8_1, mask);          // Mask
            xq8_0 = _mm256_and_si256(xq8_0, mask);          // Mask
            
            // Now compute dot products
            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));
            
            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);  // COMPUTE
            // ... more computation
            
            px += 32;
            py += 128;
        }
    }
}

// Problem: 4 shifts + 4 masks = 8 instructions per 64 weights
// With 2B weights per token: 8 × (2B/64) = 250M instr just for unpacking!
```

### BitNet.cpp Solution: Precomputed LUT

BitNet.cpp **removes weight unpacking from hot loop** using precomputed Look-Up Tables:

```c
// INITIALIZATION (not in hot loop)
inline void build_lut_for_ternary_weights(
    int8_t* lut_output,           // 256 bytes per chunk
    const float* quantized_acts,  // K activations
    float quant_scale,
    int k_chunks                   // K / 24
) {
    for (int chunk = 0; chunk < k_chunks; chunk++) {
        // Gather 3 activation values (at 24-byte spacing for layout)
        __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
        __m256 vec_b0 = _mm256_i32gather_ps(quantized_acts + chunk*24 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(quantized_acts + chunk*24 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(quantized_acts + chunk*24 + 2, vec_bi, 1);
        
        // Convert to INT8 with pre-computed scale
        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(vec_b0, _mm256_set1_ps(quant_scale)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(vec_b1, _mm256_set1_ps(quant_scale)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(
            _mm256_mul_ps(vec_b2, _mm256_set1_ps(quant_scale)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        
        // Precompute all 16 LUT entries for {-1,0,+1}^3 combinations
        // For each ternary weight triplet (w0, w1, w2) ∈ {-1,0,1}^3:
        // entry[i] = w0*b0i + w1*b1i + w2*b2i (all combinations)
        
        __m256i lut_entries[16];
        lut_entries[0]  = _mm256_setzero_si256();                     // 0,0,0
        lut_entries[1]  = vec_b2i;                                     // 0,0,1
        lut_entries[2]  = vec_b1i;                                     // 0,1,0
        lut_entries[3]  = _mm256_add_epi32(vec_b1i, vec_b2i);         // 0,1,1
        lut_entries[4]  = _mm256_sub_epi32(vec_b1i, vec_b2i);         // 0,1,-1
        lut_entries[5]  = vec_b0i;                                     // 1,0,0
        lut_entries[6]  = _mm256_add_epi32(vec_b0i, vec_b2i);         // 1,0,1
        lut_entries[7]  = _mm256_add_epi32(vec_b0i, vec_b1i);         // 1,1,0
        lut_entries[8]  = _mm256_add_epi32(_mm256_add_epi32(vec_b0i, vec_b1i), vec_b2i);  // 1,1,1
        // ... (8 more entries for negative weight combinations)
        
        // Store to output (256 bytes = 8 entries × 32 bytes)
        int8_t* out = lut_output + chunk * 256;
        for (int j = 0; j < 16; j++) {
            _mm256_storeu_si256((__m256i*)(out + j*16), 
                                _mm256_packs_epi32(lut_entries[j], lut_entries[j]));
        }
    }
}

// IN HOT LOOP (now just table lookup!)
inline void gemm_with_lut(
    int32_t* c,                    // Output
    const int8_t* lut,             // Precomputed LUT (256 bytes per chunk)
    const uint8_t* packed_weights, // 4-bit ternary indices
    const uint8_t* weight_signs    // Sign bits
) {
    for (int batch = 0; batch < batch_size; batch++) {
        __m256i vec_c = _mm256_setzero_si256();
        
        for (int k = 0; k < K / 24; k++) {
            // Load 4-bit ternary indices (packed)
            __m256i indices = _mm256_loadu_si256((const __m256i*)(packed_weights + k*32));
            
            // Load 2-bit sign information
            __m256i signs = _mm256_loadu_si256((const __m256i*)(weight_signs + k*16));
            
            // Extract top and bottom nibbles (4 bits each = 2 weight indices)
            __m256i idx_top = _mm256_and_si256(_mm256_srli_epi16(indices, 4), mask);
            __m256i idx_bot = _mm256_and_si256(indices, mask);
            
            // Shuffle lookup from LUT using PSHUFB
            // PSHUFB is FAST (~1 cycle) because LUT entries fit in L1
            __m128i lut_chunk = _mm_loadu_si128((const __m128i*)(lut + k*256 + batch*128));
            
            __m256i lookup_top = _mm256_shuffle_epi8(
                _mm256_set_m128i(lut_chunk, lut_chunk), idx_top);
            __m256i lookup_bot = _mm256_shuffle_epi8(
                _mm256_set_m128i(lut_chunk, lut_chunk), idx_bot);
            
            // Correct for signs (2's complement trick)
            __m256i sign_bits = _mm256_srai_epi16(
                _mm256_slli_epi16(signs, 1), 15);  // Expand sign to full word
            lookup_top = _mm256_xor_si256(lookup_top, sign_bits);  // Conditional negate
            
            // Accumulate
            vec_c = _mm256_add_epi32(vec_c, lookup_top);
            vec_c = _mm256_add_epi32(vec_c, lookup_bot);
        }
        
        c[batch] = hsum_epi32(vec_c);
    }
}

// Result: Weight unpacking is AMORTIZED over token time
// - LUT build: ~100M instr (done once per token, all threads can proceed)
// - GEMM hot loop: ~0 instr for unpacking (only PSHUFB)
// - Total savings: 200-250M instr/token
```

**Key Optimization Insight:**
- Weight unpacking is **data-independent** (doesn't depend on activation values)
- Can be **decoupled** from computation
- LUT entries are **small** (256 bytes per K/24 activations = fits in L1)
- PSHUFB lookup is **1 cycle** vs shift+mask sequence (~3-4 cycles)

### Integration into Project Zero

1. **Create LUT preparation step** (called once per token):
```c
struct LUTContext {
    int8_t* lut_table;           // 256 bytes × (K/24)
    float quant_scale;
};

// Call before worker threads
build_lut_for_ternary_weights(
    ctx.lut_table,
    hidden_state,
    ctx.quant_scale,
    hidden_dim / 24
);
```

2. **Update GEMM kernel** to use LUT:
```c
// Replace weight unpacking loop with:
__m128i lut_entry = _mm_loadu_si128((const __m128i*)(
    ctx.lut_table + weight_chunk_idx * 256));

__m256i lookup = _mm256_shuffle_epi8(
    _mm256_set_m128i(lut_entry, lut_entry),
    ternary_indices);

// Result directly used in accumulation
```

---

## Issue 3: Single-Threaded Execution of Quantization

### Current Problem
PZ's architecture likely has **implicit barrier** after quantization phase:
```
Token inference:
  [BARRIER] Compute quant_scale (sequential)
  [BARRIER] Quantize to int8 (per-thread)
  [BARRIER] GEMM (parallel)
  [BARRIER] Next layer
```

Each barrier adds overhead for thread synchronization.

### BitNet.cpp Solution
```
Token inference:
  Compute quant_scale (fast SIMD, no barrier needed)
  Build LUT (can be done by master thread while workers idle)
  Workers start GEMM immediately (uses shared LUT)
  [Single implicit barrier at token boundary]
```

**Savings:** 10-20M instr/token from reduced synchronization overhead

---

## Integration Checklist for Project Zero

### Phase 1: LUT Activation Quantization (Est. 10-15% speedup)
- [ ] Replace multi-pass `quantize_row_to_i8` with SIMD max reduction
- [ ] Store result as single float (not vector)
- [ ] Pass scale to worker threads via context struct
- [ ] Test correctness with current weight unpacking

### Phase 2: Weight Unpacking LUT (Est. 15-25% additional speedup)  
- [ ] Implement `build_lut_for_ternary_weights()` function
- [ ] Update GEMM kernel to use PSHUFB lookups
- [ ] Pre-allocate LUT buffer (256 bytes × K/24)
- [ ] Benchmark: measure cache hit rate on LUT

### Phase 3: Activation Parallelism (Est. 10-20% additional speedup)
- [ ] Modify GEMM to process multiple output rows simultaneously
- [ ] Reuse weight unpacking (or LUT) across rows
- [ ] Tune ROW_BLOCK_SIZE and PARALLEL_SIZE for i5-11300H

### Phase 4: Embedding Quantization (Est. 5-10% additional speedup)
- [ ] Use llama.cpp's Q6_K format for token embeddings
- [ ] Reduce wcls memory footprint by ~1.7x
- [ ] Verify quality with perplexity testing

---

## Performance Measurement

Use Intel VTune or `perf` to measure:

```bash
# Measure instructions per token
perf stat -e cycles,instructions,cache-references,cache-misses \
  ./project_zero_benchmark --model bitnet-b158-2b.gguf \
                            --tokens 128 \
                            --threads 4

# Should see improvement in instructions metric
# Current: ~2.27B instr/token
# Target: < 1.0B instr/token
```

Expected progression:
- Phase 1: 1.80-2.00B instr/token (20-25% reduction)
- Phase 2: 1.10-1.30B instr/token (additional 40-50% reduction)
- Phase 3: 0.95-1.10B instr/token (additional 10-15% reduction)
- Phase 4: 0.90-1.00B instr/token (approaching BitNet.cpp)

---

## Additional Resources

See accompanying documents:
- `BITNET_ANALYSIS.md` - Detailed technical breakdown
- BitNet.cpp Repository: https://github.com/microsoft/BitNet

