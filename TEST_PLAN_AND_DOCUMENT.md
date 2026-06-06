# End-to-End Test Plan and Test Document
Project: CPU LLM Ternary Engine (Project-Zero)

## 1. Introduction
This document outlines the comprehensive testing strategy for the adaptive AI engine, covering Blackbox (Functional/Integration), Redbox (Security/Adversarial/Boundary), and Audit (Deep Vulnerability Probing) testing. The project implements a ternary-weight custom Transformer with SIMD-accelerated math, a custom thread pool, BPE tokenizer, sampling strategies, and KV cache optimizations.

---

## 2. Redbox Test Plan (Security, Boundary, and Subsystem Integrity)

### 2.1 Memory & SIMD Alignment Subsystem
- **RB-MEM-01 (SIMD Alignment Attacks):** Force odd-size allocations to test `tn_aligned_alloc` 64-byte alignment enforcement.
- **RB-MEM-02 (OOM Resistance):** Request INT_MAX sequence length to force OOM. Must return `TN_ERR_OOM` gracefully.
- **RB-MEM-03 (Integer Overflow):** Test `tn_size_mul_overflow` and `tn_size_mul4` with `SIZE_MAX` inputs.

### 2.2 Thread Pool Concurrency & Deadlocks
- **RB-THR-01 (Rapid Dispatch):** 1000 rapid-fire dispatches testing FM-006/FM-001 regression.
- **RB-THR-02 (Epoch Integrity):** Verify dispatch_epoch prevents cross-dispatch task stealing.

### 2.3 Input/Weights Mmap Corruption
- **RB-IO-01 (Invalid Config):** Inject corrupted Config with zero dims, negative layers.

---

## 3. Blackbox Test Plan (Functional & Integration)

### 3.1 Tokenizer Subsystem
- **BB-TOK-01 (Encode):** BPE encoding with merges, cascading merges, partial merges, empty input.
- **BB-TOK-02 (Decode):** Decode basic tokens, raw bytes, after BOS, out of range.

### 3.2 Transformer Layers
- **BB-TRF-01 (Scale & Normalization):** RMSNorm, softmax, SIMD activation correctness.
- **BB-TRF-02 (MHA + GQA):** Multi-head attention with grouped query attention mapping.
- **BB-TRF-03 (KV Cache Growth):** Multi-position caching and context building.

### 3.3 End-to-End Forward Pass
- **BB-FWD-01 (Single Token):** Single token forward pass logit verification.
- **BB-FWD-02 (Multi-token Generation):** 10-token cascaded generation with KV cache verification.

### 3.4 Sampling & Generation (Phase 7)
- **BB-SAM-01 (RNG):** Xorshift64* range [0,1), reproducibility, different seeds.
- **BB-SAM-02 (Argmax):** Greedy decoding correctness across edge cases.
- **BB-SAM-03 (Temperature):** Scaling verification, identity, argmax preservation.
- **BB-SAM-04 (Top-p):** Nucleus sampling with dominant peak and uniform distribution.
- **BB-SAM-05 (Top-k):** Candidate restriction, clamping, deterministic k=1.

### 3.5 KV Cache Optimizations (Phase 8)
- **BB-KV-01 (int8 Quantization):** Roundtrip accuracy, zero vectors, sign preservation.
- **BB-KV-02 (int4 Quantization):** Packed nibble roundtrip, odd dimensions.
- **BB-KV-03 (Sliding Window):** Pinned system prompt, circular mapping, wraparound.
- **BB-KV-04 (Strategy Selection):** RAM-based tier selection, short seq_len clamping.

---

## 4. Audit Test Plan (Deep Vulnerability Probing)

### 4.1 KV Cache Memory Safety
- **AUD-MEM-01 (KV Cache Overflow):** Call `transformer_forward()` with `pos > max_seq_len` to test bounds enforcement. Regression test for FM-007.

### 4.2 Thread Pool Stress
- **AUD-THR-01 (Rapid-Fire Stress):** 1000 dispatches with 8 threads, verify exact counter via atomic increment. Regression test for FM-001/FM-006.

---

## 5. Test Environment Specification

* **Compiler:** GCC with `-std=c99 -Wall -Wextra -Wpedantic -fsanitize=address,undefined`
* **OS:** Ubuntu Linux x86_64
* **SIMD:** AVX2 baseline with scalar fallback
* **Build:** Debug mode via `make test` (Makefile auto-discovers sources)

---

## 6. Testing Document: Phases & Coverage
All implementation phases through **Phase 8** have accompanying C unit tests using `<test_harness.h>`. Testing runs via `make test` which compiles all test binaries with ASAN+UBSAN and executes sequentially.

**Current coverage:** 16 test suites, 3652 assertions, 100% pass rate.
