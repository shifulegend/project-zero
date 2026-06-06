# Phase 12 CLI Implementation & Front-End Testing Plan

This plan scopes the work to satisfy Activities A through G, focusing on bringing the Project Zero engine to life via an interactive CLI and validating it against local HuggingFace models.

## User Review Required
> [!IMPORTANT]
> Please review the approach regarding Activity D (locally downloaded Windows models). I am planning to convert the available HuggingFace models (`Llama-3.1-8B`, `Qwen2.5-7B`, `bitnet-b1.58-2B-4T`) in `/mnt/windows/huggingface_models` into binary format utilizing the `tools/convert_hf_bitnet.py` and `tools/pack_ternary.py` scripts before running the engine's frontend against them. Let me know if you would like me to use a specific model first, or adjust the conversion parameters.

## Proposed Changes

### 1. Phase 12: CLI & Main Entry Point (Activity A)
Implementation of the C-based frontend interfaces as specified in the Phase 12 architecture.

#### [NEW] `include/cli/args.h` & `src/cli/args.c`
- Implements the argument parser for `CliArgs` (model, tokenizer, prompt, image inputs, temperature, top-p, max-tokens, threads, seed).

#### [NEW] `include/cli/timer.h` & `src/cli/timer.c`
- Provides the `timer_now_us()` execution metric functions for POSIX metrics (tokens/sec).

#### [NEW] `include/cli/repl.h` & `src/cli/repl.c`
- Defines the `run_repl` infinite generation loop, capturing stdin, generation, and multi-line interactions. Includes reasoning mode (`/think`) and multimodal integrations.

#### [MODIFY] `src/cli/main.c`
- Rewrites the placeholder `main.c` file to establish the full boot sequence (parsing args, probing hardware, initializing `ThreadPool`, Memory Mapping weights, `Config` load, KV cache strategy determination, Tokenizer setup, and either single-shot or REPL).

### 2. Documentation & Git Management (Activity B, F, G)
#### [NEW] `WALKTHROUGH_PHASE12.md`
- Provide detailed architectural explanation of the boot mechanism, argument management, and the CLI execution lifecycle.
#### [NEW] `TEST_RESULTS_FRONTEND.md`
- Documentation for test results required by Activity E.
#### GitHub Best Practices
- Execution of `git` routines to ensure clean, chronological rebasing and committing.

### 3. [Phase 12.1] Performance & Correctness (BitNet Fixes)
Addressing the 0.2 tok/s bottleneck and the embedding quantization bug.

#### [MODIFY] `tools/convert_hf_bitnet.py`
- Corrects token embedding and norm layer handling to use `float32` instead of bitpacked ternary. This fixes the format mismatch with the C engine.

#### [MODIFY] `src/math/ternary_matmul_packed_avx2.c`
- Implements an unrolled bit-manipulation loop for `unpack8_to_epi32`.

#### [MODIFY] `src/core/weights.c`
- Refines mapping logic for float embeddings and introduces `w->wcls_is_ternary` flag.

### 4. [Phase 12.2] High-Precision Matmul Fallback
- **Reasoning**: BitNet 2B ties high-precision embeddings to the classifier. Standard models (Qwen) use high-precision for all layers.
#### [NEW] `parallel_matmul_float32` in `math/parallel_matmul.c`
- Implements multi-threaded float32 matrix-vector multiplication.
#### [MODIFY] `src/transformer/forward.c`
- Updates `transformer_forward` to dynamically choose between `parallel_ternary_matmul_packed` and `parallel_matmul_float32` based on the weight flag.

### 5. [Phase 12.3] Qwen Support (Activity H)
#### [NEW] `tools/convert_hf.py`
- General purpose converter for non-BitNet models (Llama/Qwen).

## Verification Plan

### Automated Tests (Activity C)
- **C Test Suite:** Run `make` to compile the `project-zero` binary and the forensic audit suite. Execute `tests/audit_suite` / `tests/test_runner` to validate no regressions in the core layers.
- **Python Integration Test:** Run `python tests/monkey_tester.py` to bombard the CLI frontend with randomized fuzzy CLI inputs, ensuring the argument parser handles edge-cases without segmentation faults.

### Local Windows Models Testing (Activity D)
- **Conversion Phase:** 
  Run `python tools/convert_hf_bitnet.py` and `tools/pack_ternary.py` on the models inside `/mnt/windows/huggingface_models/`.
- **Execution Run:**
  Execute frontend prompts via the newly created `project-zero` binary against the converted binaries, capturing output logs.
- Document inferences into `TEST_RESULTS_FRONTEND.md` (Activity E).
