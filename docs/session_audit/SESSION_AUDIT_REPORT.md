# Session Audit Report: BitNet 2B Architectural Parity & Reconversion
**Date:** 2026-03-14
**Status:** Unfinished Business - Critical Debugging Session
**Branch:** `session-archive-bitnet-reconversion-fix`

## Executive Summary
This session focused on achieving architectural parity for the BitNet b1.58 2B-4T model within the Project Zero engine. We identified and fixed several "smoking gun" bugs that caused the model to output garbage. While 1:1 parity of weights and layout has been achieved, the model's output remains incoherent, suggesting a mismatch in activation scaling or Normalization (SubLN) parameters.

## Critical Discoveries & Fixes

### 1. The Weight Packing "Transposition" Bug
- **Issue:** Microsoft's official BitNet checkpoints store ternary weights in a `(rows/4, cols)` uint8 format (re-packing rows). Our engine expects a flat row-major packing `(rows * cols / 4)`.
- **Impact:** Before the fix, the engine was reading the weight matrix essentially transposed and interleaved, resulting in pure static/hallucinations.
- **Fix:** Implemented `unpack_ms_packed` in `tools/convert_hf_bitnet.py` to correctly interpret Microsoft's format, rebuild the ternary matrix, and repack it in the engine's flat format.
- **Verification:** Diagnostic script `check_norms.py` confirmed 1:1 bit-parity between our repacked weights and the original HF weights.

### 2. Missing BitNet Sub-Norms (SubLN)
- **Issue:** BitNet 2B uses intermediate LayerNorms (`attn_sub_norm` and `ffn_sub_norm`) between projection stages. These were missing in both the converter and the engine.
- **Impact:** Binary file layout was misaligned (offsets were wrong for all subsequent layers), and activation magnitudes were likely exploding without mid-block normalization.
- **Fix:** 
  - Updated `TransformerWeights` struct and `weights_map` to handle `rms_attn_sub_norm` and `rms_ffn_sub_norm`.
  - Updated `attention_forward` and `ffn_forward` to apply these norms.
  - Updated `convert_hf_bitnet.py` to write these tensors into the binary.

### 3. BPE String Cleanup & Tokenizer EOS
- **Issue:** Llama 3 / BitNet tokenizers use special characters (`Ġ` for space, `Ċ` for newline). Also, the EOS token ID is `128001` (Llama 3) rather than `2` (Llama 2).
- **Fix:** 
  - Added `clean_bpe_string` to `src/tokenizer/tokenizer_decode.c`.
  - Updated `generate.c` to recognize EOS token `128001`.

## Current Blocker
Even with 1:1 weight parity, the model still outputs word-like garbage (e.g., `inta-dropÃ¤tuin Lovexpect`).
**Hypothesis:** The `attn_sub_norm` and `ffn_sub_norm` application in the C engine might needs specific scaling or different epsilon values compared to standard RMSNorm. Alternatively, the activation `ReLU2` implementation needs verification against the official BitNet-specific scaling factors.

## Next Steps for Developer
1. **Verify Activations:** Run a tiny prompt through the HF model in Python and the C engine, comparing intermediate activations after `attn_sub_norm`.
2. **SubLN Parameters:** Verify if `attn_sub_norm` requires a bias or different epsilon (BitNet paper mentions SubLN specificities).
3. **RAM Optimization:** The current performance is disk-starved (0.2 tok/s). Free up at least 2.5GB RAM for proper `mmap` caching.

---
*End of Audit Report*
