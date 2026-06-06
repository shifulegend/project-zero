# Task: BitNet 2B-4T Model Debugging - FINAL REPORT

## Accomplishments
1. **Weight Conversion Unblocked**: Overhauled the ternary weight un-packing to support Microsoft's `uint8` packed format. Corrected the values from `{0, 1, 2}` to `{-1, 0, 1}`.
2. **Architecture Parity**: Integrated `attn_sub_norm` and `ffn_sub_norm` layers into the C engine forward pass, matching BitNet's 1.58b specification.
3. **Generation Robustness**: Injected the BOS token and fixed the EOS token check for Llama-3 style vocabularies.
4. **Decoder Fix**: Cleaned the tokenizer output to remove BPE control character artifacts.

## Current status
The engine now correctly maps and executes the BitNet model. Output has progressed from total garbage to identifiable words, but coherence is not yet achieved.

## Remaining Blockers
- **Weight Scale Uncertainty**: There is a discrepancy in how Microsoft's `weight_scale` scalar should be applied (Direct vs Inverse). Both have been tested; direct scale produces better (word-like) results but is still incoherent.
- **Activation Drift**: BitNet was trained with 8-bit activation quantization. Running in full FP32 may prevent the model from correctly "triggering" its ternary weights.
- **RoPE Style**: The model predates Llama-3 but uses a Llama-3 sized vocabulary, creating ambiguity in the RoPE pair style (interleaved vs rotate_half).

## Recommendations
- **Simulate Activation Quantization**: Modify `ffn.c` and `attention.c` to truncate activations to 8-bit before ternary matmuls.
- **Verify Weight Ordering**: Perform an element-by-element cross-check with a Python reference for the O-projection and Down-projection layers.
- **Optimize RAM**: The machine is extremely RAM-constrained (800MB free), which limits the ability to use large KV caches or keep the model fully in disk cache.
