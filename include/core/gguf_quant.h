/**
 * gguf_quant.h — Phase 34.2b
 *
 * Dequantization for GGUF k-quant and legacy quant formats.
 * Each function expands `n_elems` quantised values from `data` into float32 `out`.
 *
 * Supported formats:
 *   Q8_0  — 8-bit signed int, block size 32, one FP16 scale per block.
 *   Q4_K  — 4-bit k-quant, super-block 256, 8 sub-blocks each with 6-bit scale + min.
 *   Q4_0  — legacy 4-bit, block size 32, one FP16 scale per block (no min offset).
 *
 * For standard float models (DeepSeek, Llama, etc.) the architecture doc mandates
 * proper PTQ (Q4_K / Q8_0) rather than ternary packing.
 * See CPU_LLM_TERNARY_ENGINE.md §"Category 3: Standard Models".
 */
#pragma once
#include <stddef.h>

void gguf_dequant_q8_0(float *out, const void *data, size_t n_elems);
void gguf_dequant_q4_k(float *out, const void *data, size_t n_elems);
void gguf_dequant_q4_0(float *out, const void *data, size_t n_elems);
void gguf_dequant_q5_0(float *out, const void *data, size_t n_elems);
void gguf_dequant_q5_1(float *out, const void *data, size_t n_elems);
void gguf_dequant_q5_k(float *out, const void *data, size_t n_elems);
void gguf_dequant_q6_k(float *out, const void *data, size_t n_elems);
void gguf_dequant_q2_k(float *out, const void *data, size_t n_elems);
void gguf_dequant_q3_k(float *out, const void *data, size_t n_elems);
void gguf_dequant_iq4_nl(float *out, const void *data, size_t n_elems);
void gguf_dequant_q2_0(float *out, const void *data, size_t n_elems);
