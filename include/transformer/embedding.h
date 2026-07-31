#ifndef TN_EMBEDDING_H
#define TN_EMBEDDING_H

#include "core/platform.h"

/**
 * Token embedding lookup: convert one row to float32.
 *
 * Two paths (selected automatically by caller):
 *   F32 path: direct memcpy from float table (GGUF quantized models) — matches
 *             llama.cpp's ggml_get_rows single-conversion Q4_K → F32.
 *   BF16 path: bit-shift BF16 → F32 (legacy .bin format, BF16/F16 GGUF).
 *
 * @param out              Output float vector of size dim (pre-allocated)
 * @param token            Token ID (0-based index into the embedding table)
 * @param embd_f32         F32 embedding table (may be NULL → use bf16 path)
 * @param embedding_table  BF16 embedding table (used when embd_f32 is NULL)
 * @param dim              Embedding dimension
 */
void embed_token(float *out, int token,
                 const float *embd_f32,
                 const tn_u16 *embedding_table,
                 int dim);

/**
 * Token embedding lookup for zero-copy Q2_0 tables (Qwen3.5/3.6 "Bonsai"
 * GGUF models). Dequantizes only the one row needed — the full table is
 * ~5 GB for a 27B-class model's vocab, far too large to dequantize upfront
 * (see weights.h's q35_token_embd_raw comment).
 *
 * @param out             Output float vector of size dim (pre-allocated)
 * @param token           Token ID (0-based index into the embedding table)
 * @param embd_q2_0_raw   Raw Q2_0 tensor bytes (mmap pointer, whole table)
 * @param dim             Embedding dimension
 */
void embed_token_q2_0(float *out, int token, const void *embd_q2_0_raw, int dim);

#endif /* TN_EMBEDDING_H */
