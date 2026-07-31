#include "transformer/embedding.h"
#include "core/gguf_quant.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Q2_0 block: 34 bytes / 128 elements — PrismML's group-128 packing, NOT
 * mainline ggml's group-64 block_q2_0 (see gguf_quant.c's gguf_dequant_q2_0
 * comment for how this was confirmed against the real downloaded file). */
#define Q35_EMBED_Q2_0_BLOCK 128
#define Q35_EMBED_Q2_0_BYTES 34

void embed_token_q2_0(float *out, int token, const void *embd_q2_0_raw, int dim) {
    size_t row_bytes = (size_t)(dim / Q35_EMBED_Q2_0_BLOCK) * Q35_EMBED_Q2_0_BYTES;
    const uint8_t *row = (const uint8_t *)embd_q2_0_raw + (size_t)token * row_bytes;
    gguf_dequant_q2_0(out, row, (size_t)dim);
}

void embed_token(float *out, int token,
                 const float *embd_f32,
                 const tn_u16 *embedding_table,
                 int dim) {
    if (embd_f32) {
        /* F32 path: direct copy — matches llama.cpp's ggml_get_rows().
         * Q4_K → F32 dequant stored at load time, no BF16 intermediate. */
        memcpy(out, embd_f32 + (size_t)token * dim, (size_t)dim * sizeof(float));
    } else {
        /* BF16 path: bit-shift tn_u16 → float32.
         * Used by .bin models and BF16/F16/F32 GGUF embeddings. */
        const tn_u16 *row = &embedding_table[(size_t)token * dim];
        for (int i = 0; i < dim; i++) {
            tn_u32 f = (tn_u32)row[i] << 16;
            memcpy(&out[i], &f, sizeof(float));
        }
    }
}
