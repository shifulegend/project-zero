#include "transformer/embedding.h"
#include <stddef.h>
#include <string.h>

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
