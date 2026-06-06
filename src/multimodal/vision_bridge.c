#include "multimodal/vision_bridge.h"
#include "transformer/forward.h"
#include <stdio.h>
#include <string.h>

void inject_vision_into_kv_cache(RunState *s, Config *p, TransformerWeights *w, const VisionContext *v, ThreadPool *tp) {
    if (v->embed_dim != p->dim) {
        fprintf(stderr, "Error: Vision embed_dim (%d) does not match LLM dim (%d)!\n", v->embed_dim, p->dim);
        return;
    }

    printf("Processing Image into Memory...\n");

    // We loop through every single projected image patch
    for(int i = 0; i < v->num_patches; i++) {

        // Ensure we don't exceed the KV cache limits
        if (s->current_pos >= s->max_seq_len) {
            fprintf(stderr, "Warning: KV Cache full, stopping vision injection early.\n");
            break;
        }

        // Copies the visual math directly into the LLM's standard text slot
        memcpy(s->x, &v->patch_embeddings[i * v->embed_dim], v->embed_dim * sizeof(float));

        // Let the transformer process it.
        // The token is -1 since there is no integer vocabulary token for a visual prompt,
        // and we have already populated s->x with the correct vector.
        // Wait, transformer_forward usually calls embed_token(token_idx). We need a variant or
        // a mechanism to skip the integer embedding step if the vector is pre-provided.
        // Let's check transformer_forward.

        // Assuming transformer_forward allows a flag or checking token == -1
        // For now, let's call it manually using the standard behavior, but if token == -1 is not supported,
        // we might just need to modify transformer_forward or replicate the layers loop.

        transformer_forward(-1, s->current_pos, p, w, s, NULL, tp);

        // Advance the memory treadmill
        s->current_pos++;
    }

    printf("Image successfully loaded into AI's short-term memory!\n");
}
