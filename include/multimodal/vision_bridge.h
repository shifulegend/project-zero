#ifndef TN_VISION_BRIDGE_H
#define TN_VISION_BRIDGE_H

#include "core/config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"

/**
 * Context holding the final projected vision embeddings ready for LLM processing.
 */
typedef struct {
    int num_patches;         /* Number of image patches */
    int embed_dim;           /* Must perfectly match Config->dim */
    float *patch_embeddings; /* [num_patches * embed_dim] */
} VisionContext;

/**
 * Sequentially copies each visual patch embedding into the LLM's scratch vector and
 * runs the standard forward pass to populate the AI's short-term KV memory.
 *
 * @param s RunState (modified in-place, advances current_pos)
 * @param p LLM Config
 * @param w LLM Weights
 * @param v Vision Context containing projected embeddings
 * @param tp Thread Pool
 */
void inject_vision_into_kv_cache(RunState *s, Config *p, TransformerWeights *w, const VisionContext *v, ThreadPool *tp);

#endif /* TN_VISION_BRIDGE_H */
