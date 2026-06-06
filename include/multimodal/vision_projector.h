#ifndef TN_VISION_PROJECTOR_H
#define TN_VISION_PROJECTOR_H

#include "core/error.h"
#include "threading/thread_pool.h"
#include <stddef.h>

/**
 * MLP Vision Projector to bridge vision encoder embeddings into LLM embedding space.
 *
 * Two modes determined by scale_factor:
 *   scale_factor == 1: two-layer MLP — gelu(x @ w_down.T + b_down) @ w_up.T + b_up
 *   scale_factor  > 1: pixel-shuffle + single linear.
 *       Patches are spatially grouped (scale_factor x scale_factor blocks) and their
 *       features concatenated, then w_up maps (vision_dim * scale_factor^2) → llm_dim.
 *       hidden_dim is 0; w_down is NULL; w_up holds the single linear weight.
 */
typedef struct {
    int vision_dim;    /* Per-patch input dimension from encoder (e.g., 768) */
    int llm_dim;       /* Output dimension matching LLM hidden size (e.g., 576) */
    int hidden_dim;    /* Hidden dimension for 2-layer MLP; 0 for single-linear mode */
    int scale_factor;  /* Pixel-shuffle scale (1 = no shuffle; 4 = SmolVLM-256M style) */

    float *w_down;     /* 2-layer MLP: [hidden_dim * vision_dim]; NULL in single-linear mode */
    float *w_up;       /* 2-layer MLP: [llm_dim * hidden_dim]; single-linear: [llm_dim * (vision_dim*s^2)] */
    float *bias_down;  /* [hidden_dim] (optional, NULL if no bias or single-linear) */
    float *bias_up;    /* [llm_dim] (optional, can be NULL) */
} VisionProjector;

/**
 * Loads the Vision Projector weights from a memory-mapped pointer.
 */
TernaryError vision_projector_load(VisionProjector *proj, const void *mapped_ptr, size_t *offset);

/**
 * Runs the MLP projection for a single patch.
 * out: [llm_dim]
 * patch_embedding: [vision_dim]
 */
void vision_projector_forward(float *out, const float *patch_embedding, const VisionProjector *proj, ThreadPool *tp);

/**
 * Runs the projection for a batch of encoder patches.
 * out:     [output_tokens * llm_dim]  — see vision_projector_output_tokens() for count
 * patches: [num_patches * vision_dim]
 */
void vision_projector_forward_batch(float *out, const float *patches, int num_patches, const VisionProjector *proj, ThreadPool *tp);

/**
 * Returns the number of LLM tokens produced by the projector given num_encoder_patches.
 * For scale_factor==1: returns num_encoder_patches (one token per patch).
 * For scale_factor>1:  returns num_encoder_patches / (scale_factor * scale_factor).
 */
int vision_projector_output_tokens(const VisionProjector *proj, int num_encoder_patches);

#endif /* TN_VISION_PROJECTOR_H */
