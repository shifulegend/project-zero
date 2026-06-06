#ifndef TN_VISION_ENCODER_H
#define TN_VISION_ENCODER_H

#include "core/config.h"
#include "threading/thread_pool.h"

typedef struct {
    int patch_dim;      /* Size of one patch (e.g. 16*16*3 = 768) */
    int embed_dim;      /* Vision encoder embedding dim (e.g. 768) */
    int hidden_dim;     /* Vision FFN hidden dim (e.g. 3072) */
    int n_layers;       /* Number of transformer layers */
    int n_heads;        /* Number of attention heads */
    int num_patches;    /* Total number of patches (e.g. 256) */
} VisionConfig;

typedef struct {
    float *patch_proj_w;  /* [embed_dim * patch_dim] */
    float *patch_proj_b;  /* [embed_dim] */
    float *pos_embed;     /* [num_patches * embed_dim] */

    float **wq;           /* [n_layers][embed_dim * embed_dim] */
    float **wk;           /* [n_layers][embed_dim * embed_dim] */
    float **wv;           /* [n_layers][embed_dim * embed_dim] */
    float **wo;           /* [n_layers][embed_dim * embed_dim] */

    float **w1;           /* [n_layers][hidden_dim * embed_dim] */
    float **w2;           /* [n_layers][embed_dim * hidden_dim] */
    float **w3;           /* [n_layers][hidden_dim * embed_dim] */

    float **rms_att_weight; /* [n_layers][embed_dim] */
    float **rms_ffn_weight; /* [n_layers][embed_dim] */
    float *rms_final_weight;/* [embed_dim] */
} VisionWeights;

/**
 * Runs the minimal ViT forward pass.
 * @param out_embeddings The output buffer of shape [num_patches * embed_dim].
 * @param patches The input patches of shape [num_patches * patch_dim].
 * @param cfg The Vision Encoder configuration.
 * @param w The Vision Encoder weights.
 * @param tp Thread pool for parallel computation.
 */
void vision_encoder_forward(float *out_embeddings, const float *patches, const VisionConfig *cfg, const VisionWeights *w, ThreadPool *tp);

#endif /* TN_VISION_ENCODER_H */
