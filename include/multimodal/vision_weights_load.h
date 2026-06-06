#ifndef TN_VISION_WEIGHTS_LOAD_H
#define TN_VISION_WEIGHTS_LOAD_H

/*
 * vision_weights_load.h — Phase 34: Vision weight file loader
 *
 * Loads real SigLIP/ViT vision encoder weights and MLP projector weights
 * extracted by tools/extract_multimodal.py into the engine's VisionWeights
 * and VisionProjector structs.
 *
 * Binary format — vision.bin:
 *   [0:4]  magic   0x57535956 ("VISW" LE)
 *   [4:8]  version 1
 *   [8:12] n_layers  (int32)
 *   [12:16] embed_dim (int32)
 *   [16:20] hidden_dim (int32)
 *   [20:24] n_heads (int32)
 *   [24:28] patch_dim (int32)   — patch_size² × 3
 *   [28:32] num_patches (int32)
 *   [32:64] reserved (zeroes, pad to 64-byte alignment)
 *   Weights (float32, row-major):
 *     patch_proj_w [embed_dim × patch_dim]
 *     patch_proj_b [embed_dim]
 *     pos_embed    [num_patches × embed_dim]
 *     rms_final_weight [embed_dim]
 *     For each layer l:
 *       rms_att_weight [embed_dim]
 *       rms_ffn_weight [embed_dim]
 *       wq [embed_dim × embed_dim]
 *       wk [embed_dim × embed_dim]
 *       wv [embed_dim × embed_dim]
 *       wo [embed_dim × embed_dim]
 *       w1 [hidden_dim × embed_dim]   (gate)
 *       w2 [embed_dim × hidden_dim]   (down)
 *       w3 [hidden_dim × embed_dim]   (up)
 *
 * Binary format — projector.bin:
 *   [0:4]  magic   0x4A4F5250 ("PROJ" LE)
 *   [4:8]  version 1
 *   [8:12]  vision_dim (int32)
 *   [12:16] llm_dim    (int32)
 *   [16:20] hidden_dim (int32)
 *   [20:24] has_bias   (int32, 0 or 1)
 *   [24:64] reserved
 *   Weights:
 *     w_down    [hidden_dim × vision_dim]
 *     bias_down [hidden_dim]  (only if has_bias)
 *     w_up      [llm_dim × hidden_dim]
 *     bias_up   [llm_dim]     (only if has_bias)
 */

#include "core/error.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "memory/mapped_file.h"

#define VISION_BIN_MAGIC    0x57535956u  /* "VISW" LE */
#define PROJECTOR_BIN_MAGIC 0x4A4F5250u  /* "PROJ" LE */

/* Loaded state: owns the mmap, VisionWeights and VisionProjector point into it. */
typedef struct {
    MappedFile    vision_mf;
    MappedFile    proj_mf;
    VisionConfig  cfg;        /* populated from vision.bin header */
    VisionWeights weights;    /* pointers into vision_mf.data */
    VisionProjector proj;     /* pointers into proj_mf.data */
    int           vision_loaded;
    int           proj_loaded;
} VisionModel;

/**
 * Load vision encoder weights from a vision.bin file.
 * Populates vm->cfg, vm->weights, vm->vision_loaded.
 * Uses memory mapping — zero-copy, O(1) load time.
 */
TernaryError vision_model_load_encoder(VisionModel *vm, const char *vision_bin_path);

/**
 * Load MLP projector weights from a projector.bin file.
 * Populates vm->proj, vm->proj_loaded.
 */
TernaryError vision_model_load_projector(VisionModel *vm, const char *proj_bin_path);

/**
 * Print a brief summary of loaded vision model parameters.
 */
void vision_model_print_info(const VisionModel *vm);

/**
 * Free all resources.
 */
void vision_model_free(VisionModel *vm);

#endif /* TN_VISION_WEIGHTS_LOAD_H */
