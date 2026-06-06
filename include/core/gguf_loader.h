#ifndef TN_GGUF_LOADER_H
#define TN_GGUF_LOADER_H

#include "core/config.h"
#include "core/weights.h"
#include "core/moe_config.h"
#include "core/gguf_reader.h"
#include "core/error.h"

/* Opaque store tracking all heap buffers allocated during GGUF weight loading.
 * Pass to weights_free_gguf() after weights_free_pointers(). */
typedef struct GGUFWeightStore GGUFWeightStore;

/* Populate Config from GGUF metadata (llama.* keys). */
TernaryError config_from_gguf(Config *cfg, const GGUFHeader *hdr);

/* Populate MoEConfig from GGUF metadata for DeepSeek-V2 models.
 * Sets is_moe=false and returns TN_OK for non-deepseek2 architectures.
 * For deepseek2: reads expert counts, MLA dimensions, etc. */
TernaryError moe_config_from_gguf(MoEConfig *mc, const GGUFHeader *hdr);

/* Allocate and populate TransformerWeights from a parsed GGUFHeader.
 * Supported tensor types: F32, F16, BF16 (others emit error and return TN_ERR).
 * F16/BF16 tensors are dequantized to float32 at load time (heap-allocated).
 * *store is set to a handle that must be freed with weights_free_gguf(). */
TernaryError weights_from_gguf(TransformerWeights *w, const Config *cfg,
                                const GGUFHeader *hdr, GGUFWeightStore **store);

/* Free all dequantized buffers allocated by weights_from_gguf(). */
void weights_free_gguf(GGUFWeightStore *store);

#endif /* TN_GGUF_LOADER_H */
