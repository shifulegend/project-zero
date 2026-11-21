#ifndef TN_RUNTIME_GGUF_LOAD_UTIL_H
#define TN_RUNTIME_GGUF_LOAD_UTIL_H

/* Internal to the GGUF weight loaders (src/runtime/gguf_load*.c): helpers they share and the
 * per-architecture entry points weights_from_gguf() dispatches to. Not part of the public API
 * (that is runtime/gguf_loader.h). */

#include <stddef.h>
#include <string.h>

#include "core/config.h"
#include "core/error.h"
#include "core/gguf_reader.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include "core/weights.h"
#include "runtime/gguf_loader.h"

/* Round-to-nearest-even float32 -> bf16 (not a plain truncation: avoids a small one-sided bias
 * across a whole classifier tensor). */
static inline tn_u16 gguf_ld_f32_to_bf16_round(float f) {
    tn_u32 bits;
    memcpy(&bits, &f, sizeof(bits));
    tn_u32 rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    bits += rounding_bias;
    return (tn_u16)(bits >> 16);
}

/* Growable list of heap buffers the loaded weights own; freed by weights_free_gguf(). */
GGUFWeightStore *gguf_ld_store_alloc(void);
int gguf_ld_store_add(GGUFWeightStore *s, void *ptr); /* 0 ok, -1 out of memory */

/* Router + stacked routed-expert tensors of MoE layer `l` (DeepSeek-2 and Qwen3-MoE share it);
 * TN_ERR_INVALID_WEIGHTS with a log line when a tensor is missing or has an unknown type. */
TernaryError gguf_ld_moe_layer(const GGUFHeader *hdr, TransformerWeights *w, int l, int num_experts,
                               size_t expert_elems);
/* Dequantises n_elems weights of `type` into `out`; -1 (nothing written) when the type has no
 * dequantiser. */
int gguf_ld_dequant(GGUFType type, float *out, const void *data, size_t n_elems);
/* A float32 buffer (n_elems floats) filled from the tensor, registered in the store; NULL and a
 * message for unsupported types. */
float *gguf_ld_tensor_to_f32(const GGUFTensor *t, size_t n_elems, GGUFWeightStore *store);
/* Norm weights: F32 zero-copy, F16 / BF16 dequantised into the store. NULL when missing. */
float *gguf_ld_norm_to_f32(const GGUFHeader *hdr, const char *name, size_t n_elems,
                           GGUFWeightStore *store);
/* Bytes of n_elems elements of a block-quantised type. */
size_t gguf_ld_quant_bytes(GGUFType type, size_t n_elems);
/* Keeps a Q4_K / Q6_K output head as raw bytes (w->wcls_raw) when no --classifier format was
 * asked for; returns 1 when it did. */
int gguf_ld_classifier_keep_raw(TransformerWeights *w, const Config *cfg, const GGUFTensor *t);
/* Dense BF16 embedding + output.weight (deepseek2, qwen3moe, mamba). */
TernaryError gguf_ld_bf16_embedding_and_output(TransformerWeights *w, const Config *cfg,
                                               const GGUFHeader *hdr, GGUFWeightStore *store);
/* Typed zero-copy projection (F16 / F32 / Q4_K / Q6_K / Q5_0 / Q5_1 / Q8_0), else dequantised. */
TernaryError gguf_ld_proj_typed(const GGUFHeader *hdr, GGUFWeightStore *store, const char *name,
                                int d_rows, int n_inner, tn_i8 **out, int *type_out);

/* One projection of the dense path: "blk.<l>.<tname>" into field[l] / type_out[l] / scale_out[l]. */
TernaryError gguf_ld_dense_projection(const GGUFHeader *hdr, GGUFWeightStore *store, int l,
                                      const char *tname, int d_rows, int n_inner, tn_i8 **field,
                                      int *type_out, float *scale_out);

/* Per-architecture loaders. */
TernaryError weights_from_gguf_deepseek2(TransformerWeights *w, const Config *cfg,
                                         const GGUFHeader *hdr, const MoEConfig *mc,
                                         GGUFWeightStore *store);
TernaryError weights_from_gguf_qwen35(TransformerWeights *w, const Config *cfg,
                                      const GGUFHeader *hdr, const MoEConfig *mc,
                                      GGUFWeightStore *store);
TernaryError weights_from_gguf_qwen3moe(TransformerWeights *w, const Config *cfg,
                                        const GGUFHeader *hdr, const MoEConfig *mc,
                                        GGUFWeightStore *store);
TernaryError weights_from_gguf_mamba(TransformerWeights *w, const Config *cfg,
                                     const GGUFHeader *hdr, const MoEConfig *mc,
                                     GGUFWeightStore *store);

#endif
