#ifndef TN_MOE_WEIGHTS_H
#define TN_MOE_WEIGHTS_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "core/error.h"

/**
 * Allocate the MoE expert weight pointer arrays inside TransformerWeights.
 * No-op when mc->is_moe == false (dense model).
 * Must be called AFTER weights_alloc_pointers() and BEFORE moe_weights_map().
 */
TernaryError moe_weights_alloc(TransformerWeights *w, const Config *cfg,
                                const MoEConfig *mc);

/**
 * Free MoE pointer arrays allocated by moe_weights_alloc().
 * No-op when mc->is_moe == false.
 * Safe to call even if allocation was partial (handles NULL pointers).
 */
void moe_weights_free(TransformerWeights *w, const MoEConfig *mc);

/**
 * Map MoE weights from a memory-mapped file.
 * ptr_inout is advanced past all MoE weight data for all layers.
 * No-op when mc->is_moe == false.
 *
 * Must be called AFTER the dense layer weight loop in weights_map().
 *
 * Binary layout per MoE layer (see moe_weights.c for full spec):
 *   - gate matrix (packed ternary) + scale
 *   - for each expert: w1 + s1, w3 + s3, w2 + s2 (all packed ternary + scale)
 */
TernaryError moe_weights_map(TransformerWeights *w, const Config *cfg,
                              const MoEConfig *mc, tn_i8 *data,
                              size_t data_size, tn_i8 **ptr_inout);

#endif /* TN_MOE_WEIGHTS_H */
