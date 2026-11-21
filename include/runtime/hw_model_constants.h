#ifndef TN_HW_MODEL_CONSTANTS_H
#define TN_HW_MODEL_CONSTANTS_H

/* Compile-time BitNet-2B shape behind the hardware profile's pre-load estimates. */
/*
 * Model constants for BitNet-b1.58-2B-4T:
 *   30 layers, dim=2560, hidden_dim=6912, vocab=128256
 *   Ternary weights: 2 bits/weight packed as 4 weights/byte
 *   Embedding table: vocab * dim * 2 bytes (BF16)
 *
 * 2026-07-17: these constants are only a PRE-LOAD ESTIMATE (profile init
 * runs before the model file is even opened, so no metadata exists yet).
 * They are correct for the native BitNet-2B path they were written for,
 * but were silently used for every model — a 7.16 GB Ternary-Bonsai-27B
 * GGUF got Data/token "1149 MB" and a ~6x-overstated tok/s ceiling (see
 * docs/ai/mistakes.md). main.c now calls
 * tn_hardware_profile_set_model_bytes() with the real loaded-model sizes
 * to correct weight_bytes_per_tok and theoretical_ceiling post-load.
 */
#define MODEL_TERNARY_BYTES (522ULL * 1024 * 1024) /* ~522 MB packed ternary */
#define MODEL_EMBED_BYTES (128256ULL * 2560 * 2)   /* ~628 MB BF16 embedding */
#define MODEL_NORM_BYTES (1ULL * 1024 * 1024)      /* ~1 MB norms */
#define MODEL_DIM 2560
#define MODEL_VOCAB 128256

#endif /* TN_HW_MODEL_CONSTANTS_H */
