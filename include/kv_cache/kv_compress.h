#ifndef TN_KV_COMPRESS_H
#define TN_KV_COMPRESS_H

#include "core/platform.h"
#include <stddef.h>

/**
 * Compressed KV cache entry using int8 absmax quantization.
 * Memory savings: 4x vs float32.
 */
typedef struct {
  tn_i8 *data; /* Quantized int8 values */
  float scale; /* Absmax scale factor: max(|src|) / 127 */
} CompressedKVSlot;

/**
 * Compressed KV cache entry using int4 packing (two 4-bit values per byte).
 * Memory savings: 8x vs float32.
 */
typedef struct {
  tn_u8 *data; /* Packed int4 values (2 per byte) */
  float scale; /* Absmax scale factor: max(|src|) / 7 */
} CompressedKV4Slot;

/**
 * Compress a float vector to int8 using absmax quantization.
 *
 * Maps float values to [-127, 127] range:
 *   scale = max(|src[i]|) / 127.0
 *   dst->data[i] = round(src[i] / scale)
 *
 * @param src   Source float vector
 * @param dst   Destination compressed slot (data must be pre-allocated, dim
 * bytes)
 * @param dim   Number of elements
 */
void kv_compress_to_8bit(const float *src, CompressedKVSlot *dst, int dim);

/**
 * Decompress int8 back to float.
 *
 *   dst[i] = src->data[i] * src->scale
 *
 * @param dst   Destination float vector (must be pre-allocated, dim floats)
 * @param src   Source compressed slot
 * @param dim   Number of elements
 */
void kv_decompress_from_8bit(float *dst, const CompressedKVSlot *src, int dim);

/**
 * Compress a float vector to int4 (packed, 2 values per byte) using absmax.
 *
 * Maps float values to [-7, 7] range:
 *   scale = max(|src[i]|) / 7.0
 *   Each byte stores two 4-bit signed values:
 *     low nibble  = src[2*j]   mapped + 8 (offset to unsigned [0..15])
 *     high nibble = src[2*j+1] mapped + 8
 *
 * @param src   Source float vector
 * @param dst   Destination packed buffer (must be pre-allocated, ceil(dim/2)
 * bytes)
 * @param scale Output scale factor
 * @param dim   Number of elements
 */
void kv_compress_to_4bit(const float *src, tn_u8 *dst, float *scale, int dim);

/**
 * Decompress int4 packed values back to float.
 *
 * @param dst   Destination float vector (must be pre-allocated, dim floats)
 * @param src   Source packed buffer
 * @param scale Scale factor from compression
 * @param dim   Number of elements
 */
void kv_decompress_from_4bit(float *dst, const tn_u8 *src, float scale,
                             int dim);

#endif /* TN_KV_COMPRESS_H */
