#include "kv_cache/kv_compress.h"
#include <math.h>

/* ================================================================
 * Phase 8.1: int8 absmax quantization
 * ================================================================ */

void kv_compress_to_8bit(const float *src, CompressedKVSlot *dst, int dim) {
  /* Step 1: Find absolute maximum */
  float amax = 0.0f;
  for (int i = 0; i < dim; i++) {
    float a = fabsf(src[i]);
    if (a > amax)
      amax = a;
  }

  /* Step 2: Compute scale (handle zero vector) */
  if (amax < 1e-30f) {
    dst->scale = 0.0f;
    for (int i = 0; i < dim; i++) {
      dst->data[i] = 0;
    }
    return;
  }

  float inv_scale = 127.0f / amax;
  dst->scale = amax / 127.0f;

  /* Step 3: Quantize */
  for (int i = 0; i < dim; i++) {
    float v = src[i] * inv_scale;
    /* Round to nearest, clamp to [-127, 127] */
    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
    if (q > 127)
      q = 127;
    if (q < -127)
      q = -127;
    dst->data[i] = (tn_i8)q;
  }
}

void kv_decompress_from_8bit(float *dst, const CompressedKVSlot *src, int dim) {
  float scale = src->scale;
  for (int i = 0; i < dim; i++) {
    dst[i] = (float)src->data[i] * scale;
  }
}

/* ================================================================
 * Phase 8.2: int4 absmax quantization (packed, 2 values per byte)
 * ================================================================ */

void kv_compress_to_4bit(const float *src, tn_u8 *dst, float *scale, int dim) {
  /* Step 1: Find absolute maximum */
  float amax = 0.0f;
  for (int i = 0; i < dim; i++) {
    float a = fabsf(src[i]);
    if (a > amax)
      amax = a;
  }

  /* Step 2: Compute scale */
  if (amax < 1e-30f) {
    *scale = 0.0f;
    int packed_len = (dim + 1) / 2;
    for (int i = 0; i < packed_len; i++) {
      dst[i] = 0x88; /* Both nibbles = 8 (which maps to 0 after -8 offset) */
    }
    return;
  }

  float inv_scale = 7.0f / amax;
  *scale = amax / 7.0f;

  /* Step 3: Quantize and pack, 2 values per byte.
   * Signed int4 range: [-7, 7]. We store as unsigned [0, 15] by adding 8.
   * Low nibble = even index, high nibble = odd index. */
  for (int i = 0; i < dim; i += 2) {
    float v0 = src[i] * inv_scale;
    int q0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f));
    if (q0 > 7)
      q0 = 7;
    if (q0 < -7)
      q0 = -7;
    tn_u8 u0 = (tn_u8)(q0 + 8); /* offset to unsigned [1..15], 0 maps to 8 */

    tn_u8 u1 = 8; /* default for odd dim */
    if (i + 1 < dim) {
      float v1 = src[i + 1] * inv_scale;
      int q1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f));
      if (q1 > 7)
        q1 = 7;
      if (q1 < -7)
        q1 = -7;
      u1 = (tn_u8)(q1 + 8);
    }

    dst[i / 2] = (u1 << 4) | (u0 & 0x0F);
  }
}

void kv_decompress_from_4bit(float *dst, const tn_u8 *src, float scale,
                             int dim) {
  for (int i = 0; i < dim; i += 2) {
    tn_u8 packed = src[i / 2];
    int q0 = (int)(packed & 0x0F) - 8; /* low nibble */
    dst[i] = (float)q0 * scale;

    if (i + 1 < dim) {
      int q1 = (int)(packed >> 4) - 8; /* high nibble */
      dst[i + 1] = (float)q1 * scale;
    }
  }
}
