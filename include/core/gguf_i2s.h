#ifndef TN_GGUF_I2S_H
#define TN_GGUF_I2S_H

/*
 * gguf_i2s.h — Microsoft BitNet I2_S tensors (bitnet.cpp's GGML_TYPE_I2_S = 36,
 * the format of the official microsoft/bitnet-b1.58-2B-4T-gguf release).
 *
 * On-disk layout (quantize_i2_s, x86 build, QK = 128):
 *   codes: n/4 bytes. Weights are taken in blocks of 128 over the flattened
 *          tensor; block b's byte k (0..31) holds weights b*128 + k + 32*g for
 *          g = 0..3 at bit shift 6 - 2g. Code 0 = -1, 1 = 0, 2 = +1.
 *   scale: one float32 right after the codes (the tensor's |w| scale), then
 *          padding to n/4 + 32 bytes.
 * The engine's packed ternary format (core/unpack.h) uses the same codes but
 * stores weights sequentially, LSB first, rows of packed_bytes(n) — so the
 * conversion is a lossless bit shuffle, not a dequantisation.
 */

#include <stddef.h>
#include <stdint.h>

#define TN_I2S_BLOCK 128

/* Bytes an I2_S tensor of n_elems weights occupies (codes + scale + pad). */
size_t tn_i2s_tensor_bytes(size_t n_elems);

/* Repack an I2_S tensor of d_rows x n_inner (row-major, n_inner contiguous)
 * into the engine's packed ternary rows (packed_bytes(n_inner) each).
 * dst must hold d_rows * packed_bytes(n_inner) bytes. *scale_out = tensor
 * scale. Returns 0, or -1 if n_inner is not a multiple of 128, a code is 3
 * (invalid), or the scale is not finite. */
int tn_i2s_repack(uint8_t *dst, float *scale_out, const uint8_t *src,
                  int d_rows, int n_inner);

/* Reference packer in bitnet.cpp's layout (tests / converters). vals are
 * ternary {-1,0,1}; dst must hold tn_i2s_tensor_bytes(n) bytes. */
void tn_i2s_pack_reference(uint8_t *dst, const int8_t *vals, size_t n, float scale);

#endif /* TN_GGUF_I2S_H */
