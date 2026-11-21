#ifndef TN_GGUF_DEQUANT_BLOCKS_H
#define TN_GGUF_DEQUANT_BLOCKS_H

/* The shared walk of every block-structured GGUF quantisation: a format supplies how one block
 * decodes (gguf_quant*.c); this decodes the whole blocks of a tensor. */

#include <stddef.h>
#include <stdint.h>

/* Decodes one block at blk into block_elems floats at dst. */
typedef void (*GgufBlockDecode)(float *dst, const uint8_t *blk);

/* Decodes the n_elems / block_elems whole blocks; returns how many weights that was (the caller
 * deals with a partial last block: some formats decode it, most zero it). */
size_t gguf_dequant_blocks(float *out, const void *data, size_t n_elems, size_t block_elems,
                           size_t block_bytes, GgufBlockDecode decode);

/* Zeroes out[done .. n_elems). */
void gguf_dequant_zero_tail(float *out, size_t done, size_t n_elems);

#endif /* TN_GGUF_DEQUANT_BLOCKS_H */
