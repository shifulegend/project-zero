#ifndef TN_UNPACK_H
#define TN_UNPACK_H

#include "core/platform.h"

/**
 * 2-bit ternary weight unpacking.
 *
 * Packed format: 4 ternary weights per byte, 2 bits each.
 *   Encoding: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10
 *   Decoding: value = bits - 1 (i.e., 0b00=0 -> -1, 0b01=1 -> 0, 0b10=2 -> 1)
 *
 * Byte layout (LSB first):
 *   bits [1:0] = weight[0], bits [3:2] = weight[1],
 *   bits [5:4] = weight[2], bits [7:6] = weight[3]
 *
 * Phase 10.4 of the implementation plan.
 */

/**
 * Unpack a single ternary weight from a packed array.
 * Scalar reference — for testing and validation only.
 *
 * @param packed  Packed weight array (4 weights per byte)
 * @param index   Index of the weight to extract (0-based)
 * @return        Ternary value in {-1, 0, 1}
 */
static inline tn_i8 unpack_ternary(const tn_u8 *packed, int index) {
    tn_u8 byte = packed[index >> 2];       /* index / 4 */
    int shift = (index & 3) << 1;           /* (index % 4) * 2 */
    return (tn_i8)((byte >> shift) & 0x03) - 1;
}

/**
 * Pack a single ternary weight into a packed array.
 * For use by the Python converter test validation and C-side round-trip tests.
 *
 * @param packed  Packed weight array (4 weights per byte), must be zeroed
 * @param index   Index of the weight to set (0-based)
 * @param value   Ternary value in {-1, 0, 1}
 */
static inline void pack_ternary(tn_u8 *packed, int index, tn_i8 value) {
    int byte_idx = index >> 2;              /* index / 4 */
    int shift = (index & 3) << 1;           /* (index % 4) * 2 */
    tn_u8 encoded = (tn_u8)(value + 1);     /* -1->0, 0->1, 1->2 */
    packed[byte_idx] |= (encoded << shift);
}

/**
 * Unpack a contiguous block of ternary weights (scalar reference).
 *
 * @param out     Output buffer for unpacked int8 values (must hold count elements)
 * @param packed  Packed weight array
 * @param count   Number of ternary weights to unpack
 */
void unpack_ternary_block(tn_i8 *out, const tn_u8 *packed, int count);

/**
 * Unpack a contiguous block using AVX2 SIMD shuffle LUT.
 * Processes 128 weights per iteration (32 packed bytes -> 128 int8 values).
 * Falls back to scalar for the tail.
 *
 * Only available when TN_HAS_AVX2 is defined.
 */
void unpack_ternary_block_avx2(tn_i8 *out, const tn_u8 *packed, int count);

/**
 * Returns the number of packed bytes needed for `count` ternary weights.
 */
static inline size_t packed_bytes(int count) {
    return ((size_t)count + 3) >> 2;  /* ceil(count / 4) */
}

#endif /* TN_UNPACK_H */
