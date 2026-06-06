#include "core/unpack.h"

/**
 * Scalar reference unpacker — processes one weight at a time.
 *
 * This is the baseline that all SIMD variants are validated against.
 * It is intentionally simple and correct rather than fast.
 */
void unpack_ternary_block(tn_i8 *out, const tn_u8 *packed, int count) {
    for (int i = 0; i < count; i++) {
        out[i] = unpack_ternary(packed, i);
    }
}
