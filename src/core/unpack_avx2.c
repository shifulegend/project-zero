#include "core/platform.h"

#if TN_HAS_AVX2

#include "core/unpack.h"
#include <immintrin.h>

/**
 * AVX2 ternary weight unpacker.
 *
 * Unpacks 32 weights (8 packed bytes) at a time using bit manipulation:
 * 1. Load 8 packed bytes, each containing 4 x 2-bit ternary values
 * 2. Use shifts and masks to extract each 2-bit field
 * 3. Subtract 1 to decode: 0b00->-1, 0b01->0, 0b10->1
 * 4. Store as 32 int8 values
 *
 * This avoids the complexity of shuffle-LUT interleaving while still
 * achieving significant throughput improvement over scalar.
 */

void unpack_ternary_block_avx2(tn_i8 *out, const tn_u8 *packed, int count) {
    const __m256i mask2 = _mm256_set1_epi8(0x03);
    const __m256i one = _mm256_set1_epi8(1);

    int i = 0;

    /* Process 32 weights (8 packed bytes) per iteration.
     * Each packed byte has 4 weights at bits [1:0], [3:2], [5:4], [7:6].
     * We expand each byte into 4 output bytes by extracting each 2-bit field. */
    for (; i + 31 < count; i += 32) {
        int byte_offset = i >> 2;  /* i / 4 */

        /* Load 8 packed bytes into a 64-bit value, zero-extend to 256-bit.
         * We'll process these 8 bytes to produce 32 unpacked values. */
        __m128i raw8 = _mm_loadl_epi64((const __m128i *)&packed[byte_offset]);

        /* We need to expand 8 bytes into 32 bytes (4x expansion).
         * Strategy: create 4 copies of each byte, shift by different amounts,
         * and mask to extract each 2-bit field.
         *
         * Use _mm_shuffle_epi8 to replicate each byte 4 times:
         * byte[0] -> positions 0,1,2,3; byte[1] -> positions 4,5,6,7; etc.
         */
        const __m128i expand_lo = _mm_setr_epi8(
            0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
        const __m128i expand_hi = _mm_setr_epi8(
            4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7);

        __m128i expanded_lo = _mm_shuffle_epi8(raw8, expand_lo); /* bytes 0-3 expanded */
        __m128i expanded_hi = _mm_shuffle_epi8(raw8, expand_hi); /* bytes 4-7 expanded */

        __m256i expanded = _mm256_set_m128i(expanded_hi, expanded_lo);

        /* Now each group of 4 consecutive bytes has the same packed byte.
         * Each position needs a different shift to extract its 2-bit field:
         *   position 0: shift right 0 (bits [1:0])
         *   position 1: shift right 2 (bits [3:2])
         *   position 2: shift right 4 (bits [5:4])
         *   position 3: shift right 6 (bits [7:6])
         * _mm256_srlv_epi32 only works for 32-bit lanes and bytes have no
         * variable-shift instruction, so each field is extracted with a
         * fixed shift + mask at the byte level instead (f0..f3 below), using
         * the already-replicated `expanded` data.
         */

        /* Field 0: bits [1:0] - no shift needed */
        __m256i f0 = _mm256_and_si256(expanded, mask2);

        /* Field 1: bits [3:2] - shift right by 2 */
        __m256i f1 = _mm256_and_si256(_mm256_srli_epi16(expanded, 2), mask2);

        /* Field 2: bits [5:4] - shift right by 4 */
        __m256i f2 = _mm256_and_si256(_mm256_srli_epi16(expanded, 4), mask2);

        /* Field 3: bits [7:6] - shift right by 6 */
        __m256i f3 = _mm256_and_si256(_mm256_srli_epi16(expanded, 6), mask2);

        /* Now we have 4 vectors where every 4th element is correct:
         *   f0: positions 0,4,8,12,... have the right value
         *   f1: positions 1,5,9,13,... have the right value
         *   f2: positions 2,6,10,14,... have the right value
         *   f3: positions 3,7,11,15,... have the right value
         *
         * But actually, because we replicated each byte 4 times, each group
         * of 4 has the same byte. So f0[0..3] all decode the same byte but
         * f0 extracts bits[1:0], f1 extracts bits[3:2], etc.
         * We need to select: from group k, take f0[4k], f1[4k+1], f2[4k+2], f3[4k+3].
         *
         * Use blend to select the right field for each position within each group:
         *   Positions 0,4,8,...  -> from f0
         *   Positions 1,5,9,...  -> from f1
         *   Positions 2,6,10,... -> from f2
         *   Positions 3,7,11,... -> from f3
         *
         * _mm256_blendv_epi8 uses the MSB of each byte as a selector.
         * We can build a mask that selects appropriately. */

        /* Build selection mask: for each position in a group of 4:
         *   pos%4 == 0: select f0 (mask = 0x00)
         *   pos%4 == 1: select f1 (mask = 0xFF)
         *   pos%4 == 2: select f2 (mask = 0xFF)
         *   pos%4 == 3: select f3 (mask = 0xFF) */

        /* Simpler: blend f0+f1 first, then blend that with f2+f3 */
        /* Even simpler: since all 4 positions within a group have the same
         * replicated byte, and each fx extracts a different 2-bit field,
         * we just need to pick fx[pos] where x = pos%4. */

        /* Let's just use a position mask to select:
         *   mask_01: 0xFF at positions 1,5,9,13,...  (select f1 over f0)
         *   mask_23: 0xFF at positions 2,3,6,7,10,11,... (select f2/f3 over f0/f1)
         *   mask_3:  0xFF at positions 3,7,11,15,... (select f3 over f2) */
        const __m256i mask_1 = _mm256_setr_epi8(
            0x00,(char)0xFF,0x00,0x00, 0x00,(char)0xFF,0x00,0x00,
            0x00,(char)0xFF,0x00,0x00, 0x00,(char)0xFF,0x00,0x00,
            0x00,(char)0xFF,0x00,0x00, 0x00,(char)0xFF,0x00,0x00,
            0x00,(char)0xFF,0x00,0x00, 0x00,(char)0xFF,0x00,0x00);

        const __m256i mask_2 = _mm256_setr_epi8(
            0x00,0x00,(char)0xFF,0x00, 0x00,0x00,(char)0xFF,0x00,
            0x00,0x00,(char)0xFF,0x00, 0x00,0x00,(char)0xFF,0x00,
            0x00,0x00,(char)0xFF,0x00, 0x00,0x00,(char)0xFF,0x00,
            0x00,0x00,(char)0xFF,0x00, 0x00,0x00,(char)0xFF,0x00);

        const __m256i mask_3_sel = _mm256_setr_epi8(
            0x00,0x00,0x00,(char)0xFF, 0x00,0x00,0x00,(char)0xFF,
            0x00,0x00,0x00,(char)0xFF, 0x00,0x00,0x00,(char)0xFF,
            0x00,0x00,0x00,(char)0xFF, 0x00,0x00,0x00,(char)0xFF,
            0x00,0x00,0x00,(char)0xFF, 0x00,0x00,0x00,(char)0xFF);

        /* Start with f0, blend in f1 at positions 1,5,9,... */
        __m256i result = _mm256_blendv_epi8(f0, f1, mask_1);
        /* Blend in f2 at positions 2,6,10,... */
        result = _mm256_blendv_epi8(result, f2, mask_2);
        /* Blend in f3 at positions 3,7,11,... */
        result = _mm256_blendv_epi8(result, f3, mask_3_sel);

        /* Subtract 1 to decode: 0->-1, 1->0, 2->1 */
        result = _mm256_sub_epi8(result, one);

        _mm256_storeu_si256((__m256i *)&out[i], result);
    }

    /* Scalar tail */
    for (; i < count; i++) {
        out[i] = unpack_ternary(packed, i);
    }
}

#endif /* TN_HAS_AVX2 */
