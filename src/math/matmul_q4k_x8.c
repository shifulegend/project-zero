/*
 * matmul_q4k_x8.c — Q4_K x8-interleaved repack + multi-row SIMD-lane GEMV.
 *
 * A byte-exact port of llama.cpp's block_q4_Kx8 repack (ggml-cpu/repack.cpp,
 * make_block_q4_Kx8 with interleave_block=8) and its AVX2 GEMV kernel
 * (ggml-cpu/arch/x86/repack.cpp, ggml_gemv_q4_K_8x8_q8_K), added 2026-07-31
 * as attempt 6 on the Qwen3-8B-vs-llama.cpp Q4_K throughput gap — see
 * docs/ai/mistakes.md for the full attempt history (prefetch: neutral;
 * VNNI: regression; a memory-only "grouped8" repack: regression; a
 * macro-duplicated independent-accumulator kernel with no repack, at 4-row
 * and 2-row granularity: both regressions).
 *
 * Every prior attempt implemented at most HALF of llama.cpp's actual
 * technique. This implements both halves together, matching their design
 * exactly: 8 weight rows are physically repacked at load time so that a
 * single SIMD load pulls bytes from all 8 rows at once, and the GEMV kernel
 * uses _mm256_blend_epi32 tricks to compute partial products for multiple
 * rows within the SAME __m256i register (genuine SIMD-lane parallelism
 * across output rows), not just independent accumulator chains.
 *
 * block_q4_K and block_q8_K (llama.cpp's own structs) are byte-identical to
 * pz's own Q4_K row layout (144B: d/dmin fp16 + scales[12] + qs[128]) and
 * TnQ8KActBlock (float d + qs[256] + bsums[16]) respectively — confirmed by
 * reading ggml-common.h directly — so this port reads pz's existing raw
 * bytes/structs with no conversion needed.
 */

#include "math/matmul_q4k_x8.h"
#include "core/platform.h"
#include <string.h>

#if TN_HAS_AVX2
#include <immintrin.h>
#endif

#define Q4K_BYTES 144

/* ── Repack: 8 rows' raw Q4_K bytes -> interleaved TnQ4KX8Block ─────────────
 * Byte-exact port of make_block_q4_Kx8(..., blck_size_interleave=8). */
void tn_q4k_repack_x8(TnQ4KX8Block *out, const uint8_t *rows[8], int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        TnQ4KX8Block *ob = out + b;
        const uint8_t *blk[8];
        for (int r = 0; r < 8; r++)
            blk[r] = rows[r] + (size_t)b * Q4K_BYTES;

        for (int r = 0; r < 8; r++) {
            uint16_t d, dmin;
            memcpy(&d,    blk[r],     2);
            memcpy(&dmin, blk[r] + 2, 2);
            ob->d[r]    = d;
            ob->dmin[r] = dmin;
        }

        /* Interleave qs: 16 column-groups, each = 8 rows x 8 bytes,
         * taken from row r's original 128-byte qs at offset g*8. */
        for (int g = 0; g < 16; g++)
            for (int r = 0; r < 8; r++)
                memcpy(ob->qs + g * 64 + r * 8, blk[r] + 16 + g * 8, 8);

        /* Scale/min repack: for each of the 8 original Q4_K sub-blocks,
         * pack that sub-block's (scale, min) across all 8 rows into 12
         * bytes, in the standard 6-bit-packed layout — same bit-packing
         * formula as a single row's own scales[12], just transposed to
         * hold 8 rows instead of 8 sub-blocks. First 4 sub-blocks (i=0..3)
         * go to scales[0:48), remaining 4 (i=4..7) to scales[48:96). */
        for (int i = 0; i < 4; i++) {
            uint8_t s[8], m[8];
            for (int j = 0; j < 8; j++) {
                const uint8_t *sc12 = blk[j] + 4;
                s[j] = sc12[i] & 63;
                m[j] = sc12[i + 4] & 63;
            }
            uint8_t *o = ob->scales + i * 12;
            o[0]  = (uint8_t)((s[0] & 63) + ((s[4] & 48) << 2));
            o[1]  = (uint8_t)((s[1] & 63) + ((s[5] & 48) << 2));
            o[2]  = (uint8_t)((s[2] & 63) + ((s[6] & 48) << 2));
            o[3]  = (uint8_t)((s[3] & 63) + ((s[7] & 48) << 2));
            o[4]  = (uint8_t)((m[0] & 63) + ((m[4] & 48) << 2));
            o[5]  = (uint8_t)((m[1] & 63) + ((m[5] & 48) << 2));
            o[6]  = (uint8_t)((m[2] & 63) + ((m[6] & 48) << 2));
            o[7]  = (uint8_t)((m[3] & 63) + ((m[7] & 48) << 2));
            o[8]  = (uint8_t)((s[4] & 15) + ((m[4] & 15) << 4));
            o[9]  = (uint8_t)((s[5] & 15) + ((m[5] & 15) << 4));
            o[10] = (uint8_t)((s[6] & 15) + ((m[6] & 15) << 4));
            o[11] = (uint8_t)((s[7] & 15) + ((m[7] & 15) << 4));
        }
        for (int i = 0; i < 4; i++) {
            uint8_t s[8], m[8];
            for (int j = 0; j < 8; j++) {
                const uint8_t *sc12 = blk[j] + 4;
                s[j] = (uint8_t)(((sc12[i] & 192) >> 2) | (sc12[i + 8] & 15));
                m[j] = (uint8_t)(((sc12[i + 4] & 192) >> 2) | ((sc12[i + 8] & 240) >> 4));
            }
            uint8_t *o = ob->scales + 48 + i * 12;
            o[0]  = (uint8_t)((s[0] & 63) + ((s[4] & 48) << 2));
            o[1]  = (uint8_t)((s[1] & 63) + ((s[5] & 48) << 2));
            o[2]  = (uint8_t)((s[2] & 63) + ((s[6] & 48) << 2));
            o[3]  = (uint8_t)((s[3] & 63) + ((s[7] & 48) << 2));
            o[4]  = (uint8_t)((m[0] & 63) + ((m[4] & 48) << 2));
            o[5]  = (uint8_t)((m[1] & 63) + ((m[5] & 48) << 2));
            o[6]  = (uint8_t)((m[2] & 63) + ((m[6] & 48) << 2));
            o[7]  = (uint8_t)((m[3] & 63) + ((m[7] & 48) << 2));
            o[8]  = (uint8_t)((s[4] & 15) + ((m[4] & 15) << 4));
            o[9]  = (uint8_t)((s[5] & 15) + ((m[5] & 15) << 4));
            o[10] = (uint8_t)((s[6] & 15) + ((m[6] & 15) << 4));
            o[11] = (uint8_t)((s[7] & 15) + ((m[7] & 15) << 4));
        }
    }
}

#if TN_HAS_AVX2
/* ── GEMV: one group of 8 interleaved rows x Q8K activations -> 8 floats ───
 * Byte-exact port of ggml_gemv_q4_K_8x8_q8_K's AVX2 body (the single-x-
 * iteration inner content, driven here per-group instead of via the
 * y/x loop nest — this file drives one group per call from a thread task). */
static void gemv8_q4k_x8_q8k(float *out8, const TnQ8KActBlock *acts,
                              const TnQ4KX8Block *b_ptr, int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    const __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    const __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    __m256 acc_row = _mm256_setzero_ps();
    __m256 acc_min_rows = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const __m256 row_scale_f32 = _mm256_set1_ps(acts[b].d);
        const __m256 col_scale_f32 = _mm256_cvtph_ps(
            _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)b_ptr[b].d), deltamask));
        const __m256 col_dmin_f32 = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)b_ptr[b].dmin));

        __m256i iacc_b = _mm256_setzero_si256();
        __m256i iacc_min_b = _mm256_setzero_si256();

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)acts[b].bsums);
        __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
        q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

        for (int sb = 0; sb < 4; sb++) {
            const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
            const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
            const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
            const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
            const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
            const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
            const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
            const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

            const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
            const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
            const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
            const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
            const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
            const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
            const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
            const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

            const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
            const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
            const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
            const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
            const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
            const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
            const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
            const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

            uint32_t utmp_0[4], utmp_1[4];
            memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
            utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
            const uint32_t uaux_0 = utmp_0[1] & kmask1;
            utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
            utmp_0[2] = uaux_0;
            utmp_0[0] &= kmask1;

            memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
            utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
            const uint32_t uaux_1 = utmp_1[1] & kmask1;
            utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
            utmp_1[2] = uaux_1;
            utmp_1[0] &= kmask1;

            const __m128i mins_and_scales_0 = _mm_set_epi32((int)utmp_0[3], (int)utmp_0[2], (int)utmp_0[1], (int)utmp_0[0]);
            const __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
            const __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

            const __m128i mins_and_scales_1 = _mm_set_epi32((int)utmp_1[3], (int)utmp_1[2], (int)utmp_1[1], (int)utmp_1[0]);
            const __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
            const __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

            const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                _mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

            __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(acts[b].qs + sb * 64)));
            __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(acts[b].qs + 16 + sb * 64)));
            __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(acts[b].qs + 32 + sb * 64)));
            __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(acts[b].qs + 48 + sb * 64)));

            lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
            lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
            lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
            lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

            __m256i iacc_0 = _mm256_setzero_si256();
            __m256i iacc_1 = _mm256_setzero_si256();

            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00, _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177), rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01, _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177), rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02, _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177), rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03, _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
            iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177), rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10, _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177), rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11, _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177), rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12, _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177), rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13, _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
            iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177), rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            const __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

            const __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
            const __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            q8s = _mm256_bsrli_epi128(q8s, 4);

            iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
    }

    acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    _mm256_storeu_ps(out8, _mm256_sub_ps(acc_row, acc_min_rows));
}
#else
/* Scalar fallback (ARM / no-AVX2 builds): de-interleave and compute each of
 * the 8 rows independently with the standard Q4_K x Q8K formula. Not
 * performance-critical (this repack format is only selected on AVX2 hosts
 * at load time — see gguf_loader.c), just needs to be correct so the
 * project compiles and runs everywhere.
 *
 * Sub-block index i (0..7) maps to original group g=i/2, half=i%2 (0=low
 * nibble, 1=high nibble) — both halves of a sub-block pair read the SAME
 * 32-byte row range [g*32, g*32+32), split by nibble (matches
 * dot_q4k_row_q8k's own group/half split in matmul_q4k.c). Within that row
 * byte range, byte k (0..31) lives in the repacked layout at column-group
 * (g*32+k)/8, byte-offset (g*32+k)%8 within that group's 8-byte-per-row
 * slice (tn_q4k_repack_x8's own interleave: column-group c, row r ->
 * ob->qs[c*64 + r*8 .. c*64+r*8+8)). */
static void gemv8_q4k_x8_q8k(float *out8, const TnQ8KActBlock *acts,
                              const TnQ4KX8Block *b_ptr, int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int r = 0; r < 8; r++) {
        float dot = 0.0f;
        for (int b = 0; b < nb; b++) {
            const TnQ4KX8Block *bb = b_ptr + b;
            const TnQ8KActBlock *act = acts + b;

            float d_w, dmin_w;
            {
                uint16_t h = bb->d[r];
                uint32_t sign = (uint32_t)(h >> 15) << 31;
                uint32_t exp  = (h >> 10) & 0x1f;
                uint32_t bits = exp == 0 ? sign
                    : sign | ((exp + 112u) << 23) | ((uint32_t)(h & 0x3ff) << 13);
                memcpy(&d_w, &bits, 4);
                h = bb->dmin[r];
                sign = (uint32_t)(h >> 15) << 31;
                exp  = (h >> 10) & 0x1f;
                bits = exp == 0 ? sign
                    : sign | ((exp + 112u) << 23) | ((uint32_t)(h & 0x3ff) << 13);
                memcpy(&dmin_w, &bits, 4);
            }

            int32_t int_dot = 0, int_min = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t utmp[4];
                memcpy(utmp, bb->scales + (i < 4 ? i * 12 : 48 + (i - 4) * 12), 12);
                utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                const uint32_t uaux = utmp[1] & kmask1;
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[2] = uaux;
                utmp[0] &= kmask1;
                const uint8_t *sc8 = (const uint8_t *)utmp;
                uint8_t sc = sc8[r];

                int g = i / 2, half = i % 2;
                int32_t sum = 0;
                for (int l = 0; l < 32; l++) {
                    int row_byte_off = g * 32 + l;
                    int col_group = row_byte_off / 8;
                    int byte_in_group = row_byte_off % 8;
                    uint8_t byte = bb->qs[col_group * 64 + r * 8 + byte_in_group];
                    int nib = half ? (byte >> 4) : (byte & 0x0F);
                    sum += nib * (int32_t)act->qs[i * 32 + l];
                }
                int_dot += (int32_t)sc * sum;
            }
            for (int j = 0; j < 8; j++) {
                int32_t bsum_sb = (int32_t)act->bsums[2 * j] + (int32_t)act->bsums[2 * j + 1];
                uint32_t utmp[4];
                memcpy(utmp, bb->scales + (j < 4 ? j * 12 : 48 + (j - 4) * 12), 12);
                utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                const uint32_t uaux = utmp[1] & kmask1;
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[2] = uaux;
                utmp[0] &= kmask1;
                const uint8_t *sc8 = (const uint8_t *)utmp;
                uint8_t mn = sc8[r + 8];
                int_min += (int32_t)mn * bsum_sb;
            }
            dot += act->d * (d_w * (float)int_dot - dmin_w * (float)int_min);
        }
        out8[r] = dot;
    }
}
#endif /* TN_HAS_AVX2 */

typedef struct {
    float *out;
    const TnQ8KActBlock *acts;
    const TnQ4KX8Block *repacked;
    int n_blocks, groups;
} MatmulQ4KX8Args;

static void matmul_q4k_x8_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MatmulQ4KX8Args *a = (const MatmulQ4KX8Args *)arg;
    for (int g = start; g < end; g++) {
        gemv8_q4k_x8_q8k(a->out + (size_t)g * 8, a->acts,
                          a->repacked + (size_t)g * a->n_blocks, a->n_blocks);
    }
}

void parallel_matmul_q4k_x8_preq(float *out, const TnQ8KActBlock *acts,
                                  const TnQ4KX8Block *repacked,
                                  int n_blocks, int d, ThreadPool *tp) {
    int groups = d / 8;
    MatmulQ4KX8Args args = {
        .out = out,
        .acts = acts,
        .repacked = repacked,
        .n_blocks = n_blocks,
        .groups = groups,
    };
    if (!tp) { matmul_q4k_x8_task(&args, 0, 0, groups); return; }
    threadpool_dispatch(tp, matmul_q4k_x8_task, &args, groups);
}
