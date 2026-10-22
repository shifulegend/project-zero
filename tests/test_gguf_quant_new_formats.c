/*
 * test_gguf_quant_new_formats.c — Phase 37 (GGUF Universal Quant Compatibility)
 *
 * Ground-truth correctness tests for the dequant functions added/fixed during the
 * 2026-09-17 Phase 37 push. Each test hand-computes expected float values from the
 * quant format's real bit layout (verified against llama.cpp's ggml-common.h /
 * ggml-quants.c upstream source) rather than re-deriving the formula under test —
 * catches byte-order/sign bugs that a self-consistency-only test would miss.
 */

#include "core/gguf_quant.h"
#include "test_harness.h"

#include <stdint.h>
#include <string.h>

/* ── fp16 encode helper (test-side only; production code decodes, doesn't encode) */
static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF); /* clamp to max fp16 */
    if (exp <= 102) return (uint16_t) sign;            /* underflow -> zero  */
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* ── Q4_1: block = [d fp16][m fp16][qs[16]], out[i] = nibble(i)*d + m ────────── */
static void test_q4_1_single_block(void) {
    uint8_t blk[20];
    float d = 0.5f, m = 1.0f;
    uint16_t d_h = f32_to_fp16(d), m_h = f32_to_fp16(m);
    memcpy(blk,     &d_h, 2);
    memcpy(blk + 2, &m_h, 2);
    /* qs[i] = (i << 4) | (15 - i): low nibble = 15-i (element i), high nibble = i (element i+16) */
    for (int i = 0; i < 16; i++) blk[4 + i] = (uint8_t)((i << 4) | (15 - i));

    float out[32];
    gguf_dequant_q4_1(out, blk, 32);

    /* Hand-computed spot checks (not re-deriving the formula under test): */
    TEST_ASSERT_FLOAT_EQ(out[0],  15 * 0.5f + 1.0f, 1e-5f, "q4_1 elem0 (low nibble=15)");   /* 8.5 */
    TEST_ASSERT_FLOAT_EQ(out[16], 0  * 0.5f + 1.0f, 1e-5f, "q4_1 elem16 (high nibble=0)");  /* 1.0 */
    TEST_ASSERT_FLOAT_EQ(out[15], 0  * 0.5f + 1.0f, 1e-5f, "q4_1 elem15 (low nibble=0)");   /* 1.0 */
    TEST_ASSERT_FLOAT_EQ(out[31], 15 * 0.5f + 1.0f, 1e-5f, "q4_1 elem31 (high nibble=15)"); /* 8.5 */
    TEST_ASSERT_FLOAT_EQ(out[7],  8  * 0.5f + 1.0f, 1e-5f, "q4_1 elem7 (low nibble=8)");    /* 5.0 */
}

/* Tail-elements-zeroed check: n_elems not a multiple of the block size must not
 * read/write past the requested count (matches the convention used by every other
 * gguf_dequant_* function in this file for partial trailing blocks). */
static void test_q4_1_partial_trailing(void) {
    uint8_t blk[20];
    memset(blk, 0, sizeof(blk));
    float out[10];
    for (int i = 0; i < 10; i++) out[i] = -999.0f; /* sentinel */
    gguf_dequant_q4_1(out, blk, 10); /* fewer than one full 32-elem block */
    for (int i = 0; i < 10; i++)
        TEST_ASSERT_FLOAT_EQ(out[i], 0.0f, 1e-5f, "q4_1 partial trailing block zero-fill");
}

/* ── Q8_1: block = [d fp16][s fp16][qs[32] int8], out[i] = qs[i]*d ──────────── */
static void test_q8_1_single_block(void) {
    uint8_t blk[36];
    float d = 0.25f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    memset(blk + 2, 0, 2); /* s unused for dequant */
    int8_t *qs = (int8_t *)(blk + 4);
    for (int i = 0; i < 32; i++) qs[i] = (int8_t)(i - 16); /* -16..15 */

    float out[32];
    gguf_dequant_q8_1(out, blk, 32);

    TEST_ASSERT_FLOAT_EQ(out[0],  -16 * 0.25f, 1e-5f, "q8_1 elem0 (-16*0.25=-4.0)");
    TEST_ASSERT_FLOAT_EQ(out[16],   0 * 0.25f, 1e-5f, "q8_1 elem16 (0*0.25=0.0)");
    TEST_ASSERT_FLOAT_EQ(out[31],  15 * 0.25f, 1e-5f, "q8_1 elem31 (15*0.25=3.75)");
}

/* ── Q8_K: block = [d fp32][qs[256] int8][bsums[16] int16], out[i] = qs[i]*d ── */
static void test_q8_k_single_block(void) {
    uint8_t blk[292];
    float d = 0.1f;
    memcpy(blk, &d, 4); /* genuine fp32, no fp16 conversion */
    int8_t *qs = (int8_t *)(blk + 4);
    for (int i = 0; i < 256; i++) qs[i] = (int8_t)((i % 200) - 100); /* stay in int8 range */
    memset(blk + 4 + 256, 0, 32); /* bsums unused for dequant */

    float out[256];
    gguf_dequant_q8_k(out, blk, 256);

    TEST_ASSERT_FLOAT_EQ(out[0],   -100 * 0.1f, 1e-5f, "q8_k elem0");
    TEST_ASSERT_FLOAT_EQ(out[100],    0 * 0.1f, 1e-5f, "q8_k elem100 ((100%200)-100=0)");
    TEST_ASSERT_FLOAT_EQ(out[199],   99 * 0.1f, 1e-5f, "q8_k elem199 ((199%200)-100=99)");
}

/* ── IQ4_XS: d(fp16)+scales_h(u16)+scales_l[4]+qs[128], reuses kvalues_iq4nl ── */
static void test_iq4_xs_single_block(void) {
    /* kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} */
    uint8_t blk[136];
    memset(blk, 0, sizeof(blk));
    float d = 0.125f; /* exact power-of-2, no fp16 rounding error to account for */
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    uint16_t scales_h = 0; /* high 2 bits of every ls contribute 0 -- isolates scales_l */
    memcpy(blk + 2, &scales_h, 2);
    uint8_t *scales_l = blk + 4;
    scales_l[0] = 0x35; /* ib=0 low nibble=5, ib=1 high nibble=3 */
    scales_l[1] = 0x0A; /* ib=2 low nibble=10, ib=3 high nibble=0 */
    uint8_t *qs = blk + 8;
    qs[0]  = 0xF0; /* ib=0, j=0: low nibble=0 -> elem0, high nibble=0xF -> elem16 */
    qs[48] = 0x0F; /* ib=3 (offset 3*16=48), j=0: low nibble=0xF -> elem96, high nibble=0 -> elem112 */

    float out[256];
    gguf_dequant_iq4_xs(out, blk, 256);

    /* ib=0: ls=5, dl = 0.125*(5-32) = -3.375 */
    TEST_ASSERT_FLOAT_EQ(out[0],  -3.375f * -127.0f, 1e-3f, "iq4_xs elem0 (ib0 low nibble=0)");
    TEST_ASSERT_FLOAT_EQ(out[16], -3.375f *  113.0f, 1e-3f, "iq4_xs elem16 (ib0 high nibble=15)");
    TEST_ASSERT_FLOAT_EQ(out[5],  -3.375f * -127.0f, 1e-3f, "iq4_xs elem5 (background nibble=0)");
    /* ib=3: ls=0, dl = 0.125*(0-32) = -4.0 */
    TEST_ASSERT_FLOAT_EQ(out[96],  -4.0f *  113.0f, 1e-3f, "iq4_xs elem96 (ib3 low nibble=15)");
    TEST_ASSERT_FLOAT_EQ(out[112], -4.0f * -127.0f, 1e-3f, "iq4_xs elem112 (ib3 high nibble=0)");
}

/* ── IQ2_XXS: d(fp16)+qs[64] (8 sub-blocks of 2x u32 = grid idx + signs+scale) */
static void test_iq2_xxs_single_block(void) {
    uint8_t blk[66];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f; /* exact in fp16 */
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    uint8_t *qs = blk + 2; /* 64 bytes, all zero except ib32=0's aux32[1] low byte */
    /* ib32=0: aux32[0]=0 (grid index 0 for all 4 groups), aux32[1]=1 (bit0 set:
     * sign-table index 1 for l=0 group only -> ksigns_iq2xs[1]=129=0b10000001,
     * flips j=0 and j=7 within that first 8-element group; l=1,2,3 groups still
     * read sign-table index 0 -> all positive). Every other ib32 (1..7) stays
     * all-zero -> grid index 0, sign index 0, scale nibble 0. */
    qs[4] = 1;

    float out[256];
    gguf_dequant_iq2_xxs(out, blk, 256);

    /* iq2xxs_grid[0] = 0x0808080808080808 (all bytes 8). Scale nibble=0 ->
     * db = d*(0.5+0)*0.25 = 0.125. Baseline (no sign flip) = 0.125*8 = 1.0. */
    TEST_ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-5f, "iq2_xxs elem1 (no sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "iq2_xxs elem0 (sign flip, ksigns[1] bit0)");
    TEST_ASSERT_FLOAT_EQ(out[7], -1.0f, 1e-5f, "iq2_xxs elem7 (sign flip, ksigns[1] bit7)");
    TEST_ASSERT_FLOAT_EQ(out[8],  1.0f, 1e-5f, "iq2_xxs elem8 (l=1 group, sign index 0)");
    /* All-zero sub-blocks (ib32=1..7, elements 32..255) must be the uniform
     * baseline 1.0 too -- same all-zero aux32 as ib32=0's l=1..3 groups. */
    TEST_ASSERT_FLOAT_EQ(out[100], 1.0f, 1e-5f, "iq2_xxs elem100 (all-zero sub-block)");
    TEST_ASSERT_FLOAT_EQ(out[255], 1.0f, 1e-5f, "iq2_xxs elem255 (all-zero sub-block)");
}

/* ── IQ2_XS: d(fp16)+qs[32] u16+scales[8], grid idx = low9 bits, sign = hi7 ── */
static void test_iq2_xs_single_block(void) {
    uint8_t blk[74];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    uint16_t *qs = (uint16_t *)(blk + 2);
    /* ib32=0, l=0: word = 1<<9 = 512 -> grid idx=0, sign idx=1 (ksigns[1]=129,
     * flips j=0 and j=7, same pattern as the IQ2_XXS test above). */
    qs[0] = 512;

    float out[256];
    gguf_dequant_iq2_xs(out, blk, 256);

    /* iq2xs_grid[0] = 0x0808080808080808 (all 8s). scales[0]=0 -> db=0.125. */
    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "iq2_xs elem0 (sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-5f, "iq2_xs elem1 (no sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[7], -1.0f, 1e-5f, "iq2_xs elem7 (sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[8],  1.0f, 1e-5f, "iq2_xs elem8 (l=1 group, all-zero word)");
    TEST_ASSERT_FLOAT_EQ(out[255], 1.0f, 1e-5f, "iq2_xs elem255 (all-zero sub-block)");
}

/* ── IQ2_S: d(fp16)+qs[64](idx[32]+signs[32])+qh[8]+scales[8] ──────────────── */
static void test_iq2_s_single_block(void) {
    uint8_t blk[82];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    uint8_t *qs_base = blk + 2;
    /* ib32=0, l=0: qs[0]=0, qh[0]=0 -> grid idx=0 -> iq2s_grid[0] (all 8s).
     * signs[0] (byte at offset 32 within qs_base) = 1 -> flips only j=0
     * (raw bitmask, no ksigns lookup for this format). */
    qs_base[32] = 1;

    float out[256];
    gguf_dequant_iq2_s(out, blk, 256);

    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "iq2_s elem0 (sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-5f, "iq2_s elem1 (no sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[7],  1.0f, 1e-5f, "iq2_s elem7 (bit7 not set, unlike ksigns[1])");
    TEST_ASSERT_FLOAT_EQ(out[8],  1.0f, 1e-5f, "iq2_s elem8 (l=1 group, all-zero)");
    TEST_ASSERT_FLOAT_EQ(out[255], 1.0f, 1e-5f, "iq2_s elem255 (all-zero sub-block)");
}

/* ── IQ3_XXS: d(fp16)+qs[96](idx[64]+scales_and_signs[32]) ─────────────────── */
static void test_iq3_xxs_single_block(void) {
    uint8_t blk[98];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    /* scales_and_signs starts at blk+2+64=blk[66]; ib32=0's aux32 = blk[66..69].
     * Set to 1 (LE) -> sign index 1 -> ksigns[1]=129 -> flips j=0,7; scale
     * nibble (aux32>>28) stays 0. */
    blk[66] = 1;

    float out[256];
    gguf_dequant_iq3_xxs(out, blk, 256);

    /* iq3xxs_grid[0] = 0x04040404 (all bytes 4). db = d*(0.5+0)*0.5 = 0.25. */
    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "iq3_xxs elem0 (sign flip, grid1 j=0)");
    TEST_ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-5f, "iq3_xxs elem1 (no flip)");
    TEST_ASSERT_FLOAT_EQ(out[7], -1.0f, 1e-5f, "iq3_xxs elem7 (sign flip, grid2 j=3->dst j+4=7)");
    TEST_ASSERT_FLOAT_EQ(out[8],  1.0f, 1e-5f, "iq3_xxs elem8 (l=1 group, all-zero)");
    TEST_ASSERT_FLOAT_EQ(out[255], 1.0f, 1e-5f, "iq3_xxs elem255 (all-zero sub-block)");
}

/* ── IQ3_S: d(fp16)+qs[64]+qh[8]+signs[32]+scales[4] ───────────────────────── */
static void test_iq3_s_single_block(void) {
    uint8_t blk[110];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    /* signs_base = blk+2+64+8 = blk[74]; ib32=0,l=0's signs[0] = blk[74].
     * Set to 1 -> flips only j=0 (grid1's first byte). scales[0]=0 -> db1=1. */
    blk[74] = 1;

    float out[256];
    gguf_dequant_iq3_s(out, blk, 256);

    /* iq3s_grid[0] = 0x01010101 (all bytes 1). db1 = d*(1+2*0) = 1.0. */
    TEST_ASSERT_FLOAT_EQ(out[0], -1.0f, 1e-5f, "iq3_s elem0 (sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[1],  1.0f, 1e-5f, "iq3_s elem1 (no flip)");
    TEST_ASSERT_FLOAT_EQ(out[7],  1.0f, 1e-5f, "iq3_s elem7 (bit7 not set)");
    TEST_ASSERT_FLOAT_EQ(out[8],  1.0f, 1e-5f, "iq3_s elem8 (l=1 group, all-zero)");
    TEST_ASSERT_FLOAT_EQ(out[255], 1.0f, 1e-5f, "iq3_s elem255 (all-zero sub-block)");
}

/* ── IQ1_S: d(fp16)+qs[32]+qh[8]xu16, shares iq1s_grid[2048] with IQ1_M ────── */
static void test_iq1_s_single_block(void) {
    uint8_t blk[50];
    memset(blk, 0, sizeof(blk));
    float d = 1.0f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    /* All-zero qs/qh: idx=0 -> iq1s_grid[0]=all -1 (int8). dl=d*(2*0+1)=d=1.
     * delta = +IQ1S_DELTA (qh bit15=0) -> dst = 1*(-1+0.125) = -0.875 uniform. */
    float out[256];
    gguf_dequant_iq1_s(out, blk, 256);
    TEST_ASSERT_FLOAT_EQ(out[0],   -0.875f, 1e-5f, "iq1_s elem0 (all-zero baseline)");
    TEST_ASSERT_FLOAT_EQ(out[255], -0.875f, 1e-5f, "iq1_s elem255 (all-zero baseline)");

    /* Set qh for ib=0 to 0x8000 (bit15) -> delta sign flips for that whole
     * sub-block only; idx untouched since bits 0-11 of 0x8000 are all 0. */
    uint8_t *qh_bytes = blk + 2 + 32;
    qh_bytes[0] = 0x00; qh_bytes[1] = 0x80;
    gguf_dequant_iq1_s(out, blk, 256);
    TEST_ASSERT_FLOAT_EQ(out[0],  -1.125f, 1e-5f, "iq1_s elem0 (delta sign flip, ib=0)");
    TEST_ASSERT_FLOAT_EQ(out[31], -1.125f, 1e-5f, "iq1_s elem31 (still ib=0's sub-block)");
    TEST_ASSERT_FLOAT_EQ(out[32], -0.875f, 1e-5f, "iq1_s elem32 (ib=1, unaffected)");
}

/* ── IQ1_M: no top-level d -- packed across 4 scale u16's top nibbles ──────── */
static void test_iq1_m_single_block(void) {
    uint8_t blk[56];
    memset(blk, 0, sizeof(blk));
    /* scales_bytes = blk[48..55], viewed as sc[0..3]. Craft sc[2]/sc[3] so the
     * packed scale.u16 = 0x3C00 = f32_to_fp16(1.0) (verified: sc[0] bits[15:12]
     * -> result[3:0]=0; sc[1] bits[11:8] -> result[7:4]=0; sc[2] bits[15:12]=0xC
     * -> result[11:8]=0xC; sc[3] bits[15:12]=0x3 -> result[15:12]=0x3). sc[0]/
     * sc[1] left at 0 also zeroes the low bits used for ib=0..3's dl1/dl2. */
    uint16_t sc2 = 0xC000, sc3 = 0x3000;
    memcpy(blk + 48 + 4, &sc2, 2);
    memcpy(blk + 48 + 6, &sc3, 2);

    float out[256];
    gguf_dequant_iq1_m(out, blk, 256);
    /* All-zero qs/qh -> idx=0 -> iq1s_grid[0]=all -1. dl=d*(2*0+1)=d=1 for
     * every ib (verified by hand that sc[2]/sc[3]'s low 12 bits are 0, so
     * ib=4..7's dl also come out to 1 despite the nonzero top nibbles).
     * delta=+IQ1S_DELTA -> dst = 1*(-1+0.125) = -0.875 uniform. */
    TEST_ASSERT_FLOAT_EQ(out[0],   -0.875f, 1e-5f, "iq1_m elem0 (all-zero baseline)");
    TEST_ASSERT_FLOAT_EQ(out[255], -0.875f, 1e-5f, "iq1_m elem255 (all-zero baseline)");

    /* qh[0] bit3 set -> delta[0] (l=0, dst[0..7]) flips sign; idx0 untouched
     * since (0x08<<8)&0x700 = 0x0800&0x0700 = 0. */
    blk[32] = 0x08;
    gguf_dequant_iq1_m(out, blk, 256);
    TEST_ASSERT_FLOAT_EQ(out[0], -1.125f, 1e-5f, "iq1_m elem0 (delta sign flip)");
    TEST_ASSERT_FLOAT_EQ(out[8], -0.875f, 1e-5f, "iq1_m elem8 (l=1, unaffected)");
}

/* ── BF16: top 16 bits of an IEEE-754 float32, no block/scale ──────────────── */
static void test_bf16_dequant(void) {
    /* 1.0f = 0x3F800000 -> top 16 bits = 0x3F80 */
    uint16_t in[3] = { 0x3F80, 0x0000, 0xBF80 }; /* 1.0, 0.0, -1.0 */
    float out[3];
    gguf_dequant_bf16(out, in, 3);
    TEST_ASSERT_FLOAT_EQ(out[0],  1.0f, 1e-5f, "bf16 1.0");
    TEST_ASSERT_FLOAT_EQ(out[1],  0.0f, 1e-5f, "bf16 0.0");
    TEST_ASSERT_FLOAT_EQ(out[2], -1.0f, 1e-5f, "bf16 -1.0");
}

/* ── IQ4_NL: block = [d fp16][qs[16]], "split-half" nibble packing ──────────
 * Regression test for a real bug caught by differential testing against
 * ggml's dequantize_row_iq4_nl (2026-09-18): this decoder previously used
 * interleaved-pair nibble packing (low->2i, high->2i+1) copied from the
 * Q4_1/Q4_0 layout, but IQ4_NL actually uses ggml's "split-half" layout —
 * for j in [0,16), qs[j]'s low nibble is element j and its high nibble is
 * element j+16 (same layout IQ4_XS already used correctly for the same
 * kvalues_iq4nl codebook). The bug produced a completely wrong element
 * permutation, not just a scale/sign error — the leading suspect for the
 * open degenerate-repeating-token bug (see docs/ai/mistakes.md). */
static void test_iq4_nl_single_block(void) {
    uint8_t blk[18];
    float d = 0.5f;
    uint16_t d_h = f32_to_fp16(d);
    memcpy(blk, &d_h, 2);
    /* qs[j] = (j << 4) | (15 - j): low nibble = 15-j (element j), high nibble = j (element j+16) */
    for (int j = 0; j < 16; j++) blk[2 + j] = (uint8_t)((j << 4) | (15 - j));

    float out[32];
    gguf_dequant_iq4_nl(out, blk, 32);

    /* kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} */
    TEST_ASSERT_FLOAT_EQ(out[0],  0.5f * 113.0f, 1e-4f, "iq4_nl elem0 (low nibble=15, split-half)");
    TEST_ASSERT_FLOAT_EQ(out[16], 0.5f * -127.0f, 1e-4f, "iq4_nl elem16 (high nibble=0, split-half)");
    TEST_ASSERT_FLOAT_EQ(out[15], 0.5f * -127.0f, 1e-4f, "iq4_nl elem15 (low nibble=0, split-half)");
    TEST_ASSERT_FLOAT_EQ(out[31], 0.5f * 113.0f, 1e-4f, "iq4_nl elem31 (high nibble=15, split-half)");
    TEST_ASSERT_FLOAT_EQ(out[8],  0.5f * -10.0f, 1e-4f, "iq4_nl elem8 (low nibble=7, split-half)");
}

int main(void) {
    RUN_TEST(test_bf16_dequant);
    RUN_TEST(test_iq4_nl_single_block);
    RUN_TEST(test_q4_1_single_block);
    RUN_TEST(test_q4_1_partial_trailing);
    RUN_TEST(test_q8_1_single_block);
    RUN_TEST(test_q8_k_single_block);
    RUN_TEST(test_iq4_xs_single_block);
    RUN_TEST(test_iq2_xxs_single_block);
    RUN_TEST(test_iq2_xs_single_block);
    RUN_TEST(test_iq2_s_single_block);
    RUN_TEST(test_iq3_xxs_single_block);
    RUN_TEST(test_iq3_s_single_block);
    RUN_TEST(test_iq1_s_single_block);
    RUN_TEST(test_iq1_m_single_block);
    TEST_SUMMARY();
}
