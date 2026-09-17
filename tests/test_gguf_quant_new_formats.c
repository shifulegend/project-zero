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

int main(void) {
    RUN_TEST(test_q4_1_single_block);
    RUN_TEST(test_q4_1_partial_trailing);
    RUN_TEST(test_q8_1_single_block);
    RUN_TEST(test_q8_k_single_block);
    RUN_TEST(test_iq4_xs_single_block);
    TEST_SUMMARY();
}
