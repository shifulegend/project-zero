/*
 * tests/test_gguf_i2s.c — BitNet I2_S support and RoPE pair layouts:
 *  - I2_S (bitnet.cpp layout) -> engine packed ternary repack is lossless,
 *    keeps the tensor scale, and gives the same matmul as the float weights;
 *  - malformed I2_S (code 3, columns not a multiple of 128, NaN scale) refused;
 *  - NEOX rotary equals NORM rotary on the de-interleaved vector, and the
 *    architecture -> layout table follows llama.cpp.
 */
#include "test_harness.h"
#include "core/gguf_i2s.h"
#include "core/unpack.h"
#include "math/ternary_matmul_packed.h"
#include "math/rope.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 12345u;
static int next_tern(void) { rng = rng * 1664525u + 1013904223u; return (int)((rng >> 16) % 3u) - 1; }

static void test_repack_roundtrip(void) {
    const int d = 6, n = 256;
    size_t nel = (size_t)d * n;
    int8_t *vals = (int8_t *)malloc(nel);
    uint8_t *i2s = (uint8_t *)malloc(tn_i2s_tensor_bytes(nel));
    uint8_t *pk = (uint8_t *)malloc((size_t)d * packed_bytes(n));
    for (size_t i = 0; i < nel; i++) vals[i] = (int8_t)next_tern();
    tn_i2s_pack_reference(i2s, vals, nel, 0.75f);
    float sc = 0.0f;
    TEST_ASSERT_EQ(tn_i2s_repack(pk, &sc, i2s, d, n), 0, "repack ok");
    TEST_ASSERT_FLOAT_EQ(sc, 0.75f, 1e-7f, "tensor scale preserved");
    int same = 1;
    for (int r = 0; r < d; r++)
        for (int j = 0; j < n; j++)
            if (unpack_ternary(pk + (size_t)r * packed_bytes(n), j) != vals[(size_t)r * n + j]) same = 0;
    TEST_ASSERT(same, "every weight survives the repack (row-major, packed_bytes rows)");

    /* Matmul on the repacked weights == float reference sum_j w*x * scale. */
    float x[256], out[6], ref[6];
    for (int j = 0; j < n; j++) x[j] = 0.01f * (float)((j * 7) % 23) - 0.1f;
    ternary_matmul_packed(out, x, pk, n, d, &sc, 0);
    int close = 1;
    for (int r = 0; r < d; r++) {
        double a = 0.0;
        for (int j = 0; j < n; j++) a += (double)vals[(size_t)r * n + j] * x[j];
        ref[r] = (float)(a * sc);
        if (fabsf(ref[r] - out[r]) > 1e-4f) close = 0;
    }
    TEST_ASSERT(close, "ternary matmul on repacked I2_S matches float reference");
    free(vals); free(i2s); free(pk);
}

static void test_repack_rejects(void) {
    uint8_t src[128 / 4 + 32];
    uint8_t dst[128];
    float sc;
    int8_t vals[128] = {0};
    tn_i2s_pack_reference(src, vals, 128, 1.0f);
    TEST_ASSERT_EQ(tn_i2s_repack(dst, &sc, src, 1, 128), 0, "valid single block");
    TEST_ASSERT_EQ(tn_i2s_repack(dst, &sc, src, 2, 64), -1, "columns not multiple of 128");
    src[5] = 0xFF;                                    /* code 3 = invalid */
    TEST_ASSERT_EQ(tn_i2s_repack(dst, &sc, src, 1, 128), -1, "invalid code 3 refused");
    tn_i2s_pack_reference(src, vals, 128, NAN);
    TEST_ASSERT_EQ(tn_i2s_repack(dst, &sc, src, 1, 128), -1, "non-finite scale refused");
    TEST_ASSERT_EQ((int)tn_i2s_tensor_bytes(2560 * 640), 2560 * 640 / 4 + 32, "tensor size formula");
}

static void test_rope_neox(void) {
    enum { HD = 8 };
    float freq[HD / 2];
    rope_precompute_freqs(freq, HD, 10000.0f);
    float v[HD], perm[HD], kk[HD], kp[HD];
    for (int i = 0; i < HD; i++) v[i] = (float)(i + 1) * 0.3f - 1.0f;
    /* de-interleave: perm[2i] = v[i], perm[2i+1] = v[i + HD/2] */
    for (int i = 0; i < HD / 2; i++) { perm[2 * i] = v[i]; perm[2 * i + 1] = v[i + HD / 2]; }
    memcpy(kk, v, sizeof(v));
    memcpy(kp, perm, sizeof(perm));
    const float corr[2] = { 0.0f, 0.0f };
    apply_rope_ex(v, kk, freq, HD, 7, 1, 1, 1.0f, 0.0f, 1.0f, corr, ROPE_LAYOUT_NEOX);
    apply_rope_ex(perm, kp, freq, HD, 7, 1, 1, 1.0f, 0.0f, 1.0f, corr, ROPE_LAYOUT_NORM);
    int ok = 1;
    for (int i = 0; i < HD / 2; i++)
        if (fabsf(v[i] - perm[2 * i]) > 1e-6f || fabsf(v[i + HD / 2] - perm[2 * i + 1]) > 1e-6f) ok = 0;
    TEST_ASSERT(ok, "NEOX == NORM on the de-interleaved vector");
    float n2a = 0, n2b = 0;
    for (int i = 0; i < HD; i++) { n2a += kk[i] * kk[i]; }
    for (int i = 0; i < HD; i++) { float t = (float)(i + 1) * 0.3f - 1.0f; n2b += t * t; }
    TEST_ASSERT_FLOAT_EQ(n2a, n2b, 1e-4f, "rotation preserves the norm");

    TEST_ASSERT_EQ(rope_layout_for_arch("llama"), ROPE_LAYOUT_NORM, "llama -> norm");
    TEST_ASSERT_EQ(rope_layout_for_arch("deepseek2"), ROPE_LAYOUT_NORM, "deepseek2 -> norm");
    TEST_ASSERT_EQ(rope_layout_for_arch("bitnet-b1.58"), ROPE_LAYOUT_NEOX, "bitnet-b1.58 -> neox");
    TEST_ASSERT_EQ(rope_layout_for_arch("qwen3"), ROPE_LAYOUT_NEOX, "qwen3 -> neox");
    TEST_ASSERT_EQ(rope_layout_for_arch(NULL), ROPE_LAYOUT_NORM, "unknown -> norm");
    TEST_ASSERT_EQ(rope_layout_parse("neox"), ROPE_LAYOUT_NEOX, "parse neox");
    TEST_ASSERT_EQ(rope_layout_parse("x"), -1, "parse invalid");
}

int main(void) {
    printf("=== I2_S + RoPE layout Tests ===\n");
    RUN_TEST(test_repack_roundtrip);
    RUN_TEST(test_repack_rejects);
    RUN_TEST(test_rope_neox);
    TEST_SUMMARY();
}
