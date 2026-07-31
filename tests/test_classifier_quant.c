/*
 * test_classifier_quant.c — correctness test for the classifier INT8/INT4
 * quantizer (weights_build_classifier_quant, src/core/weights.c) and its
 * consumers (parallel_matmul_i8/i4, src/math/parallel_matmul.c).
 *
 * There was previously ZERO test coverage for this path (grep across
 * tests/ turns up nothing besides a comment mentioning the field names).
 * Added 2026-07-31 alongside the block-wise rewrite: a GitHub issue (#27)
 * flagged the previous one-scale-per-ROW quantizer as "a mere casting"
 * that "completely ignores the whole issue of perplexity." This test
 * verifies both that the new block-wise quantizer is numerically correct
 * (reconstructed dot products match a full-precision reference within the
 * expected quantization error bound) and that it actually behaves
 * block-wise (per-block scales differ when block magnitudes differ, which
 * the old per-row scheme could never represent).
 */

#include "core/weights.h"
#include "core/config.h"
#include "math/parallel_matmul.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Reference dot product computed directly from the full-precision BF16
 * table -- independent of anything under test. */
static float ref_dot(const uint16_t *row, const float *x, int dim) {
    float acc = 0.0f;
    for (int j = 0; j < dim; j++) acc += bf16_to_f32(row[j]) * x[j];
    return acc;
}

/* Builds a synthetic classifier table with deliberately different
 * magnitudes per TN_CLS_QUANT_BLOCK-sized block within each row, so a
 * per-row scheme (one scale for the whole row) and a per-block scheme
 * produce measurably different reconstruction quality -- exactly the
 * defect GitHub issue #27 flagged. */
static void run_case(int vocab, int dim, const char *label) {
    TransformerWeights w;
    Config cfg;
    memset(&w, 0, sizeof(w));
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim = dim;
    cfg.vocab_size = vocab;
    w.wcls_is_ternary = false;

    uint16_t *table = (uint16_t *)malloc((size_t)vocab * (size_t)dim * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)dim * sizeof(float));
    float *ref = (float *)malloc((size_t)vocab * sizeof(float));

    for (int j = 0; j < dim; j++)
        x[j] = 0.5f + 0.1f * (float)((j * 7) % 13 - 6);

    for (int i = 0; i < vocab; i++) {
        for (int j = 0; j < dim; j++) {
            /* Each 32-element block gets a moderately different magnitude
             * (1x vs 4x) and all-positive-biased values within a block, so
             * blocks don't cancel against each other (avoiding a
             * catastrophic-cancellation reference sum, which would make
             * relative error meaningless regardless of quantizer quality)
             * while still giving a block-wise scale something real to
             * adapt to that a single row-wide scale cannot represent. */
            int block = j / 32;
            float magnitude = (block % 2 == 0) ? 1.0f : 4.0f;
            float v = magnitude * (1.0f + 0.3f * (float)(((i * 31 + j * 17) % 21) - 10) / 10.0f);
            table[(size_t)i * dim + j] = f32_to_bf16(v);
        }
        ref[i] = ref_dot(table + (size_t)i * dim, x, dim);
    }
    w.wcls = table;

    weights_build_classifier_quant(&w, &cfg);
    TEST_ASSERT(w.wcls_i8 != NULL && w.wcls_i8_scales != NULL, "INT8 classifier built");
    TEST_ASSERT(w.wcls_i4 != NULL && w.wcls_i4_scales != NULL, "INT4 classifier built");

    int n_blocks = (dim + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;

    /* Per-block scales must actually differ within a row given the
     * deliberately alternating block magnitudes above -- this is the
     * direct, observable signature of the block-wise fix (the old
     * per-row scheme structurally cannot represent this: one scale for
     * the whole row, no matter how block magnitudes vary). */
    if (n_blocks >= 2) {
        float s0 = w.wcls_i8_scales[0];
        float s1 = w.wcls_i8_scales[1];
        float ratio = (s0 > s1) ? s0 / s1 : s1 / s0;
        char msg[160];
        snprintf(msg, sizeof(msg), "%s: block scales differ (s0=%.6f s1=%.6f ratio=%.3f)",
                 label, s0, s1, ratio);
        TEST_ASSERT(s0 > 0.0f && s1 > 0.0f && ratio > 2.0f, msg);
    }

    float *out_i8 = (float *)malloc((size_t)vocab * sizeof(float));
    float *out_i4 = (float *)malloc((size_t)vocab * sizeof(float));
    parallel_matmul_i8(out_i8, x, w.wcls_i8, w.wcls_i8_scales, dim, vocab, NULL);
    parallel_matmul_i4(out_i4, x, w.wcls_i4, w.wcls_i4_scales, dim, vocab, NULL);

    for (int i = 0; i < vocab; i++) {
        /* INT8: ~1/127 per-block relative precision; INT4: ~1/7. Generous
         * bounds (not tight to the theoretical minimum) since this is a
         * correctness check, not a precision benchmark. Reference sums
         * here are all-positive-biased (no cancellation), so relative
         * error is a meaningful metric, unlike the earlier cancellation-
         * prone version of this test. */
        float scale = fabsf(ref[i]) > 1.0f ? fabsf(ref[i]) : 1.0f;
        float err_i8 = fabsf(out_i8[i] - ref[i]) / scale;
        float err_i4 = fabsf(out_i4[i] - ref[i]) / scale;
        char msg8[160], msg4[160];
        snprintf(msg8, sizeof(msg8), "%s INT8 row %d: got=%.6f ref=%.6f rel_err=%.6f",
                 label, i, out_i8[i], ref[i], err_i8);
        snprintf(msg4, sizeof(msg4), "%s INT4 row %d: got=%.6f ref=%.6f rel_err=%.6f",
                 label, i, out_i4[i], ref[i], err_i4);
        TEST_ASSERT(err_i8 < 0.05f, msg8);
        TEST_ASSERT(err_i4 < 0.30f, msg4);
    }

    free(table); free(x); free(ref); free(out_i8); free(out_i4);
    free(w.wcls_i8); free(w.wcls_i8_scales); free(w.wcls_i4); free(w.wcls_i4_scales);
    (void)n_blocks;
}

static void test_dim_multiple_of_block(void) { run_case(6, 128, "dim=128 (4 blocks, exact)"); }
static void test_dim_not_multiple_of_block(void) { run_case(5, 100, "dim=100 (tail block, 4 full + 4 partial)"); }
static void test_single_block(void) { run_case(4, 20, "dim=20 (1 partial block)"); }

int main(void) {
    RUN_TEST(test_dim_multiple_of_block);
    RUN_TEST(test_dim_not_multiple_of_block);
    RUN_TEST(test_single_block);
    TEST_SUMMARY();
}
