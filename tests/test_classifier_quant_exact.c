/*
 * tests/test_classifier_quant_exact.c — weights_build_classifier_quant produces, bit for bit, the INT8
 * and INT4 tables and scales it did before its two block loops were merged: a hash over the output
 * for synthetic BF16 tables of several shapes (dimensions that are and are not multiples of the
 * quantisation block or of 2, an all-zero row, huge and tiny magnitudes). The hash was recorded from
 * the previous implementation. With no hardware profile initialised both formats are built.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"
#include "core/weights.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x6eccf39cd8e18d94ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static uint16_t bf16_of(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return (uint16_t)((bits + (((bits >> 16) & 1u) + 0x7FFFu)) >> 16);
}

/* A table with rows of different character: zero, tiny, huge, mixed-sign, one outlier per block. */
static void fill_table(uint16_t *t, int vocab, int dim) {
    for (int i = 0; i < vocab; i++) {
        for (int j = 0; j < dim; j++) {
            float v = ((float)rnd(2001) - 1000.0f) / 1000.0f;
            switch (i % 6) {
                case 0:
                    v = 0.0f;
                    break;
                case 1:
                    v *= 1e-6f;
                    break;
                case 2:
                    v *= 1e4f;
                    break;
                case 3:
                    v *= (j % 32 == 5) ? 50.0f : 1.0f;
                    break;
                default:
                    break;
            }
            t[(size_t)i * dim + j] = bf16_of(v);
        }
    }
}

static void hash_case(int vocab, int dim) {
    uint16_t *table = (uint16_t *)malloc((size_t)vocab * dim * sizeof(uint16_t));
    fill_table(table, vocab, dim);
    TransformerWeights w;
    Config cfg;
    memset(&w, 0, sizeof w);
    memset(&cfg, 0, sizeof cfg);
    cfg.dim        = dim;
    cfg.vocab_size = vocab;
    w.cls.w        = table;
    weights_build_classifier_quant(&w, &cfg);
    size_t n_blocks = ((size_t)dim + TN_CLS_QUANT_BLOCK - 1) / TN_CLS_QUANT_BLOCK;
    size_t i4_row   = ((size_t)dim + 1) / 2;
    TEST_ASSERT(w.cls.i8 && w.cls.i8_scales && w.cls.i4 && w.cls.i4_scales, "both formats built");
    if (w.cls.i8 && w.cls.i4) {
        mix(w.cls.i8, (size_t)vocab * dim);
        mix(w.cls.i8_scales, (size_t)vocab * n_blocks * sizeof(float));
        mix(w.cls.i4, (size_t)vocab * i4_row);
        mix(w.cls.i4_scales, (size_t)vocab * n_blocks * sizeof(float));
    }
    free(w.cls.i8);
    free(w.cls.i8_scales);
    free(w.cls.i4);
    free(w.cls.i4_scales);
    free(table);
}

static void test_exact(void) {
    static const int shapes[][2] = {{12, 32}, {12, 64},  {13, 33}, {7, 70},  {5, 1},   {6, 2},
                                    {9, 95},  {16, 128}, {3, 31},  {8, 100}, {1, 257}, {30, 96}};
    for (size_t k = 0; k < sizeof shapes / sizeof shapes[0]; k++)
        hash_case(shapes[k][0], shapes[k][1]);
    if (getenv("CLS_EXACT_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "INT8/INT4 tables match the recorded hash");
}

int main(void) {
    printf("=== classifier quantiser, exact ===\n");
    RUN_TEST(test_exact);
    TEST_SUMMARY();
}
