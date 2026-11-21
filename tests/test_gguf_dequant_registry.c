/* The tensor-to-float32 dispatch of the GGUF loaders is one table. For every tensor type the
 * result must equal calling that type's dequantiser directly (the old per-type switch), the byte
 * count must equal the old formulas, and types without a dequantiser must still be refused. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/fp16.h"
#include "core/gguf_quant.h"
#include "core/gguf_reader.h"
#include "runtime/gguf_load_util.h"
#include "runtime/gguf_loader.h"
#include "test_harness.h"

enum { N_ELEMS = 512 }; /* a multiple of every block size (32, 128, 256) */

typedef void (*Dq)(float *, const void *, size_t);

static void dq_f32(float *o, const void *d, size_t n) {
    memcpy(o, d, n * sizeof(float));
}
static void dq_f16(float *o, const void *d, size_t n) {
    const uint16_t *s = (const uint16_t *)d;
    for (size_t i = 0; i < n; i++) o[i] = tn_fp16_to_f32(s[i]);
}

/* The old switch, as data: type, oracle dequantiser, old bytes per block, old elems per block
 * (0 bytes = the old byte-count switch returned 0 for it). */
static const struct {
    GGUFType type;
    Dq fn;
    size_t old_bytes, old_elems;
} k_old[] = {
    {GGUF_TYPE_F32, dq_f32, 4, 1},
    {GGUF_TYPE_F16, dq_f16, 2, 1},
    {GGUF_TYPE_BF16, gguf_dequant_bf16, 2, 1},
    {GGUF_TYPE_Q8_0, gguf_dequant_q8_0, 34, 32},
    {GGUF_TYPE_Q4_K, gguf_dequant_q4_k, 144, 256},
    {GGUF_TYPE_Q4_0, gguf_dequant_q4_0, 18, 32},
    {GGUF_TYPE_Q5_0, gguf_dequant_q5_0, 22, 32},
    {GGUF_TYPE_Q5_1, gguf_dequant_q5_1, 24, 32},
    {GGUF_TYPE_Q5_K, gguf_dequant_q5_k, 176, 256},
    {GGUF_TYPE_Q6_K, gguf_dequant_q6_k, 210, 256},
    {GGUF_TYPE_Q2_K, gguf_dequant_q2_k, 84, 256},
    {GGUF_TYPE_Q3_K, gguf_dequant_q3_k, 110, 256},
    {GGUF_TYPE_IQ4_NL, gguf_dequant_iq4_nl, 18, 32},
    {GGUF_TYPE_Q2_0, gguf_dequant_q2_0, 0, 0}, /* old byte switch had no Q2_0 row (returned 0) */
    {GGUF_TYPE_Q4_1, gguf_dequant_q4_1, 20, 32},
    {GGUF_TYPE_Q8_1, gguf_dequant_q8_1, 36, 32},
    {GGUF_TYPE_Q8_K, gguf_dequant_q8_k, 292, 256},
    {GGUF_TYPE_IQ4_XS, gguf_dequant_iq4_xs, 136, 256},
    {GGUF_TYPE_IQ2_XXS, gguf_dequant_iq2_xxs, 66, 256},
    {GGUF_TYPE_IQ2_XS, gguf_dequant_iq2_xs, 74, 256},
    {GGUF_TYPE_IQ2_S, gguf_dequant_iq2_s, 82, 256},
    {GGUF_TYPE_IQ3_XXS, gguf_dequant_iq3_xxs, 98, 256},
    {GGUF_TYPE_IQ3_S, gguf_dequant_iq3_s, 110, 256},
    {GGUF_TYPE_IQ1_S, gguf_dequant_iq1_s, 50, 256},
    {GGUF_TYPE_IQ1_M, gguf_dequant_iq1_m, 56, 256},
};
#define N_OLD (sizeof(k_old) / sizeof(k_old[0]))

static int in_old(int type) {
    for (size_t i = 0; i < N_OLD; i++)
        if ((int)k_old[i].type == type) return 1;
    return 0;
}

static void fill_pseudo_random(uint8_t *p, size_t n, uint32_t seed) {
    uint32_t x = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        p[i] = (uint8_t)(x >> 11);
    }
}

static void test_quant_bytes_match_old(void) {
    for (int t = 0; t < 70; t++) {
        size_t got  = gguf_ld_quant_bytes((GGUFType)t, N_ELEMS);
        size_t want = 0;
        for (size_t i = 0; i < N_OLD; i++)
            if ((int)k_old[i].type == t && k_old[i].old_elems)
                want = (N_ELEMS / k_old[i].old_elems) * k_old[i].old_bytes;
        if (t == GGUF_TYPE_Q2_0) want = (N_ELEMS / 128) * 34; /* the fix: was 0 */
        TEST_ASSERT_EQ((long)got, (long)want, "byte count per type");
    }
}

static void test_tensor_to_f32_matches_direct_dequant(void) {
    GGUFWeightStore *store = gguf_ld_store_alloc();
    TEST_ASSERT(store != NULL, "store");
    for (size_t i = 0; i < N_OLD; i++) {
        size_t bytes  = gguf_ld_quant_bytes(k_old[i].type, N_ELEMS);
        uint8_t *data = (uint8_t *)calloc(bytes + 64, 1);
        float *want   = (float *)calloc(N_ELEMS, sizeof(float));
        fill_pseudo_random(data, bytes, 12345u + (uint32_t)i);
        GGUFTensor t;
        memset(&t, 0, sizeof(t));
        t.type = k_old[i].type;
        t.data = data;
        snprintf(t.name, sizeof(t.name), "tensor_%zu", i);
        float *got = gguf_ld_tensor_to_f32(&t, N_ELEMS, store);
        k_old[i].fn(want, data, N_ELEMS);
        TEST_ASSERT(got && memcmp(got, want, N_ELEMS * sizeof(float)) == 0,
                    "dispatch equals the type's own dequantiser");
        float *direct = (float *)calloc(N_ELEMS, sizeof(float));
        TEST_ASSERT_EQ(gguf_ld_dequant(k_old[i].type, direct, data, N_ELEMS), 0, "direct entry");
        TEST_ASSERT(memcmp(direct, want, N_ELEMS * sizeof(float)) == 0, "direct entry output");
        free(direct);
        free(data);
        free(want);
    }
    weights_free_gguf(store);
}

static void test_unsupported_types_refused(void) {
    GGUFWeightStore *store = gguf_ld_store_alloc();
    uint8_t data[64]       = {0};
    for (int type = 0; type < 70; type++) {
        if (in_old(type) || type == GGUF_TYPE_Q2_0) continue;
        GGUFTensor t;
        memset(&t, 0, sizeof(t));
        t.type = (GGUFType)type;
        t.data = data;
        float out[16];
        TEST_ASSERT(gguf_ld_tensor_to_f32(&t, 16, store) == NULL, "type without a dequantiser");
        TEST_ASSERT_EQ(gguf_ld_dequant((GGUFType)type, out, data, 16), -1, "direct entry refuses");
    }
    weights_free_gguf(store);
}

int main(void) {
    RUN_TEST(test_quant_bytes_match_old);
    RUN_TEST(test_tensor_to_f32_matches_direct_dequant);
    RUN_TEST(test_unsupported_types_refused);
    TEST_SUMMARY();
}
