/* Bit-exact reference for every GGUF dequantiser. Each format decodes 1024 weights from
 * deterministic pseudo-random blocks (fp16/bf16/f32 scales kept finite) and the output bytes are
 * hashed; the hashes below were taken from the code before the dequantiser loops were refactored.
 * One differing bit anywhere fails the test. `test_gguf_dequant_golden --print` lists the current
 * hashes (for a deliberate update). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/fp16.h"
#include "core/gguf_quant.h"
#include "core/gguf_reader.h"
#include "test_harness.h"

enum { N_ELEMS = 1024 }; /* a multiple of every block size (32, 128, 256) */

typedef void (*Dq)(float *, const void *, size_t);

static void dq_f32(float *o, const void *d, size_t n) {
    memcpy(o, d, n * sizeof(float));
}
static void dq_f16(float *o, const void *d, size_t n) {
    const uint16_t *s = (const uint16_t *)d;
    for (size_t i = 0; i < n; i++) o[i] = tn_fp16_to_f32(s[i]);
}

static const struct {
    const char *name;
    GGUFType type;
    Dq fn;
    uint64_t golden;
} k_cases[] = {
    {"F32", GGUF_TYPE_F32, dq_f32, 0xfef18ca12047e57aull},
    {"F16", GGUF_TYPE_F16, dq_f16, 0x1551d91dc1c27333ull},
    {"BF16", GGUF_TYPE_BF16, gguf_dequant_bf16, 0x72baad0efce759f3ull},
    {"Q8_0", GGUF_TYPE_Q8_0, gguf_dequant_q8_0, 0x8612410128603f61ull},
    {"Q4_0", GGUF_TYPE_Q4_0, gguf_dequant_q4_0, 0x9319c581ffbc21fdull},
    {"Q4_1", GGUF_TYPE_Q4_1, gguf_dequant_q4_1, 0x56360804b110e230ull},
    {"Q5_0", GGUF_TYPE_Q5_0, gguf_dequant_q5_0, 0xcd1961be42c96353ull},
    {"Q5_1", GGUF_TYPE_Q5_1, gguf_dequant_q5_1, 0xf6e5aea1bd664f13ull},
    {"Q8_1", GGUF_TYPE_Q8_1, gguf_dequant_q8_1, 0x31a9cb29181090adull},
    {"Q2_0", GGUF_TYPE_Q2_0, gguf_dequant_q2_0, 0x1f5649230f83b765ull},
    {"Q2_K", GGUF_TYPE_Q2_K, gguf_dequant_q2_k, 0x0cd49083a7b375aaull},
    {"Q3_K", GGUF_TYPE_Q3_K, gguf_dequant_q3_k, 0xc9d49031d3df5a93ull},
    {"Q4_K", GGUF_TYPE_Q4_K, gguf_dequant_q4_k, 0x94610791a9fcf28eull},
    {"Q5_K", GGUF_TYPE_Q5_K, gguf_dequant_q5_k, 0x68d4bb9467fa3e8cull},
    {"Q6_K", GGUF_TYPE_Q6_K, gguf_dequant_q6_k, 0xc89a74c373b59912ull},
    {"Q8_K", GGUF_TYPE_Q8_K, gguf_dequant_q8_k, 0xf9281c98fe507d0dull},
    {"IQ4_NL", GGUF_TYPE_IQ4_NL, gguf_dequant_iq4_nl, 0x0906183ac4790926ull},
    {"IQ4_XS", GGUF_TYPE_IQ4_XS, gguf_dequant_iq4_xs, 0x48d4b31285a1e2a7ull},
    {"IQ2_XXS", GGUF_TYPE_IQ2_XXS, gguf_dequant_iq2_xxs, 0x5a4bd90975cf5637ull},
    {"IQ2_XS", GGUF_TYPE_IQ2_XS, gguf_dequant_iq2_xs, 0xce3bd24353bfa811ull},
    {"IQ2_S", GGUF_TYPE_IQ2_S, gguf_dequant_iq2_s, 0xcbfc1089ca700b51ull},
    {"IQ3_XXS", GGUF_TYPE_IQ3_XXS, gguf_dequant_iq3_xxs, 0x369676a59e3d02faull},
    {"IQ3_S", GGUF_TYPE_IQ3_S, gguf_dequant_iq3_s, 0xfdac760fbed31943ull},
    {"IQ1_S", GGUF_TYPE_IQ1_S, gguf_dequant_iq1_s, 0xd5ffeffeec31bf27ull},
    {"IQ1_M", GGUF_TYPE_IQ1_M, gguf_dequant_iq1_m, 0x29111beae89520fdull},
};
#define N_CASES (sizeof(k_cases) / sizeof(k_cases[0]))

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h       = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static uint64_t run_case(size_t idx) {
    const size_t bytes = (N_ELEMS / gguf_type_block_elems(k_cases[idx].type)) *
                         gguf_type_block_bytes(k_cases[idx].type);
    uint8_t *data = (uint8_t *)malloc(bytes + 64);
    float *out    = (float *)calloc(N_ELEMS, sizeof(float));
    uint32_t x    = 0x9E3779B9u + (uint32_t)idx * 7919u;
    for (size_t i = 0; i < bytes; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[i] = (uint8_t)(x >> 11);
        if (i & 1) data[i] &= 0x3F; /* keeps every 16-bit float and bf16 / f32 high half finite */
    }
    k_cases[idx].fn(out, data, N_ELEMS);
    uint64_t h = fnv1a(out, N_ELEMS * sizeof(float));
    free(data);
    free(out);
    return h;
}

/* Partial last block (40 weights = one block + 8): Q8_0 decodes the partial block, Q5_0, Q5_1,
 * Q4_1, Q8_1 and Q2_0 zero the weights past the last whole block, Q4_0 leaves them alone. The
 * output starts as a sentinel pattern so "left alone" is visible in the hash. */
static const struct {
    const char *name;
    GGUFType type;
    Dq fn;
    uint64_t golden;
} k_tail_cases[] = {
    {"Q8_0", GGUF_TYPE_Q8_0, gguf_dequant_q8_0, 0x51a05b9d15d9f0f5ull},
    {"Q4_0", GGUF_TYPE_Q4_0, gguf_dequant_q4_0, 0x4937c87c4c1bb069ull},
    {"Q4_1", GGUF_TYPE_Q4_1, gguf_dequant_q4_1, 0x5d025a1ba4f0f6d4ull},
    {"Q5_0", GGUF_TYPE_Q5_0, gguf_dequant_q5_0, 0x3d9ab310c6d459f8ull},
    {"Q5_1", GGUF_TYPE_Q5_1, gguf_dequant_q5_1, 0x898815e0907cd3c0ull},
    {"Q8_1", GGUF_TYPE_Q8_1, gguf_dequant_q8_1, 0x49e81c90dc77cd4eull},
    {"Q2_0", GGUF_TYPE_Q2_0, gguf_dequant_q2_0, 0xe69fafcd15fb0b03ull},
};
#define N_TAIL (sizeof(k_tail_cases) / sizeof(k_tail_cases[0]))

static uint64_t run_tail(size_t idx) {
    enum { N_TAIL_ELEMS = 40 };
    const size_t blocks = (N_TAIL_ELEMS + gguf_type_block_elems(k_tail_cases[idx].type) - 1) /
                          gguf_type_block_elems(k_tail_cases[idx].type);
    const size_t bytes = blocks * gguf_type_block_bytes(k_tail_cases[idx].type);
    uint8_t *data      = (uint8_t *)malloc(bytes + 64);
    float out[N_TAIL_ELEMS];
    uint32_t x = 0xC0FFEEu + (uint32_t)idx * 104729u;
    for (size_t i = 0; i < bytes; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[i] = (uint8_t)(x >> 11);
        if (i & 1) data[i] &= 0x3F;
    }
    for (int i = 0; i < N_TAIL_ELEMS; i++) out[i] = 123.5f; /* sentinel */
    k_tail_cases[idx].fn(out, data, N_TAIL_ELEMS);
    uint64_t h = fnv1a(out, sizeof(out));
    free(data);
    return h;
}

static void test_tail_blocks(void) {
    for (size_t i = 0; i < N_TAIL; i++) {
        uint64_t h = run_tail(i);
        if (h != k_tail_cases[i].golden)
            printf("  tail mismatch %s: got %016llx\n", k_tail_cases[i].name,
                   (unsigned long long)h);
        TEST_ASSERT(h == k_tail_cases[i].golden, "partial last block behaves as before");
    }
}

static void test_golden(void) {
    for (size_t i = 0; i < N_CASES; i++) {
        uint64_t h = run_case(i);
        if (h != k_cases[i].golden)
            printf("  mismatch %s: got %016llx\n", k_cases[i].name, (unsigned long long)h);
        TEST_ASSERT(h == k_cases[i].golden, "dequantiser output is bit-identical to the reference");
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--print") == 0) {
        for (size_t i = 0; i < N_CASES; i++)
            printf("%s %016llx\n", k_cases[i].name, (unsigned long long)run_case(i));
        for (size_t i = 0; i < N_TAIL; i++)
            printf("tail-%s %016llx\n", k_tail_cases[i].name, (unsigned long long)run_tail(i));
        return 0;
    }
    RUN_TEST(test_golden);
    RUN_TEST(test_tail_blocks);
    TEST_SUMMARY();
}
