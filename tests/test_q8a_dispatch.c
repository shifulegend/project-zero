/* parallel_matmul_q8a / _tokens pick their integer dot kernel and block size from a table. For
 * every supported weight format the result must be bit-identical to dotting each row against the
 * quantised activations directly, with and without a thread pool; other formats are refused. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"
#include "math/q8_0_act.h"
#include "test_harness.h"
#include "threading/thread_pool.h"

enum { N = 128, D = 8, TOKENS = 3 };

typedef float (*Dot)(const uint8_t *, const TnQ8_0ActBlock *, int, int);

static const struct {
    GGUFType type;
    Dot dot;
} k_formats[] = {
    {GGUF_TYPE_Q5_0, tn_dot_q5_0_q8_0},
    {GGUF_TYPE_Q5_1, tn_dot_q5_1_q8_0},
    {GGUF_TYPE_Q8_0, tn_dot_q8_0_q8_0},
};

static uint32_t g_rng = 2463534242u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static void check_format(GGUFType type, Dot dot, ThreadPool *tp) {
    const size_t block     = gguf_type_block_bytes(type);
    const int nb           = N / TN_Q8_0_BLOCK;
    const size_t row_bytes = (size_t)nb * block;
    uint8_t *w             = (uint8_t *)malloc(row_bytes * D);
    float *x               = (float *)malloc(sizeof(float) * N * TOKENS);
    float *got             = (float *)calloc((size_t)D * TOKENS, sizeof(float));
    TnQ8_0ActBlock *acts   = (TnQ8_0ActBlock *)malloc(sizeof(TnQ8_0ActBlock) * nb);
    for (size_t i = 0; i < row_bytes * D; i++) w[i] = (uint8_t)(next_u32() >> 9);
    /* fp16 scales of random bytes can be NaN or inf; pin each block's scale to a finite value. */
    for (size_t b = 0; b < (size_t)D * nb; b++) {
        w[b * block]     = 0x00;
        w[b * block + 1] = 0x3C; /* 1.0 in fp16 */
        if (type == GGUF_TYPE_Q5_1) {
            w[b * block + 2] = 0x00;
            w[b * block + 3] = 0x34; /* 0.25 */
        }
    }
    for (int i = 0; i < N * TOKENS; i++) x[i] = (float)((int)(next_u32() % 2001) - 1000) / 250.0f;

    TEST_ASSERT_EQ(parallel_matmul_q8a_tokens(got, x, w, (int)type, N, D, TOKENS, tp), 0, "tokens");
    int same = 1;
    for (int k = 0; k < TOKENS; k++) {
        tn_quantize_q8_0_act(acts, x + (size_t)k * N, nb);
        for (int r = 0; r < D; r++) {
            float want = dot(w + (size_t)r * row_bytes, acts, nb, tn_q8_0_act_vnni_ok());
            if (memcmp(&want, &got[(size_t)k * D + r], sizeof(float)) != 0) same = 0;
        }
    }
    TEST_ASSERT(same, "multi-token result equals the direct dots");

    float one[D];
    TEST_ASSERT_EQ(parallel_matmul_q8a(one, x, w, (int)type, N, D, tp), 0, "single token");
    TEST_ASSERT(memcmp(one, got, sizeof(one)) == 0, "single token equals token 0 of the batch");
    free(w);
    free(x);
    free(got);
    free(acts);
}

static void test_formats(void) {
    ThreadPool *pool = threadpool_create(3);
    for (size_t i = 0; i < sizeof(k_formats) / sizeof(k_formats[0]); i++) {
        check_format(k_formats[i].type, k_formats[i].dot, NULL);
        check_format(k_formats[i].type, k_formats[i].dot, pool);
    }
    threadpool_destroy(pool);
}

static void test_unsupported_refused(void) {
    float x[N]      = {0}, out[D];
    uint8_t w[4096] = {0};
    for (int type = 0; type < 70; type++) {
        if (type == GGUF_TYPE_Q5_0 || type == GGUF_TYPE_Q5_1 || type == GGUF_TYPE_Q8_0) continue;
        TEST_ASSERT_EQ(parallel_matmul_q8a(out, x, w, type, N, D, NULL), -1, "single token");
        TEST_ASSERT_EQ(parallel_matmul_q8a_tokens(out, x, w, type, N, D, 2, NULL), -1, "tokens");
    }
}

int main(void) {
    RUN_TEST(test_formats);
    RUN_TEST(test_unsupported_refused);
    TEST_SUMMARY();
}
