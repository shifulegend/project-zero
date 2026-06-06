#include "test_harness.h"
#include "core/config.h"
#include "core/platform.h"
#include "core/run_state.h"
#include "memory/aligned_alloc.h"

static void test_config_read_valid(void) {
    /* Construct a synthetic mapped file blob */
    tn_u8 blob[512];
    memset(blob, 0, sizeof(blob));
    tn_u8 *p = blob;

    /* Magic */
    tn_u32 magic = TN_MAGIC;
    memcpy(p, &magic, sizeof(magic)); p += sizeof(magic);
    /* Version */
    tn_u32 version = TN_VERSION;
    memcpy(p, &version, sizeof(version)); p += sizeof(version);
    /* Config */
    Config cfg_in = {
        .dim = 256, .hidden_dim = 512, .n_layers = 4,
        .n_heads = 4, .n_kv_heads = 2, .vocab_size = 1000, .seq_len = 512
    };
    memcpy(p, &cfg_in, sizeof(Config));

    Config cfg_out;
    TernaryError err = config_read(&cfg_out, blob, sizeof(blob));
    TEST_ASSERT_EQ(err, TN_OK, "config_read returns OK");
    TEST_ASSERT_EQ(cfg_out.dim, 256, "dim matches");
    TEST_ASSERT_EQ(cfg_out.n_layers, 4, "n_layers matches");
    TEST_ASSERT_EQ(cfg_out.n_kv_heads, 2, "n_kv_heads matches");
    TEST_ASSERT_EQ(config_head_dim(&cfg_out), 64, "head_dim = 256/4 = 64");
    TEST_ASSERT_EQ(config_kv_dim(&cfg_out), 128, "kv_dim = 64*2 = 128");
}

static void test_config_bad_magic(void) {
    tn_u8 blob[512];
    memset(blob, 0, sizeof(blob));
    tn_u32 bad_magic = 0xDEADBEEF;
    memcpy(blob, &bad_magic, sizeof(bad_magic));

    Config cfg;
    TernaryError err = config_read(&cfg, blob, sizeof(blob));
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_MAGIC, "bad magic detected");
}

static void test_config_invalid_fields(void) {
    tn_u8 blob[512];
    memset(blob, 0, sizeof(blob));
    tn_u8 *p = blob;

    tn_u32 magic = TN_MAGIC;
    memcpy(p, &magic, sizeof(magic)); p += sizeof(magic);
    tn_u32 version = TN_VERSION;
    memcpy(p, &version, sizeof(version)); p += sizeof(version);

    /* Invalid: dim not divisible by n_heads */
    Config cfg_in = {
        .dim = 255, .hidden_dim = 512, .n_layers = 4,
        .n_heads = 4, .n_kv_heads = 2, .vocab_size = 1000, .seq_len = 512
    };
    memcpy(p, &cfg_in, sizeof(Config));

    Config cfg;
    TernaryError err = config_read(&cfg, blob, sizeof(blob));
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_CONFIG, "dim%n_heads!=0 rejected");
}

static void test_run_state_alloc_free(void) {
    Config cfg = {
        .dim = 64, .hidden_dim = 128, .n_layers = 2,
        .n_heads = 4, .n_kv_heads = 2, .vocab_size = 100, .seq_len = 256
    };
    RunState s;
    TernaryError err = run_state_alloc(&s, &cfg, 128);
    TEST_ASSERT_EQ(err, TN_OK, "run_state_alloc OK");
    TEST_ASSERT(s.x != NULL, "x allocated");
    TEST_ASSERT(s.key_cache != NULL, "key_cache allocated");
    TEST_ASSERT(s.value_cache != NULL, "value_cache allocated");
    TEST_ASSERT_EQ(s.max_seq_len, 128, "max_seq_len set");
    TEST_ASSERT_EQ(s.current_pos, 0, "current_pos is 0");

    /* Verify alignment */
    TEST_ASSERT(((uintptr_t)s.x % TN_SIMD_ALIGN) == 0, "x is aligned");
    TEST_ASSERT(((uintptr_t)s.key_cache % TN_SIMD_ALIGN) == 0, "key_cache aligned");

    /* Verify KV cache layout: contiguous positions for a single head */
    int head_dim = config_head_dim(&cfg);
    int idx0 = KV_CACHE_IDX(0, 0, 0, 0, cfg.n_kv_heads, 128, head_dim);
    int idx1 = KV_CACHE_IDX(0, 0, 1, 0, cfg.n_kv_heads, 128, head_dim);
    TEST_ASSERT_EQ(idx1 - idx0, head_dim, "positions are head_dim apart (contiguous)");

    run_state_free(&s);
    TEST_ASSERT(s.x == NULL, "x freed");
}

int main(void) {
    RUN_TEST(test_config_read_valid);
    RUN_TEST(test_config_bad_magic);
    RUN_TEST(test_config_invalid_fields);
    RUN_TEST(test_run_state_alloc_free);

    TEST_SUMMARY();
}
