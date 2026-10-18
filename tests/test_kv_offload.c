/*
 * tests/test_kv_offload.c — Phase 31: disk-backed memory (file_backed) and
 * the --kv-offload KV cache: zeroed mapping, read/write, free-space probe,
 * run_state carving for F32 and int8 KV, disk-budgeted context sizing.
 * Skips gracefully when no writable temp directory supports mmap.
 */
#include "test_harness.h"
#include "memory/file_backed.h"
#include "core/run_state.h"
#include "kv_cache/kv_strategy.h"
#include "kv_cache/kv_quant_ops.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char *tmp_dir(void) {
    const char *d = getenv("TMPDIR");
    return (d && *d) ? d : "/tmp";
}

static int g_skip;

static void test_file_backed(void) {
    FileBacked fb;
    TernaryError e = file_backed_alloc(&fb, tmp_dir(), (size_t)8 << 20);
    if (e == TN_ERR_UNSUPPORTED || e == TN_ERR_FILE_OPEN) {
        printf("  [SKIP] no disk-backed mapping available in %s\n", tmp_dir());
        g_skip = 1;
        return;
    }
    TEST_ASSERT_EQ(e, TN_OK, "allocate 8 MiB disk-backed");
    unsigned char *p = (unsigned char *)fb.base;
    int zero = 1;
    for (size_t i = 0; i < fb.bytes; i += 4096) if (p[i]) zero = 0;
    TEST_ASSERT(zero, "mapping starts zeroed");
    for (size_t i = 0; i < fb.bytes; i++) p[i] = (unsigned char)(i * 31u);
    file_backed_prefetch(p + 12345, 100000);
    int same = 1;
    for (size_t i = 0; i < fb.bytes; i++) if (p[i] != (unsigned char)(i * 31u)) { same = 0; break; }
    TEST_ASSERT(same, "contents read back");
    file_backed_free(&fb);
    TEST_ASSERT(fb.base == NULL && fb.bytes == 0, "free clears the handle");

    TEST_ASSERT(file_backed_free_space(tmp_dir()) > 0, "free space probe");
    TEST_ASSERT(file_backed_free_space("/nonexistent/pz-dir") < 0, "missing dir -> -1");
    TEST_ASSERT(file_backed_alloc(&fb, "/nonexistent/pz-dir", 4096) != TN_OK, "missing dir rejected");
    TEST_ASSERT(file_backed_alloc(&fb, tmp_dir(), 0) != TN_OK, "zero size rejected");
}

static Config tiny_cfg(void) {
    Config c;
    memset(&c, 0, sizeof(c));
    c.dim = 64; c.hidden_dim = 128; c.n_layers = 3; c.n_heads = 4; c.n_kv_heads = 2;
    c.vocab_size = 32; c.seq_len = 256; c.rope_theta = 10000.0f;
    return c;
}

static int inside(const RunState *s, const void *p) {
    uintptr_t b = (uintptr_t)s->kv_offload_base;
    return (uintptr_t)p >= b && (uintptr_t)p < b + s->kv_offload_bytes && ((uintptr_t)p % 64) == 0;
}

static void test_run_state_offload(void) {
    if (g_skip) return;
    Config c = tiny_cfg();
    KVStrategyOptions o = {0, 0.0f, 0, tmp_dir()};
    kv_strategy_set_options(&o);
    RunState s;
    TEST_ASSERT_EQ(run_state_alloc(&s, &c, 128), TN_OK, "F32 KV on disk");
    TEST_ASSERT(s.kv_offload_base != NULL, "mapping recorded");
    TEST_ASSERT(inside(&s, s.key_cache) && inside(&s, s.value_cache), "K/V carved, 64-B aligned");
    size_t n = (size_t)c.n_layers * c.n_kv_heads * 128 * (c.dim / c.n_heads);
    s.key_cache[n - 1] = 3.5f;
    s.value_cache[0] = -1.25f;
    TEST_ASSERT(s.key_cache[n - 1] == 3.5f && s.value_cache[0] == -1.25f, "K/V writable end to end");
    TEST_ASSERT(s.value_cache[n - 1] == 0.0f, "V starts zeroed");
    run_state_kv_prefetch_layer(&s, &c, 2, 100);
    run_state_kv_prefetch_layer(&s, &c, 3, 100); /* past the last layer: no-op */
    run_state_free(&s);
    TEST_ASSERT(s.kv_offload_base == NULL && s.key_cache == NULL, "freed");

    KVStrategyOptions o8 = {0, 0.0f, KV_TYPE_I8, tmp_dir()};
    kv_strategy_set_options(&o8);
    TEST_ASSERT_EQ(run_state_alloc(&s, &c, 128), TN_OK, "int8 KV on disk");
    TEST_ASSERT(inside(&s, s.key_cache_q) && inside(&s, s.value_cache_q) &&
                inside(&s, s.key_scale) && inside(&s, s.value_scale), "int8 buffers carved");
    TEST_ASSERT(s.key_cache == NULL, "no F32 cache with int8");
    run_state_kv_prefetch_layer(&s, &c, 1, 64);
    run_state_free(&s);

    kv_strategy_set_options(NULL);
    TEST_ASSERT_EQ(run_state_alloc(&s, &c, 128), TN_OK, "RAM KV still default");
    TEST_ASSERT(s.kv_offload_base == NULL && s.key_cache != NULL, "heap path unchanged");
    run_state_free(&s);
}

static void test_disk_budget(void) {
    if (g_skip) return;
    Config c = tiny_cfg();
    c.n_layers = 32; c.n_kv_heads = 8; c.dim = 4096; c.n_heads = 32; c.seq_len = 1 << 20;
    tn_i64 ram = (tn_i64)1 << 30; /* 1 GiB "free RAM" -> F32 tier caps hard */
    KVStrategyOptions ctx = {1 << 20, 0.0f, 0, NULL};
    KVStrategyResult r_ram = select_kv_strategy_ex(&c, ram, NULL, &ctx);
    ctx.offload_dir = tmp_dir();
    KVStrategyResult r_disk = select_kv_strategy_ex(&c, ram, NULL, &ctx);
    tn_i64 disk = file_backed_free_space(tmp_dir());
    TEST_ASSERT(r_ram.max_seq_len >= 512, "RAM cap keeps >= 512");
    if (disk > ram)
        TEST_ASSERT(r_disk.max_seq_len > r_ram.max_seq_len, "disk budget allows a longer context");
    else
        TEST_ASSERT(r_disk.max_seq_len >= 512, "disk budget sane on a small disk");
}

int main(void) {
    printf("=== KV Offload / Disk-Backed Memory (Phase 31) Tests ===\n");
    RUN_TEST(test_file_backed);
    RUN_TEST(test_run_state_offload);
    RUN_TEST(test_disk_budget);
    TEST_SUMMARY();
}
