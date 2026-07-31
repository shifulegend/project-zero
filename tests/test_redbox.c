#include "core/config.h"
#include "core/error.h"
#include "core/platform.h"
#include "core/run_state.h"
#include "memory/aligned_alloc.h"
#include "test_harness.h"
#include <stdint.h>
#include <string.h>

const char *__asan_default_options(void) { return "allocator_may_return_null=1"; }

/**
 * Redbox Test: RB-MEM-01 (SIMD Alignment Attacks)
 * Validates that odd-size allocations still enforce strict SIMD boundaries.
 */
static void rb_mem_01_simd_alignment(void) {
  /* 33 is highly irregular; testing 64-byte alignment enforcement */
  void *ptr = tn_aligned_alloc(33, 64);
  TEST_ASSERT(ptr != NULL, "Allocated odd size safely");
  TEST_ASSERT(((uintptr_t)ptr % 64) == 0,
              "Address is strictly 64-byte aligned (TN_SIMD_ALIGN)");
  if (ptr)
    tn_aligned_free(ptr);
}

/**
 * Redbox Test: RB-MEM-02 (OOM Resistance via aligned_calloc)
 * Validates memory allocation boundaries don't abruptly OS trap on integer
 * overflow.
 */
static void rb_mem_02_oom_resistance(void) {
  printf("[TRACKER redbox] Starting rb_mem_02_oom_resistance...\n");
  fflush(stdout);

  Config cfg = {.dim = 4096,
                .hidden_dim = 11008,
                .n_layers = 120,
                .n_heads = 32,
                .n_kv_heads = 8,
                .vocab_size = 32000,
                .seq_len = 8192};
  RunState s;

  /* Requesting INT_MAX token context triggers a massive ~900TB layout.
     This absolutely must fail either at size multiplication or memory map,
     preventing the OS from attempting a 100GB memset zero-loop. */
  TernaryError err = run_state_alloc(&s, &cfg, 2147483647);

  printf("[TRACKER redbox] Finished run_state_alloc, err=%d\n", err);
  fflush(stdout);

  TEST_ASSERT(err == TN_ERR_OOM,
              "run_state_alloc traps absurd OOM allocation bounds gracefully");
  if (err == TN_OK)
    run_state_free(&s);
}

/**
 * Redbox Test: RB-MEM-03 (Integer Overflow during Allocation)
 */
static void rb_mem_03_integer_overflow(void) {
  size_t result = 0;
  size_t max = (size_t)-1;

  int overflow = tn_size_mul_overflow(max, 2, &result);
  TEST_ASSERT(overflow == 1, "tn_size_mul_overflow correctly flags overflow");

  overflow = tn_size_mul_overflow(100, 100, &result);
  TEST_ASSERT(overflow == 0, "tn_size_mul_overflow allows safe sizes");
  TEST_ASSERT(result == 10000,
              "tn_size_mul_overflow calculates correct sizing");

  overflow = tn_size_mul4(max, max, max, max, &result);
  TEST_ASSERT(overflow == 1,
              "tn_size_mul4 catches chained 4-way multiplication overflows");
}

/**
 * Redbox Test: RB-IO-01 (Invalid Config Traversal/Injection)
 */
static void rb_io_01_invalid_config(void) {
  tn_u8 blob[512];
  memset(blob, 0, sizeof(blob));
  tn_u8 *p = blob;

  tn_u32 magic = TN_MAGIC;
  memcpy(p, &magic, sizeof(magic));
  p += sizeof(magic);
  tn_u32 version = TN_VERSION;
  memcpy(p, &version, sizeof(version));
  p += sizeof(version);

  /* Construct adversarial corrupted fields. (0 dims, negative layers) */
  Config cfg_in = {.dim = 0,
                   .hidden_dim = -400,
                   .n_layers = -1,
                   .n_heads = 0,
                   .n_kv_heads = 0,
                   .vocab_size = 0,
                   .seq_len = -500};
  memcpy(p, &cfg_in, sizeof(Config));

  Config cfg_out;
  TernaryError err = config_read(&cfg_out, blob, sizeof(blob));
  TEST_ASSERT(
      err == TN_ERR_INVALID_CONFIG,
      "config_read natively rejects structurally broken adversarial configs");
}

int main(void) {
  RUN_TEST(rb_mem_01_simd_alignment);
  RUN_TEST(rb_mem_02_oom_resistance);
  RUN_TEST(rb_mem_03_integer_overflow);
  RUN_TEST(rb_io_01_invalid_config);
  TEST_SUMMARY();
}
