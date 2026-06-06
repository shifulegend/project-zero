/**
 * test_kv_cache.c — Unit tests for Phase 8: KV Cache Optimizations
 *
 * Tests:
 *  1. int8 compression/decompression roundtrip accuracy
 *  2. int4 compression/decompression roundtrip accuracy
 *  3. Zero vector handling
 *  4. Negative value handling
 *  5. Sliding window initialization and mapping
 *  6. Sliding window wraparound behavior
 *  7. System prompt pinning
 *  8. KV strategy selection based on RAM thresholds
 *  9. Strategy name lookup
 */

#include "kv_cache/kv_compress.h"
#include "kv_cache/kv_strategy.h"
#include "kv_cache/sliding_window.h"
#include "memory/aligned_alloc.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * TEST: int8 KV Compression (Phase 8.1)
 * ================================================================ */

static void test_kv_i8_roundtrip(void) {
  int dim = 16;
  float src[16] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f,  -6.0f, 7.0f,  -8.0f,
                   0.5f, -0.5f, 0.0f, 1.5f,  -1.5f, 2.5f,  -2.5f, 3.5f};

  CompressedKVSlot slot;
  slot.data = (tn_i8 *)calloc(dim, sizeof(tn_i8));

  kv_compress_to_8bit(src, &slot, dim);

  /* Scale should be max(|src|) / 127 = 8.0 / 127 */
  float expected_scale = 8.0f / 127.0f;
  TEST_ASSERT_FLOAT_EQ(slot.scale, expected_scale, 1e-5f, "i8 scale correct");

  float dst[16];
  kv_decompress_from_8bit(dst, &slot, dim);

  /* Roundtrip error should be small (within quantization error) */
  for (int i = 0; i < dim; i++) {
    float err = fabsf(dst[i] - src[i]);
    /* Max quantization error = scale / 2 = (8/127)/2 ≈ 0.031 */
    TEST_ASSERT(err < 0.1f, "i8 roundtrip error within tolerance");
  }

  free(slot.data);
}

static void test_kv_i8_zero_vector(void) {
  int dim = 8;
  float src[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  CompressedKVSlot slot;
  slot.data = (tn_i8 *)calloc(dim, sizeof(tn_i8));

  kv_compress_to_8bit(src, &slot, dim);

  TEST_ASSERT_FLOAT_EQ(slot.scale, 0.0f, 1e-9f, "i8 zero vector scale = 0");

  float dst[8];
  kv_decompress_from_8bit(dst, &slot, dim);
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT_FLOAT_EQ(dst[i], 0.0f, 1e-9f, "i8 zero roundtrip = 0");
  }

  free(slot.data);
}

static void test_kv_i8_negative_values(void) {
  int dim = 4;
  float src[4] = {-10.0f, -5.0f, -1.0f, -0.1f};

  CompressedKVSlot slot;
  slot.data = (tn_i8 *)calloc(dim, sizeof(tn_i8));

  kv_compress_to_8bit(src, &slot, dim);

  /* All quantized values should be negative */
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT(slot.data[i] <= 0, "i8 negative values quantize negative");
  }

  float dst[4];
  kv_decompress_from_8bit(dst, &slot, dim);
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT(dst[i] <= 0.0f, "i8 decompressed negatives are negative");
    float err = fabsf(dst[i] - src[i]);
    TEST_ASSERT(err < 0.2f, "i8 negative roundtrip within tolerance");
  }

  free(slot.data);
}

static void test_kv_i8_preserves_sign(void) {
  int dim = 6;
  float src[6] = {1.0f, -1.0f, 2.0f, -2.0f, 0.0f, 3.0f};

  CompressedKVSlot slot;
  slot.data = (tn_i8 *)calloc(dim, sizeof(tn_i8));
  kv_compress_to_8bit(src, &slot, dim);

  float dst[6];
  kv_decompress_from_8bit(dst, &slot, dim);

  TEST_ASSERT(dst[0] > 0.0f, "positive stays positive");
  TEST_ASSERT(dst[1] < 0.0f, "negative stays negative");
  TEST_ASSERT(dst[2] > 0.0f, "positive stays positive (2)");
  TEST_ASSERT(dst[3] < 0.0f, "negative stays negative (2)");

  free(slot.data);
}

/* ================================================================
 * TEST: int4 KV Compression (Phase 8.2)
 * ================================================================ */

static void test_kv_i4_roundtrip(void) {
  int dim = 8;
  float src[8] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f, -0.5f};

  int packed_len = (dim + 1) / 2;
  tn_u8 *packed = (tn_u8 *)calloc(packed_len, 1);
  float scale;

  kv_compress_to_4bit(src, packed, &scale, dim);

  /* Scale = max(|src|)/7 = 3.0/7 ≈ 0.4286 */
  TEST_ASSERT(scale > 0.0f, "i4 scale is positive");
  TEST_ASSERT_FLOAT_EQ(scale, 3.0f / 7.0f, 1e-5f, "i4 scale correct");

  float dst[8];
  kv_decompress_from_4bit(dst, packed, scale, dim);

  for (int i = 0; i < dim; i++) {
    float err = fabsf(dst[i] - src[i]);
    /* int4 has coarser quantization (7 levels), so larger error */
    TEST_ASSERT(err < 0.5f, "i4 roundtrip within tolerance");
  }

  free(packed);
}

static void test_kv_i4_zero_vector(void) {
  int dim = 4;
  float src[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int packed_len = (dim + 1) / 2;
  tn_u8 *packed = (tn_u8 *)calloc(packed_len, 1);
  float scale;

  kv_compress_to_4bit(src, packed, &scale, dim);

  TEST_ASSERT_FLOAT_EQ(scale, 0.0f, 1e-9f, "i4 zero scale = 0");

  float dst[4];
  kv_decompress_from_4bit(dst, packed, scale, dim);
  for (int i = 0; i < dim; i++) {
    TEST_ASSERT_FLOAT_EQ(dst[i], 0.0f, 1e-9f, "i4 zero roundtrip = 0");
  }

  free(packed);
}

static void test_kv_i4_odd_dimension(void) {
  /* Odd dim means the last byte has one valid nibble */
  int dim = 7;
  float src[7] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f};

  int packed_len = (dim + 1) / 2; /* = 4 bytes */
  tn_u8 *packed = (tn_u8 *)calloc(packed_len, 1);
  float scale;

  kv_compress_to_4bit(src, packed, &scale, dim);

  float dst[7];
  kv_decompress_from_4bit(dst, packed, scale, dim);

  for (int i = 0; i < dim; i++) {
    float err = fabsf(dst[i] - src[i]);
    TEST_ASSERT(err < 0.5f, "i4 odd dim roundtrip within tolerance");
  }

  free(packed);
}

static void test_kv_i4_preserves_sign(void) {
  int dim = 4;
  float src[4] = {5.0f, -5.0f, 3.0f, -3.0f};

  int packed_len = (dim + 1) / 2;
  tn_u8 *packed = (tn_u8 *)calloc(packed_len, 1);
  float scale;

  kv_compress_to_4bit(src, packed, &scale, dim);

  float dst[4];
  kv_decompress_from_4bit(dst, packed, scale, dim);

  TEST_ASSERT(dst[0] > 0.0f, "i4 positive stays positive");
  TEST_ASSERT(dst[1] < 0.0f, "i4 negative stays negative");
  TEST_ASSERT(dst[2] > 0.0f, "i4 positive stays positive (2)");
  TEST_ASSERT(dst[3] < 0.0f, "i4 negative stays negative (2)");

  free(packed);
}

/* ================================================================
 * TEST: Sliding Window (Phase 8.3)
 * ================================================================ */

static void test_sw_init(void) {
  SlidingWindow sw;
  sw_init(&sw, 64, 8);

  TEST_ASSERT_EQ(sw.window_size, 64, "window_size = 64");
  TEST_ASSERT_EQ(sw.system_prompt_len, 8, "system_prompt_len = 8");
  TEST_ASSERT_EQ(sw.write_head, 8, "write_head starts at spl");
  TEST_ASSERT(sw.wrapped == false, "not wrapped initially");
}

static void test_sw_pinned_positions(void) {
  SlidingWindow sw;
  sw_init(&sw, 32, 4);

  /* System prompt positions should map to themselves */
  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_EQ(sw_map_position(&sw, i), i, "pinned position is identity");
  }
}

static void test_sw_non_prompt_mapping(void) {
  SlidingWindow sw;
  sw_init(&sw, 16, 4);

  /* Positions 4..15 should map into [4, 16) */
  for (int i = 4; i < 16; i++) {
    int mapped = sw_map_position(&sw, i);
    TEST_ASSERT(mapped >= 4, "mapped >= spl");
    TEST_ASSERT(mapped < 16, "mapped < window_size");
  }
}

static void test_sw_wraparound(void) {
  SlidingWindow sw;
  sw_init(&sw, 8, 2); /* 6 circular slots */

  /* Advance 6 times to fill, then 1 more to wrap */
  for (int i = 0; i < 7; i++) {
    sw_advance(&sw);
  }
  TEST_ASSERT(sw.wrapped == true, "should wrap after filling circular region");

  /* Position well beyond window should still map validly */
  int mapped = sw_map_position(&sw, 100);
  TEST_ASSERT(mapped >= 2 && mapped < 8,
              "wrapped position maps to valid range");
}

static void test_sw_valid_count(void) {
  SlidingWindow sw;
  sw_init(&sw, 16, 4);

  /* Before filling anything past prompt */
  TEST_ASSERT_EQ(sw_valid_count(&sw, 1), 2,
                 "logical pos 1 (2 tokens) = 2 valid");
  TEST_ASSERT_EQ(sw_valid_count(&sw, 3), 4,
                 "logical pos 3 (4 tokens) = prompt filled");

  /* Partially filling circular region */
  TEST_ASSERT_EQ(sw_valid_count(&sw, 6), 7,
                 "logical pos 6 (7 tokens) = 4 pinned + 3 circular");

  /* Full window */
  TEST_ASSERT_EQ(sw_valid_count(&sw, 20), 16,
                 "past window capacity = window_size");
}

static void test_sw_no_prompt(void) {
  SlidingWindow sw;
  sw_init(&sw, 8, 0); /* No pinned slots */

  TEST_ASSERT_EQ(sw.write_head, 0, "write_head starts at 0 with no prompt");

  int mapped = sw_map_position(&sw, 5);
  TEST_ASSERT(mapped >= 0 && mapped < 8, "no-prompt mapping is valid");
}

/* ================================================================
 * TEST: KV Strategy Selection (Phase 8.4)
 * ================================================================ */

static void test_strategy_full_f32(void) {
  Config cfg = {.dim = 4096,
                .hidden_dim = 11008,
                .n_layers = 32,
                .n_heads = 32,
                .n_kv_heads = 8,
                .vocab_size = 32000,
                .seq_len = 4096};

  /* 36 GB free RAM → should choose full F32 (K-3 threshold: > GB_32 = 32 GB) */
  tn_i64 ram = (tn_i64)36 * 1024 * 1024 * 1024;
  KVStrategyResult r = select_kv_strategy(&cfg, ram);
  TEST_ASSERT(r.strategy == KV_FULL_F32, "36GB → full F32");
  TEST_ASSERT_EQ(r.max_seq_len, 4096, "full F32 uses full seq_len");
}

static void test_strategy_quant_i8(void) {
  Config cfg = {.dim = 4096,
                .hidden_dim = 11008,
                .n_layers = 32,
                .n_heads = 32,
                .n_kv_heads = 8,
                .vocab_size = 32000,
                .seq_len = 4096};

  /* 10 GB free RAM → should choose quantized I8 (K-3 threshold: GB_8 < ram ≤ GB_32) */
  tn_i64 ram = (tn_i64)10 * 1024 * 1024 * 1024;
  KVStrategyResult r = select_kv_strategy(&cfg, ram);
  TEST_ASSERT(r.strategy == KV_QUANT_I8, "10GB → quant I8");
  TEST_ASSERT_EQ(r.max_seq_len, 4096, "quant I8 uses full seq_len");
}

static void test_strategy_sliding_i8(void) {
  Config cfg = {.dim = 4096,
                .hidden_dim = 11008,
                .n_layers = 32,
                .n_heads = 32,
                .n_kv_heads = 8,
                .vocab_size = 32000,
                .seq_len = 4096};

  /* 6 GB free RAM → sliding window I8 (K-3 threshold: GB_5 < ram ≤ GB_8) */
  tn_i64 ram = (tn_i64)6 * 1024 * 1024 * 1024;
  KVStrategyResult r = select_kv_strategy(&cfg, ram);
  TEST_ASSERT(r.strategy == KV_SLIDING_I8, "6GB → sliding I8");
  TEST_ASSERT(r.max_seq_len <= 2048, "sliding I8 limits seq_len to window");
}

static void test_strategy_sliding_i4(void) {
  Config cfg = {.dim = 4096,
                .hidden_dim = 11008,
                .n_layers = 32,
                .n_heads = 32,
                .n_kv_heads = 8,
                .vocab_size = 32000,
                .seq_len = 4096};

  /* 512 MB free RAM → sliding window I4 */
  tn_i64 ram = (tn_i64)512 * 1024 * 1024;
  KVStrategyResult r = select_kv_strategy(&cfg, ram);
  TEST_ASSERT(r.strategy == KV_SLIDING_I4, "512MB → sliding I4");
  TEST_ASSERT(r.max_seq_len <= 1024, "sliding I4 limits to small window");
}

static void test_strategy_names(void) {
  TEST_ASSERT(strcmp(kv_strategy_name(KV_FULL_F32), "Full F32") == 0,
              "name: Full F32");
  TEST_ASSERT(strcmp(kv_strategy_name(KV_QUANT_I8), "Quantized I8") == 0,
              "name: Quantized I8");
  TEST_ASSERT(strcmp(kv_strategy_name(KV_SLIDING_I4), "Sliding Window I4") == 0,
              "name: Sliding Window I4");
}

static void test_strategy_short_seq_len(void) {
  /* Model with very short seq_len shouldn't be enlarged by sliding window */
  Config cfg = {.dim = 256,
                .hidden_dim = 512,
                .n_layers = 4,
                .n_heads = 4,
                .n_kv_heads = 2,
                .vocab_size = 1000,
                .seq_len = 128};

  tn_i64 ram = (tn_i64)2 * 1024 * 1024 * 1024;
  KVStrategyResult r = select_kv_strategy(&cfg, ram);
  TEST_ASSERT(r.max_seq_len <= 128, "short seq_len not enlarged");
}

static void test_get_free_ram(void) {
  tn_i64 ram = tn_get_free_ram();
  /* On Linux, this should return a positive value */
  TEST_ASSERT(ram > 0, "tn_get_free_ram returns positive on Linux");
  /* Sanity: should be at least 100 MB on any reasonable system */
  TEST_ASSERT(ram > 100 * 1024 * 1024, "at least 100MB available");
}

/* ================================================================ */

int main(void) {
  RUN_TEST(test_kv_i8_roundtrip);
  RUN_TEST(test_kv_i8_zero_vector);
  RUN_TEST(test_kv_i8_negative_values);
  RUN_TEST(test_kv_i8_preserves_sign);

  RUN_TEST(test_kv_i4_roundtrip);
  RUN_TEST(test_kv_i4_zero_vector);
  RUN_TEST(test_kv_i4_odd_dimension);
  RUN_TEST(test_kv_i4_preserves_sign);

  RUN_TEST(test_sw_init);
  RUN_TEST(test_sw_pinned_positions);
  RUN_TEST(test_sw_non_prompt_mapping);
  RUN_TEST(test_sw_wraparound);
  RUN_TEST(test_sw_valid_count);
  RUN_TEST(test_sw_no_prompt);

  RUN_TEST(test_strategy_full_f32);
  RUN_TEST(test_strategy_quant_i8);
  RUN_TEST(test_strategy_sliding_i8);
  RUN_TEST(test_strategy_sliding_i4);
  RUN_TEST(test_strategy_names);
  RUN_TEST(test_strategy_short_seq_len);
  RUN_TEST(test_get_free_ram);

  TEST_SUMMARY();
}
