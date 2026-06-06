/**
 * test_sampling.c — Unit tests for Phase 7: Sampling & Text Generation
 *
 * Tests:
 *  1. RNG reproducibility and range
 *  2. Argmax sampler correctness
 *  3. Temperature scaling
 *  4. Top-p (nucleus) sampling
 *  5. Top-k sampling
 *  6. Sampling distribution properties
 */

#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "test_harness.h"

#include <math.h>
#include <string.h>

/* ================================================================
 * TEST: RNG (Phase 7.5)
 * ================================================================ */

static void test_rng_seed_and_range(void) {
  unsigned long long state;
  rng_seed(&state, 42);

  /* Generate many values and check they are in [0, 1) */
  for (int i = 0; i < 1000; i++) {
    float v = rng_float(&state);
    TEST_ASSERT(v >= 0.0f, "rng_float >= 0");
    TEST_ASSERT(v < 1.0f, "rng_float < 1");
  }
}

static void test_rng_reproducibility(void) {
  unsigned long long state1, state2;
  rng_seed(&state1, 12345);
  rng_seed(&state2, 12345);

  for (int i = 0; i < 100; i++) {
    float a = rng_float(&state1);
    float b = rng_float(&state2);
    TEST_ASSERT_FLOAT_EQ(a, b, 1e-9f, "same seed produces same sequence");
  }
}

static void test_rng_different_seeds(void) {
  unsigned long long state1, state2;
  rng_seed(&state1, 111);
  rng_seed(&state2, 222);

  /* Different seeds should produce different sequences */
  int differs = 0;
  for (int i = 0; i < 10; i++) {
    float a = rng_float(&state1);
    float b = rng_float(&state2);
    if (fabsf(a - b) > 1e-9f)
      differs = 1;
  }
  TEST_ASSERT(differs, "different seeds produce different sequences");
}

static void test_rng_zero_seed(void) {
  /* Zero seed should be handled (converted to non-zero) */
  unsigned long long state;
  rng_seed(&state, 0);
  float v = rng_float(&state);
  TEST_ASSERT(v >= 0.0f && v < 1.0f, "zero seed still produces valid output");
}

/* ================================================================
 * TEST: Argmax (Phase 7.1)
 * ================================================================ */

static void test_argmax_basic(void) {
  float logits[] = {1.0f, 3.0f, 2.0f, 0.5f};
  int result = sample_argmax(logits, 4);
  TEST_ASSERT_EQ(result, 1, "argmax finds index 1");
}

static void test_argmax_first_element(void) {
  float logits[] = {10.0f, 3.0f, 2.0f, 0.5f};
  int result = sample_argmax(logits, 4);
  TEST_ASSERT_EQ(result, 0, "argmax finds first element");
}

static void test_argmax_last_element(void) {
  float logits[] = {1.0f, 2.0f, 3.0f, 100.0f};
  int result = sample_argmax(logits, 4);
  TEST_ASSERT_EQ(result, 3, "argmax finds last element");
}

static void test_argmax_negative(void) {
  float logits[] = {-5.0f, -1.0f, -3.0f, -2.0f};
  int result = sample_argmax(logits, 4);
  TEST_ASSERT_EQ(result, 1, "argmax works with all negative logits");
}

static void test_argmax_single(void) {
  float logits[] = {42.0f};
  int result = sample_argmax(logits, 1);
  TEST_ASSERT_EQ(result, 0, "argmax with single element");
}

/* ================================================================
 * TEST: Temperature (Phase 7.2)
 * ================================================================ */

static void test_temperature_scaling(void) {
  float logits[] = {2.0f, 4.0f, 6.0f, 8.0f};
  float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};

  apply_temperature(logits, 4, 2.0f);

  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_FLOAT_EQ(logits[i], expected[i], 1e-6f, "temp=2 halves logits");
  }
}

static void test_temperature_one(void) {
  float logits[] = {1.0f, 2.0f, 3.0f};
  float original[] = {1.0f, 2.0f, 3.0f};

  apply_temperature(logits, 3, 1.0f);

  for (int i = 0; i < 3; i++) {
    TEST_ASSERT_FLOAT_EQ(logits[i], original[i], 1e-6f,
                         "temp=1 preserves logits");
  }
}

static void test_temperature_low(void) {
  /* Low temperature should sharpen the distribution */
  float logits[] = {1.0f, 2.0f, 3.0f};
  apply_temperature(logits, 3, 0.5f);

  TEST_ASSERT_FLOAT_EQ(logits[0], 2.0f, 1e-6f, "temp=0.5 doubles logits[0]");
  TEST_ASSERT_FLOAT_EQ(logits[1], 4.0f, 1e-6f, "temp=0.5 doubles logits[1]");
  TEST_ASSERT_FLOAT_EQ(logits[2], 6.0f, 1e-6f, "temp=0.5 doubles logits[2]");
}

static void test_temperature_preserves_argmax(void) {
  /* Temperature should not change which element is max */
  float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
  apply_temperature(logits, 4, 0.7f);
  int max_idx = sample_argmax(logits, 4);
  TEST_ASSERT_EQ(max_idx, 1, "temperature preserves argmax");
}

/* ================================================================
 * TEST: Top-p Sampling (Phase 7.3)
 * ================================================================ */

static void test_top_p_deterministic_peak(void) {
  /* With one overwhelmingly dominant logit and top_p close to 0,
   * sampling should always pick it */
  float logits[] = {-100.0f, 100.0f, -100.0f, -100.0f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  for (int trial = 0; trial < 20; trial++) {
    float tmp[4];
    memcpy(tmp, logits, sizeof(logits));
    int result = sample_top_p(tmp, 4, 0.5f, &rng);
    TEST_ASSERT_EQ(result, 1, "top_p picks dominant logit");
  }
}

static void test_top_p_full_distribution(void) {
  /* With top_p = 1.0, all tokens are candidates */
  float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  int seen[4] = {0};
  for (int trial = 0; trial < 200; trial++) {
    float tmp[4];
    memcpy(tmp, logits, sizeof(logits));
    int result = sample_top_p(tmp, 4, 1.0f, &rng);
    TEST_ASSERT(result >= 0 && result < 4, "top_p result in range");
    seen[result] = 1;
  }

  /* With uniform logits and enough trials, we should see all tokens */
  int total_seen = seen[0] + seen[1] + seen[2] + seen[3];
  TEST_ASSERT(total_seen >= 3,
              "top_p=1.0 explores most of uniform distribution");
}

/* ================================================================
 * TEST: Top-k Sampling (Phase 7.4)
 * ================================================================ */

static void test_top_k_deterministic(void) {
  /* k=1 should behave like argmax */
  float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  for (int trial = 0; trial < 20; trial++) {
    float tmp[4];
    memcpy(tmp, logits, sizeof(logits));
    int result = sample_top_k(tmp, 4, 1, &rng);
    TEST_ASSERT_EQ(result, 1, "top_k=1 is argmax");
  }
}

static void test_top_k_range(void) {
  /* All sampled tokens should be valid indices */
  float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  for (int trial = 0; trial < 100; trial++) {
    float tmp[8];
    memcpy(tmp, logits, sizeof(logits));
    int result = sample_top_k(tmp, 8, 3, &rng);
    TEST_ASSERT(result >= 0 && result < 8, "top_k result in valid range");
  }
}

static void test_top_k_restricts_candidates(void) {
  /* With k=2 on [1, 10, 100, 0.1], only indices 1 and 2 should be picked */
  float logits[] = {1.0f, 10.0f, 100.0f, 0.1f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  for (int trial = 0; trial < 50; trial++) {
    float tmp[4];
    memcpy(tmp, logits, sizeof(logits));
    int result = sample_top_k(tmp, 4, 2, &rng);
    TEST_ASSERT(result == 1 || result == 2, "top_k=2 only picks top 2");
  }
}

static void test_top_k_exceeds_vocab(void) {
  /* k > vocab_size should be clamped */
  float logits[] = {1.0f, 2.0f};
  unsigned long long rng;
  rng_seed(&rng, 42);

  int result = sample_top_k(logits, 2, 100, &rng);
  TEST_ASSERT(result >= 0 && result < 2, "top_k clamped to vocab_size");
}

/* ================================================================ */

int main(void) {
  RUN_TEST(test_rng_seed_and_range);
  RUN_TEST(test_rng_reproducibility);
  RUN_TEST(test_rng_different_seeds);
  RUN_TEST(test_rng_zero_seed);

  RUN_TEST(test_argmax_basic);
  RUN_TEST(test_argmax_first_element);
  RUN_TEST(test_argmax_last_element);
  RUN_TEST(test_argmax_negative);
  RUN_TEST(test_argmax_single);

  RUN_TEST(test_temperature_scaling);
  RUN_TEST(test_temperature_one);
  RUN_TEST(test_temperature_low);
  RUN_TEST(test_temperature_preserves_argmax);

  RUN_TEST(test_top_p_deterministic_peak);
  RUN_TEST(test_top_p_full_distribution);

  RUN_TEST(test_top_k_deterministic);
  RUN_TEST(test_top_k_range);
  RUN_TEST(test_top_k_restricts_candidates);
  RUN_TEST(test_top_k_exceeds_vocab);

  TEST_SUMMARY();
}
