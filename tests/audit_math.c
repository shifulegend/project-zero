#include "core/config.h"
#include "core/platform.h"
#include "math/rmsnorm.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "math/softmax.h"
#include "memory/aligned_alloc.h"
#include "test_harness.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Audit Test: Math-01 (Softmax Extreme Values)
 * Verifies that softmax doesn't produce NaNs when given highly negative
 * or highly positive numbers (due to max_val subtraction).
 */
static void aud_math_softmax_extreme(void) {
  float x[8] = {1e10f, 1e10f, 10.0f, -1e10f, -1e10f, 0.0f, 0.0f, 0.0f};

  /* Expected: The two 1e10f values should dominate and get ~0.5 each.
     Without max_val subtraction, exp(1e10f) is +Inf -> Softmax becomes NaN. */
  tn_softmax(x, 8);

  for (int i = 0; i < 8; i++) {
    TEST_ASSERT(!isnan(x[i]), "Softmax output is not NaN");
  }
  TEST_ASSERT_FLOAT_EQ(x[0], 0.5f, 1e-4f, "Dominant value 1 gets ~0.5");
  TEST_ASSERT_FLOAT_EQ(x[1], 0.5f, 1e-4f, "Dominant value 2 gets ~0.5");
  TEST_ASSERT_FLOAT_EQ(x[2], 0.0f, 1e-4f, "Small values get ~0.0");
  TEST_ASSERT_FLOAT_EQ(x[3], 0.0f, 1e-4f, "Very negative values get ~0.0");
}

/**
 * Audit Test: Math-02 (RMSNorm Zero/Subnormal Input)
 * Verifies that RMSNorm doesn't produce NaNs (div by zero) when input is all
 * zeros, relying on the 1e-5f epsilon.
 */
static void aud_math_rmsnorm_zero(void) {
  float x[8] = {0};
  float weight[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  float out[8] = {1, 1, 1, 1, 1, 1, 1, 1}; // Should be overwritten with 0

  tn_rmsnorm(out, x, weight, 8, 1e-5f);

  for (int i = 0; i < 8; i++) {
    TEST_ASSERT(!isnan(out[i]), "RMSNorm output is not NaN");
    TEST_ASSERT_FLOAT_EQ(out[i], 0.0f, 1e-6f, "Zero input gives zero output");
  }
}

/**
 * Audit Test: Math-03 (RoPE Position Overflow)
 * Verifies RoPE handles very large position integers without math domain errors
 * in sin/cos. POS > 2^24 can cause precision loss in float representations of
 * angles, but should not crash or return NaN.
 */
static void aud_math_rope_large_pos(void) {
  int head_dim = 128;
  float *q = tn_aligned_alloc(head_dim * sizeof(float), TN_SIMD_ALIGN);
  float *k = tn_aligned_alloc(head_dim * sizeof(float), TN_SIMD_ALIGN);
  float *freq = tn_aligned_alloc((head_dim / 2) * sizeof(float), TN_SIMD_ALIGN);

  for (int i = 0; i < head_dim; i++) {
    q[i] = 1.0f;
    k[i] = 1.0f;
  }

  rope_precompute_freqs(freq, head_dim, 10000.0f);

  /* Huge position (e.g. 100 million tokens) */
  int huge_pos = 100000000;
  float no_yarn_corr[2] = {0.0f, 0.0f};
  apply_rope(q, k, freq, head_dim, huge_pos, 1, 1, 1.0f, 0.0f, 1.0f,
             no_yarn_corr);

  for (int i = 0; i < head_dim; i++) {
    TEST_ASSERT(!isnan(q[i]), "RoPE Q is not NaN for very large pos");
    TEST_ASSERT(!isnan(k[i]), "RoPE K is not NaN for very large pos");
  }

  tn_aligned_free(q);
  tn_aligned_free(k);
  tn_aligned_free(freq);
}

int main(void) {
  tn_simd_init();
  RUN_TEST(aud_math_softmax_extreme);
  RUN_TEST(aud_math_rmsnorm_zero);
  RUN_TEST(aud_math_rope_large_pos);
  TEST_SUMMARY();
}
